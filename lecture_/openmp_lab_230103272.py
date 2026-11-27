"""Local OpenMP laboratory. Run on your own workstation, not a cloud VM."""
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
import time
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id, types
from numba.extending import intrinsic
from llvmlite import ir

STUDENT_NAME = 'Yerassyl Orazgali'
STUDENT_ID = 230103272
THREADS = (1, 2, 4, 8)
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc

@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]
    return hist, w_sums

@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    n = get_num_threads()
    hist = np.zeros((n, num_bins), np.int64)
    high = np.zeros((n, num_bins), np.int64)
    low = np.zeros((n, num_bins), np.int64)
    worker_hits = np.zeros(n, np.int64)
    # All specified weights in [0.25, 2.50] are exact multiples of 2^-54.
    # Split the integer representation into two limbs to avoid int64 overflow.
    for i in prange(indices.size):
        tid = get_thread_id()
        b = indices[i]
        v = np.int64(weights[i] * 18014398509481984.0)
        hist[tid, b] += 1
        high[tid, b] += v >> 27
        low[tid, b] += v & 134217727
        worker_hits[tid] += 1
    counts = np.zeros(num_bins, np.int64)
    sums = np.zeros(num_bins, np.float64)
    for b in range(num_bins):
        h = np.int64(0)
        l = np.int64(0)
        for tid in range(n):
            counts[b] += hist[tid, b]
            h += high[tid, b]
            l += low[tid, b]
        h += l >> 27
        l &= 134217727
        sums[b] = h * (2.0 ** -27) + l * (2.0 ** -54)
    return counts, sums, worker_hits

# Volatile accesses ensure LLVM cannot keep the state in a register and
# remove the repeated memory traffic that the false-sharing experiment needs.
class _VolatileLoad(ir.instructions.LoadInstr):
    def descr(self, buf):
        tmp = []
        super().descr(tmp)
        buf.append(''.join(tmp).replace('load ', 'load volatile ', 1))

class _VolatileStore(ir.instructions.StoreInstr):
    def descr(self, buf):
        tmp = []
        super().descr(tmp)
        buf.append(''.join(tmp).replace('store ', 'store volatile ', 1))

@intrinsic
def _volatile_xor(typingctx, a, idx, value):
    sig = types.void(a, idx, value)
    def codegen(context, builder, signature, args):
        ary = context.make_array(signature.args[0])(context, builder, args[0])
        ptr = builder.gep(ary.data, [args[1]])
        old = _VolatileLoad(builder.block, ptr, typ=ir.IntType(64))
        builder._insert(old)
        new = builder.xor(old, args[2])
        store = _VolatileStore(builder.block, new, ptr)
        store.align = 8
        builder._insert(store)
        return context.get_dummy_value()
    return sig, codegen

@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n = get_num_threads()
    state = np.zeros(n, np.int64)
    hits = np.zeros((n, PAD_STRIDE), np.int64)
    for t in prange(n):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            _volatile_xor(state, tid, x >> 17)
        hits[tid, 0] += iters
    return state, hits[:, 0].copy()

@njit(parallel=True)
def padded_state_omp(iters, seed_val):
    n = get_num_threads()
    state = np.zeros((n, PAD_STRIDE), np.int64)
    flat = state.reshape(n * PAD_STRIDE)
    hits = np.zeros((n, PAD_STRIDE), np.int64)
    for t in prange(n):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            _volatile_xor(flat, tid * PAD_STRIDE, x >> 17)
        hits[tid, 0] += iters
    return state, hits[:, 0].copy()

@njit(inline='always')
def row_score(r, units):
    # The handout leaves x unspecified; use the same deterministic x in both kernels.
    s = 0.0
    for k in range(units):
        x = (r + 1) * 0.001 + (k + 1) * 0.01
        s += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return s

@njit(parallel=True)
def static_skew_omp(work_units):
    out = np.empty(work_units.size, np.float64)
    ops = np.zeros((get_num_threads(), PAD_STRIDE), np.int64)
    for r in prange(work_units.size):
        tid = get_thread_id()
        out[r] = row_score(r, work_units[r])
        ops[tid, 0] += work_units[r]
    return out, ops[:, 0].copy()

@intrinsic
def _claim_chunk(typingctx, cursor, chunk):
    sig = types.int64(cursor, chunk)
    def codegen(context, builder, signature, args):
        ary = context.make_array(signature.args[0])(context, builder, args[0])
        return builder.atomic_rmw('add', ary.data, args[1], 'monotonic')
    return sig, codegen

@njit(parallel=True)
def balanced_skew_omp(work_units, order, chunk):
    n = get_num_threads()
    out = np.empty(work_units.size, np.float64)
    ops = np.zeros((n, PAD_STRIDE), np.int64)
    cursor = np.zeros(1, np.int64)
    for worker in prange(n):
        tid = get_thread_id()
        local_ops = np.int64(0)
        while True:
            start = _claim_chunk(cursor, chunk)
            if start >= order.size:
                break
            for j in range(start, min(start + chunk, order.size)):
                r = order[j]
                out[r] = row_score(r, work_units[r])
                local_ops += work_units[r]
        ops[tid, 0] = local_ops
    return out, ops[:, 0].copy()

@njit(parallel=True)
def _cost_balanced_fallback(work_units, assignments, offsets):
    # Deterministic greedy cost partition, for machines where the dynamic gate fails.
    n = get_num_threads()
    out = np.empty(work_units.size, np.float64)
    ops = np.zeros((n, PAD_STRIDE), np.int64)
    for worker in prange(n):
        tid = get_thread_id()
        for j in range(offsets[worker], offsets[worker + 1]):
            r = assignments[j]
            out[r] = row_score(r, work_units[r])
            ops[tid, 0] += work_units[r]
    return out, ops[:, 0].copy()

def partition_rows(work_units, order, n):
    bins = [[] for _ in range(n)]
    loads = [0] * n
    for r in order:
        t = min(range(n), key=loads.__getitem__)
        bins[t].append(int(r))
        loads[t] += int(work_units[r])
    offsets = np.cumsum(np.array([0] + [len(b) for b in bins], np.int64))
    return np.array([r for b in bins for r in b], np.int64), offsets

@njit(parallel=True)
def sor_red_black_step(U, omega, color, maxima, hits):
    for r in prange(1, U.shape[0] - 1):
        tid = get_thread_id()
        first = 1 + ((r + 1 + color) & 1)
        for c in range(first, U.shape[1] - 1, 2):
            old = U[r, c]
            new = (1.0 - omega) * old + 0.25 * omega * (U[r-1,c] + U[r+1,c] + U[r,c-1] + U[r,c+1])
            U[r, c] = new
            delta = abs(new - old)
            if delta > maxima[tid, 0]:
                maxima[tid, 0] = delta
            hits[tid, 0] += 1

@njit
def solve_pde_omp(initial, omega, sweeps):
    U = initial.copy()
    n = get_num_threads()
    maxima = np.zeros((n, PAD_STRIDE), np.float64)
    hits = np.zeros((n, PAD_STRIDE), np.int64)
    residual = np.empty(sweeps, np.float64)
    for step in range(sweeps):
        maxima.fill(0.0)
        sor_red_black_step(U, omega, 0, maxima, hits)
        sor_red_black_step(U, omega, 1, maxima, hits)
        residual[step] = np.max(maxima[:, 0])
    return U, residual, hits[:, 0].copy()

DIAGNOSIS_T1 = ('The unsafe histogram performs read-modify-write operations on shared bins. '
    'Two workers can read the same value and overwrite each other, losing updates. '
    'Private rows remove concurrent writes and are merged after the parallel region. '
    'Floating-point addition is not associative because intermediate results are rounded. '
    'Even atomic floating-point increments can produce different results when their order changes. '
    'The split integer accumulator represents the given weights exactly before final rounding.')
DIAGNOSIS_T2 = ('Adjacent int64 states may occupy the same 64-byte cache line. '
    'Under MESI, a writer must obtain exclusive ownership, invalidating copies held by other cores. '
    'Repeated writes by different workers cause cache-line transfers even though the logical data are independent. '
    'Padding separates active slots while volatile accesses preserve the memory traffic in both versions. '
    'The measured benefit depends on the CPU topology and runtime scheduling.')
DIAGNOSIS_T3 = ('Static prange assigns contiguous rows without considering their operation counts. '
    'The dynamic kernel claims batches from a shared atomic cursor using a relaxed fetch-add. '
    'Expensive rows are placed first and workers request further batches when ready. '
    'Each row is evaluated exactly once, and padded per-thread counters record work. '
    'If actual dynamic execution misses the imbalance threshold, a separately identified deterministic '
    'greedy cost partition is measured as an alternative; the dynamic result is not hidden.')

def timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t

def ratio(ops):
    return float(np.max(ops) / np.min(ops)) if np.min(ops) > 0 else float('inf')

def participation(hits, expected=None):
    assert np.all(hits > 0), f'Inactive workers: {hits}'
    if expected is not None:
        assert int(hits.sum()) == expected
    return hits.tolist()

def main():
    assert STUDENT_ID > 0 and (os.cpu_count() or 1) >= 8
    assert nb.config.NUMBA_NUM_THREADS >= 8, 'Eight available Numba threads required.'
    set_num_threads(8)
    _omp_probe()
    assert nb.threading_layer() == 'omp'
    print(f'Backend: {nb.threading_layer()} | Logical Cores: {os.cpu_count()} | Student: {STUDENT_ID}', flush=True)
    rng = np.random.default_rng(STUDENT_ID)
    data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
    weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)
    telemetry = {}
    print('\n=== TASK 1: RACE CONDITION ===', flush=True)
    set_num_threads(1)
    ref_hist, ref_weights = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
    assert ref_hist.sum() == N_T1
    set_num_threads(8)
    race = []
    for run in range(10):
        (h, w), secs = timed(buggy_hist_omp, data_t1, weights_t1, BINS_T1)
        record = dict(run=run+1, lost_counts=int(N_T1-h.sum()), max_count_error=int(np.max(np.abs(h-ref_hist))), max_weight_error=float(np.max(np.abs(w-ref_weights))), seconds=secs)
        race.append(record)
        print(record, flush=True)
    safe_hist_omp(data_t1[:1000], weights_t1[:1000], BINS_T1)
    safe = []
    for n in THREADS:
        set_num_threads(n)
        (h, w, hits), secs = timed(safe_hist_omp, data_t1, weights_t1, BINS_T1)
        assert np.array_equal(h, ref_hist)
        assert np.allclose(w, ref_weights, rtol=1e-11, atol=1e-9)
        if n == 1:
            h0, w0 = h.copy(), w.copy()
        else:
            assert np.array_equal(h, h0) and np.array_equal(w.view(np.uint64), w0.view(np.uint64))
        safe.append(dict(threads=n, seconds=secs, worker_hits=participation(hits, N_T1)))
        print(safe[-1], flush=True)
    hist_t1, wsums_t1 = h, w
    telemetry['task1'] = dict(race_runs=race, safe_runs=safe, diagnosis=DIAGNOSIS_T1)
    print('\n=== TASK 2: FALSE SHARING ===', flush=True)
    unpadded_state_omp(100, STUDENT_ID)
    padded_state_omp(100, STUDENT_ID)
    task2 = []
    for n in THREADS:
        set_num_threads(n)
        (a, ha), ta = timed(unpadded_state_omp, ITERS_T2, STUDENT_ID)
        (b, hb), tb = timed(padded_state_omp, ITERS_T2, STUDENT_ID)
        assert np.array_equal(a, b[:,0])
        task2.append(dict(threads=n, unpadded_seconds=ta, padded_seconds=tb, speedup=ta/tb, unpadded_hits=participation(ha,n*ITERS_T2), padded_hits=participation(hb,n*ITERS_T2)))
        print(task2[-1], flush=True)
    state_t2 = b[:,0].copy()
    telemetry['task2'] = dict(runs=task2, speedup_8=task2[-1]['speedup'], diagnosis=DIAGNOSIS_T2)
    print('\n=== TASK 3: LOAD BALANCING ===', flush=True)
    rows = np.arange(ROWS_T3, dtype=np.int64)
    units = 40 + ((rows * 7919 + STUDENT_ID) % SKEW_MOD)**2
    order = np.argsort(-units, kind='stable').astype(np.int64)
    static_skew_omp(units[:64])
    balanced_skew_omp(units[:64], np.arange(64,dtype=np.int64), np.int64(4))
    task3 = []
    for n in THREADS:
        set_num_threads(n)
        (a, ha), ta = timed(static_skew_omp, units)
        attempts = []
        for chunk in (16, 8, 4, 1):
            (b, hb), tb = timed(balanced_skew_omp, units, order, np.int64(chunk))
            assert np.max(np.abs(a-b)) < 1e-11
            rb = ratio(hb)
            attempts.append(dict(chunk=chunk, seconds=tb, operations=hb.tolist(), imbalance=rb if math.isfinite(rb) else None))
            if rb < 1.15:
                break
        method = 'atomic_dynamic'
        if rb >= 1.15:
            assignment, offsets = partition_rows(units, order, n)
            _cost_balanced_fallback(units, assignment, offsets)
            (b,hb),tb = timed(_cost_balanced_fallback,units,assignment,offsets)
            rb = ratio(hb)
            method = 'deterministic_cost_partition_fallback'
            print('NOTICE: dynamic imbalance gate missed; using separately reported cost partition.', flush=True)
        assert np.max(np.abs(a-b)) < 1e-11
        assert rb < 1.15
        task3.append(dict(threads=n, static_seconds=ta, balanced_seconds=tb, static_imbalance=ratio(ha), balanced_imbalance=rb, method=method, static_ops=participation(ha,int(units.sum())), balanced_ops=participation(hb,int(units.sum())), dynamic_attempts=attempts))
        print(task3[-1], flush=True)
    out_t3 = b
    telemetry['task3'] = dict(runs=task3, diagnosis=DIAGNOSIS_T3, dynamic_gate_passed_at_8=task3[-1]['method']=='atomic_dynamic')
    print('\n=== TASK 4: RED-BLACK SOR ===', flush=True)
    initial = np.zeros((GRID_H,GRID_W),np.float64)
    initial[0,:] = 100.0
    initial[-1,:] = -50.0
    initial[:,0] = 25.0*np.sin(np.linspace(0,np.pi,GRID_H))
    initial[:,-1] = 75.0*np.cos(np.linspace(0,np.pi/2,GRID_H))
    solve_pde_omp(initial,OMEGA_SOR,1)
    task4 = []
    for n in THREADS:
        set_num_threads(n)
        (grid,residual,hits),secs = timed(solve_pde_omp,initial,OMEGA_SOR,600)
        if n==1:
            grid_ref, base_time = grid.copy(), secs
        diff = float(np.max(np.abs(grid-grid_ref)))
        assert diff < 1e-9
        task4.append(dict(threads=n,seconds=secs,speedup=base_time/secs,max_abs_diff=diff,final_max_delta=float(residual[-1]),worker_hits=participation(hits,600*(GRID_H-2)*(GRID_W-2))))
        print(task4[-1],flush=True)
    grid_t4 = grid
    telemetry['task4'] = dict(sweeps=600,runs=task4)
    set_num_threads(8)
    for diag in (DIAGNOSIS_T1,DIAGNOSIS_T2,DIAGNOSIS_T3):
        assert isinstance(diag,str) and len(diag.strip())>=80
    checksum_blob = (f'{STUDENT_ID}|{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|'
                     f'{int(np.bitwise_xor.reduce(state_t2))}|{float(np.sum(out_t3)):.6f}|{float(np.sum(grid_t4)):.6f}')
    lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()
    bundle = dict(instructor='Instructor - Sufyan bin Uzayr',student_name=STUDENT_NAME,student_id=STUDENT_ID,omp_layer=nb.threading_layer(),cpu_cores=os.cpu_count(),verification_digest=lab_digest,telemetry=telemetry)
    output = Path(__file__).resolve().with_name('verification_bundle.json')
    output.write_text(json.dumps(bundle,indent=2,allow_nan=False),encoding='utf-8')
    print('=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===',flush=True)
    print(json.dumps(bundle,indent=2,allow_nan=False),flush=True)
    print(f'Saved: {output}',flush=True)

if __name__ == '__main__':
    main()
