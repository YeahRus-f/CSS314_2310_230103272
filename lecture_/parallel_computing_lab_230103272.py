#!/usr/bin/env python3
# Performance Evaluation and Optimization in Parallel Computing
# Solo Python Systems Lab — Colab/local executable reference implementation
# Student: Yerassyl Orazgali | ID: 230103272
#
# IMPORTANT:
# - Uses only Python standard-library modules.
# - Run this as a .py file (not by pasting multiprocessing code into a notebook cell).
# - QUICK_MODE=True is a validation-sized run. Set QUICK_MODE=False for the assignment
#   workload sizes; the full run can take a very long time, especially Phase 2–4.
# - Colab is not a substitute for the assignment's required offline local-laptop run.

import os
import sys
import math
import time
import json
import math
import ctypes
import platform
import statistics
import multiprocessing as mp
from multiprocessing import shared_memory
from dataclasses import dataclass
from array import array

STUDENT_NAME = "Yerassyl Orazgali"
STUDENT_ID = 230103272
QUICK_MODE = True

# Exact assignment sizes when QUICK_MODE=False.
if QUICK_MODE:
    FS_M = 250_000
    STRONG_N = 40_000
    WEAK_N0 = 8_000
    ROOFLINE_BYTES = 8 * 1024 * 1024
    FMA_FLOPS_PER_WORKER = 300_000
    GEMM_N = 96
    SCHED_N = 1_200
    SCHED_K_DIVISOR = 600
    WARMUPS = 1
    TIMED_RUNS = 3
else:
    FS_M = 24_000_000
    STRONG_N = 1_600_000
    WEAK_N0 = 250_000
    ROOFLINE_BYTES = 96_000_000
    FMA_FLOPS_PER_WORKER = 80_000_000
    GEMM_N = 384
    SCHED_N = 12_000
    SCHED_K_DIVISOR = 60
    WARMUPS = 2
    TIMED_RUNS = 7

PLOG = max(1, mp.cpu_count())
# Physical-core count is not exposed consistently by the standard library.
# Set PHYSICAL_CORES manually if you know it; otherwise this is an estimate.
P_PHYS_ENV = os.environ.get("PHYSICAL_CORES")
PPHYS = max(1, min(int(P_PHYS_ENV), PLOG)) if P_PHYS_ENV else max(1, PLOG // 2)

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float

def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def run_benchmark(fn, warmup_runs=WARMUPS, timed_runs=TIMED_RUNS):
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    n = len(samples)
    med = statistics.median(samples)
    q25 = samples[n // 4]
    q75 = samples[(3 * n) // 4]
    return BenchStats(samples[0], med, q75 - q25, samples[-1])

def verify_tag(task_id, seed, primary, secondary):
    payload = f"{task_id}|{seed}|{PLOG}|{primary:.4f}|{secondary:.4f}"
    return f"VERIFY_HASH [{task_id}]: {fnv1a_hash(payload):016X} (P_log={PLOG})"

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)

def hardware_info():
    section("PHASE 0 — HARDWARE TOPOLOGY AUDIT")
    print(f"Student: {STUDENT_NAME} | Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical processors (mp.cpu_count): {PLOG}")
    print(f"Physical cores: {PPHYS} (estimate unless PHYSICAL_CORES env var is set)")
    print("Cache sizes / cache-line size: not reliably exposed by Python standard library.")
    print("Record these from your operating system's hardware tools if required.")
    t0 = time.perf_counter_ns()
    reads = 200_000 if QUICK_MODE else 2_000_000
    for _ in range(reads):
        time.perf_counter_ns()
    avg_ns = (time.perf_counter_ns() - t0) / reads
    print(f"Average perf_counter_ns() call latency: {avg_ns:.2f} ns")
    print(verify_tag("PHASE0", STUDENT_ID, avg_ns, PLOG))
    return avg_ns

# ------------------------------ PHASE 1 ------------------------------------
def fs_worker(shm_name, slot_idx, iters, n_words):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * n_words).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr
    shm.close()

def run_false_sharing(p, stride, total_iters):
    n_words = max(512, (p - 1) * stride + 1)
    shm = shared_memory.SharedMemory(create=True, size=n_words * 8)
    try:
        slots = (ctypes.c_uint64 * n_words).from_buffer(shm.buf)
        for i in range(n_words):
            slots[i] = i + 1
        del slots
        base, rem = divmod(total_iters, p)
        procs = []
        for t in range(p):
            iters = base + (1 if t < rem else 0)
            proc = mp.Process(target=fs_worker, args=(shm.name, t * stride, iters, n_words))
            procs.append(proc)
        t0 = time.perf_counter_ns()
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"False-sharing worker exited with code {proc.exitcode}")
        return (time.perf_counter_ns() - t0) / 1e6
    finally:
        shm.close()
        shm.unlink()

def phase1():
    section("PHASE 1 — FALSE SHARING & KARp-FLATT")
    print(f"Iterations total: {FS_M:,}; physical-core estimate: {PPHYS}")
    strides = [1, 2, 4, 8, 16, 32]
    stride_results = {}
    for stride in strides:
        stat = run_benchmark(lambda s=stride: run_false_sharing(PPHYS, s, FS_M))
        throughput = (FS_M / 1e6) / (stat.median_ms / 1000.0)
        stride_results[stride] = stat.median_ms
        print(f"stride={stride:2d} | separation={stride*8:3d} B | "
              f"median={stat.median_ms:10.3f} ms | IQR={stat.iqr_ms:9.3f} ms | "
              f"throughput={throughput:8.3f} Mops/s")
    scaling = {}
    process_counts = sorted(set([1, 2, 4, 8, PPHYS, PLOG]))
    process_counts = [p for p in process_counts if 1 <= p <= PLOG]
    for stride in (1, 16):
        print(f"\nProcess scaling, stride={stride}")
        t1 = None
        scaling[stride] = {}
        for p in process_counts:
            stat = run_benchmark(lambda pp=p, ss=stride: run_false_sharing(pp, ss, FS_M))
            if p == 1:
                t1 = stat.median_ms
            speedup = t1 / stat.median_ms if t1 else 1.0
            efficiency = speedup / p
            kf = ((1.0 / speedup) - (1.0 / p)) / (1.0 - (1.0 / p)) if p > 1 else float("nan")
            scaling[stride][p] = (stat.median_ms, speedup, efficiency, kf)
            print(f"p={p:2d} T={stat.median_ms:10.3f} ms S={speedup:7.3f} "
                  f"E={efficiency:7.3f} e(p)={kf:9.5f}")
    t1 = stride_results[1]
    t16 = stride_results[16]
    print("\nTable 1A: stride, bytes, median ms, throughput can be copied from lines above.")
    print("Table 1B: process scaling values are printed above.")
    print(verify_tag("PHASE1", STUDENT_ID, t1, t16))
    return stride_results, scaling

# ------------------------------ PHASE 2 ------------------------------------
def stage_a(n):
    total = 0.0
    limit = max(1, int(n * 0.05))
    for i in range(limit):
        x = (i + 1) * 0.00001
        for _ in range(6):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        total += x
    return total

def stage_b_range(args):
    start, end = args
    total = 0.0
    for i in range(start, end):
        x = (i + 1) * 0.000001
        for j in range(35):
            x = x * 1.0000001 + math.sin(x + j * 0.00001) * 0.000001
        total += x
    return total

def parallel_pipeline(pool, n, p):
    stage_a(n)
    chunk = (n + p - 1) // p
    ranges = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    return sum(pool.map(stage_b_range, ranges))

def phase2():
    section("PHASE 2 — STRONG / WEAK SCALING")
    ps = list(range(1, PLOG + 1))
    strong = {}
    # Each worker pool is created before timing and reused for all trials at that p.
    for p in ps:
        with mp.Pool(processes=p) as subpool:
            stat = run_benchmark(lambda pp=p, pl=subpool: parallel_pipeline(pl, STRONG_N, pp))
            strong[p] = stat.median_ms
            print(f"Strong p={p:2d} N={STRONG_N:,} T={stat.median_ms:.3f} ms")
    t1 = strong[1]
    print("\nStrong scaling metrics:")
    for p in ps:
        s = t1 / strong[p]
        print(f"p={p:2d} T={strong[p]:10.3f} ms S={s:7.3f} E={s/p:7.3f}")
    stage_a_time = run_benchmark(lambda: stage_a(STRONG_N)).median_ms
    serial_fraction = min(1.0, stage_a_time / max(t1, 1e-12))
    amdahl_inf = 1.0 / max(serial_fraction, 1e-12)
    amdahl_plog = 1.0 / (serial_fraction + (1-serial_fraction)/PLOG)
    print(f"\nStage A serial time={stage_a_time:.3f} ms; estimated serial fraction={serial_fraction:.5f}")
    print(f"Amdahl S(infinity)={amdahl_inf:.3f}; Amdahl S(P_log)={amdahl_plog:.3f}")
    weak = {}
    weak_a = {}
    for p in ps:
        n = p * WEAK_N0
        with mp.Pool(processes=p) as pool:
            stat = run_benchmark(lambda pp=p, nn=n, pl=pool: parallel_pipeline(pl, nn, pp))
        weak[p] = stat.median_ms
        weak_a[p] = run_benchmark(lambda nn=n: stage_a(nn)).median_ms
        alpha = min(1.0, weak_a[p] / max(weak[p], 1e-12))
        gust = p - alpha * (p - 1)
        efficiency = weak[1] / weak[p] if p == 1 else weak[1] / weak[p]
        print(f"Weak p={p:2d} N={n:,} T={weak[p]:10.3f} ms "
              f"Eweak={efficiency:7.3f} alpha={alpha:.5f} Gustafson={gust:.3f}")
    observed = t1 / strong[PLOG]
    weak_alpha = min(1.0, weak_a[PLOG] / max(weak[PLOG], 1e-12))
    gust_log = PLOG - weak_alpha * (PLOG - 1)
    print(verify_tag("PHASE2", STUDENT_ID, observed, gust_log))
    return strong, weak

# ------------------------------ PHASE 3A -----------------------------------
def copy_worker(shm_a_name, shm_b_name, start, end):
    a = shared_memory.SharedMemory(name=shm_a_name)
    b = shared_memory.SharedMemory(name=shm_b_name)
    va = memoryview(a.buf)[start:end]
    vb = memoryview(b.buf)[start:end]
    va[:] = vb
    del va, vb
    a.close()
    b.close()

def fma_worker(flops):
    loops = max(1, flops // 32)
    vals = [1.000001 + i * 0.000001 for i in range(16)]
    a, b = 1.00000001, 0.0000001
    for _ in range(loops):
        for j in range(16):
            vals[j] = vals[j] * a + b
    return sum(vals)

def phase3a():
    section("PHASE 3A — EMPIRICAL ROOFLINE")
    size = ROOFLINE_BYTES
    a = shared_memory.SharedMemory(create=True, size=size)
    b = shared_memory.SharedMemory(create=True, size=size)
    try:
        b.buf[:] = b"\x01" * size
        chunk = (size + PPHYS - 1) // PPHYS
        def copy_all():
            jobs = []
            for i in range(PPHYS):
                start = i * chunk
                end = min(size, start + chunk)
                if start < end:
                    jobs.append((a.name, b.name, start, end))
            with mp.Pool(PPHYS) as pool:
                pool.starmap(copy_worker, jobs)
        copy_stat = run_benchmark(copy_all, warmup_runs=1, timed_runs=3)
        bw = (2 * size) / (copy_stat.median_ms / 1000) / 1e9
    finally:
        a.close(); a.unlink()
        b.close(); b.unlink()
    with mp.Pool(PPHYS) as pool:
        fma_stat = run_benchmark(lambda: pool.map(fma_worker, [FMA_FLOPS_PER_WORKER]*PPHYS),
                                 warmup_runs=1, timed_runs=3)
    total_flops = FMA_FLOPS_PER_WORKER * PPHYS
    # The assignment's "FLOPs per worker" wording is ambiguous about loop accounting.
    # This estimate counts 2 FLOPs per multiply-add, per scalar update.
    actual_flops = max(1, (FMA_FLOPS_PER_WORKER // 2) * 16 * 2 * PPHYS)
    p_gflops = actual_flops / (fma_stat.median_ms / 1000) / 1e9
    ridge = p_gflops / bw if bw else float("inf")
    print(f"Memory copy median: {copy_stat.median_ms:.3f} ms")
    print(f"Sustained bandwidth (read+write): {bw:.4f} GB/s")
    print(f"FMA benchmark median: {fma_stat.median_ms:.3f} ms")
    print(f"Estimated compute ceiling: {p_gflops:.4f} GFLOP/s")
    print(f"Empirical ridge point: {ridge:.4f} FLOP/Byte")
    return bw, p_gflops, ridge

# ------------------------------ PHASE 3B -----------------------------------
def gemm_naive_worker(args):
    A, B, n, i0, i1 = args
    C = [0.0] * ((i1-i0)*n)
    for i in range(i0, i1):
        row = i*n
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[row+k] * B[k*n+j]
            C[(i-i0)*n+j] = s
    return i0, C

def gemm_reordered_worker(args):
    A, B, n, i0, i1 = args
    C = [0.0] * ((i1-i0)*n)
    for i in range(i0, i1):
        ci = (i-i0)*n
        ai = i*n
        for k in range(n):
            aik = A[ai+k]
            bk = k*n
            for j in range(n):
                C[ci+j] += aik * B[bk+j]
    return i0, C

def gemm_block_worker(args):
    A, B, n, i0, i1, bs = args
    C = [0.0] * ((i1-i0)*n)
    for ii in range(i0, i1, bs):
        iend = min(i1, ii+bs)
        for kk in range(0, n, bs):
            kend = min(n, kk+bs)
            for jj in range(0, n, bs):
                jend = min(n, jj+bs)
                for i in range(ii, iend):
                    ci = (i-i0)*n
                    ai = i*n
                    for k in range(kk, kend):
                        aik = A[ai+k]
                        bk = k*n
                        for j in range(jj, jend):
                            C[ci+j] += aik * B[bk+j]
    return i0, C

def run_gemm(kind, A, B, n, bs=None):
    ranges = []
    rows_per = (n + PPHYS - 1)//PPHYS
    for i in range(0, n, rows_per):
        j = min(n, i+rows_per)
        ranges.append((A, B, n, i, j) if kind != "blocked"
                      else (A, B, n, i, j, bs))
    worker = gemm_naive_worker if kind == "naive" else (
        gemm_reordered_worker if kind == "reordered" else gemm_block_worker)
    with mp.Pool(PPHYS) as pool:
        parts = pool.map(worker, ranges)
    C = [0.0]*(n*n)
    for i0, part in parts:
        C[i0*n:(i0*n)+len(part)] = part
    return C

def phase3b():
    section("PHASE 3B — GEMM OPTIMIZATION & TILE SWEEP")
    n = GEMM_N
    # Deterministic input generated without third-party packages.
    A = [((i*17 + STUDENT_ID) % 101)/101.0 for i in range(n*n)]
    B = [((i*29 + STUDENT_ID) % 103)/103.0 for i in range(n*n)]
    results = {}
    for kind in ("naive", "reordered"):
        stat = run_benchmark(lambda k=kind: run_gemm(k, A, B, n),
                             warmup_runs=1, timed_runs=2 if QUICK_MODE else 3)
        gflops = (2*n**3)/(stat.median_ms*1e6)
        results[kind] = (stat.median_ms, gflops)
        print(f"{kind:10s} median={stat.median_ms:.3f} ms GFLOP/s={gflops:.5f}")
    reference = run_gemm("reordered", A, B, n)
    tile_sizes = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    best = (-1.0, None, None)
    for bs in tile_sizes:
        stat = run_benchmark(lambda bsize=bs: run_gemm("blocked", A, B, n, bsize),
                             warmup_runs=1, timed_runs=2 if QUICK_MODE else 3)
        c = run_gemm("blocked", A, B, n, bs)
        max_err = max(abs(x-y) for x, y in zip(c, reference))
        gflops = (2*n**3)/(stat.median_ms*1e6)
        footprint = 3*bs*bs*8
        print(f"BS={bs:3d} median={stat.median_ms:10.3f} ms "
              f"GFLOP/s={gflops:9.5f} tile_footprint={footprint:8d} B "
              f"max_error={max_err:.3e}")
        if gflops > best[0]:
            best = (gflops, bs, stat.median_ms)
    print(f"Best tile: BS*={best[1]}, throughput={best[0]:.5f} GFLOP/s")
    print(verify_tag("PHASE3", STUDENT_ID, results["naive"][1], best[0]))
    return results, best

# ------------------------------ PHASE 4 ------------------------------------
def item_work(i):
    k = 1 + int((i / SCHED_K_DIVISOR)**2)
    x = (i + 1) * 0.0001
    acc = 0.0
    for j in range(k):
        x = math.sin(x + j*0.000001)
        x = math.cos(x) + math.sqrt(abs(x) + 1.0)
        acc += x
    return acc

def worker_block(start, end):
    t0 = time.perf_counter_ns()
    s = 0.0
    for i in range(start, end):
        s += item_work(i)
    return s, (time.perf_counter_ns()-t0)/1e6

def worker_cyclic(worker_id, p, n):
    t0 = time.perf_counter_ns()
    s = 0.0
    for i in range(worker_id, n, p):
        s += item_work(i)
    return s, (time.perf_counter_ns()-t0)/1e6

def dynamic_worker(shm_name, lock, counter, chunk, n, times, idx):
    t0 = time.perf_counter_ns()
    s = 0.0
    while True:
        with lock:
            start = counter.value
            counter.value += chunk
        if start >= n:
            break
        for i in range(start, min(n, start+chunk)):
            s += item_work(i)
    times[idx] = (time.perf_counter_ns()-t0)/1e6
    return s

def run_schedule(strategy, p, n, chunk=1):
    with mp.Manager() as manager:
        times = manager.list([0.0]*p)
        if strategy == "block":
            ranges = []
            size = (n+p-1)//p
            for t in range(p):
                ranges.append((t*size, min(n, (t+1)*size)))
            t0 = time.perf_counter_ns()
            with mp.Pool(p) as pool:
                results = pool.map(worker_block, ranges)
            wall = (time.perf_counter_ns()-t0)/1e6
            worker_times = [r[1] for r in results]
        elif strategy == "cyclic":
            t0 = time.perf_counter_ns()
            with mp.Pool(p) as pool:
                results = pool.starmap(worker_cyclic, [(t,p,n) for t in range(p)])
            wall = (time.perf_counter_ns()-t0)/1e6
            worker_times = [r[1] for r in results]
        else:
            counter = mp.Value('i', 0)
            lock = mp.Lock()
            t0 = time.perf_counter_ns()
            procs = [mp.Process(target=dynamic_worker,
                                args=("unused", lock, counter, chunk, n, times, t))
                     for t in range(p)]
            for proc in procs: proc.start()
            for proc in procs:
                proc.join()
                if proc.exitcode != 0:
                    raise RuntimeError("Dynamic scheduling worker failed")
            wall = (time.perf_counter_ns()-t0)/1e6
            worker_times = list(times)
        avg = sum(worker_times)/len(worker_times) if worker_times else 0.0
        imbalance = max(worker_times)/avg if avg else float("nan")
        return wall, imbalance, worker_times

def phase4():
    section("PHASE 4 — SKEWED WORKLOAD SCHEDULING")
    p = PPHYS
    n = SCHED_N
    print(f"N={n:,}; workers={p}; K(i)=1+floor((i/{SCHED_K_DIVISOR})^2)")
    block = run_schedule("block", p, n)
    cyclic = run_schedule("cyclic", p, n)
    print(f"Static contiguous: wall={block[0]:.3f} ms lambda={block[1]:.3f}")
    print(f"Static cyclic:     wall={cyclic[0]:.3f} ms lambda={cyclic[1]:.3f}")
    dynamic = {}
    for chunk in (1, 4, 16, 64, 256, 1024):
        result = run_schedule("dynamic", p, n, chunk)
        dynamic[chunk] = result
        print(f"Dynamic C={chunk:4d}: wall={result[0]:.3f} ms lambda={result[1]:.3f}")
    best_chunk = min(dynamic, key=lambda c: dynamic[c][0])
    print(f"Best observed chunk C*={best_chunk}")
    print(verify_tag("PHASE4", STUDENT_ID, block[0], dynamic[best_chunk][0]))
    return block, cyclic, dynamic, best_chunk

def write_report(results):
    filename = f"lab_report_{STUDENT_ID}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Parallel Computing Lab Report\n\n")
        f.write(f"- Student: {STUDENT_NAME}\n- Student ID: {STUDENT_ID}\n")
        f.write(f"- Runtime environment: {platform.platform()}\n")
        f.write(f"- Python: {sys.version.split()[0]}\n")
        f.write(f"- Logical processors: {PLOG}\n- Physical-core estimate: {PPHYS}\n")
        f.write(f"- QUICK_MODE: {QUICK_MODE}\n\n")
        f.write("## Raw execution output\n\n")
        f.write("The complete console output should be saved alongside this report. ")
        f.write("Copy the actual VERIFY_HASH lines and measured tables below after execution.\n\n")
        f.write("## Results returned by program\n\n")
        f.write("```text\n")
        f.write(json.dumps(results, indent=2, default=str))
        f.write("\n```\n")
        f.write("\n## Analysis\n\n")
        f.write("Interpret all values using the actual hardware measurements. ")
        f.write("Do not infer cache-line size or physical core count from estimates alone.\n")
    print(f"\nReport skeleton saved: {filename}")

def main():
    mp.freeze_support()
    print("Performance Evaluation and Optimization in Parallel Computing")
    print(f"{STUDENT_NAME} | ID {STUDENT_ID} | QUICK_MODE={QUICK_MODE}")
    print(f"Logical cores={PLOG}; physical-core estimate={PPHYS}")
    print("For a full assignment-sized run, edit QUICK_MODE=False at the top.")
    results = {}
    results["phase0"] = hardware_info()
    results["phase1"] = phase1()
    results["phase2"] = phase2()
    results["phase3a"] = phase3a()
    results["phase3b"] = phase3b()
    results["phase4"] = phase4()
    write_report(results)
    print("\nALL PHASES FINISHED.")
    print("Note: this program is a runnable implementation, not proof of the required 120-minute")
    print("offline local execution. Verify physical cores/cache sizes using your OS tools.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
