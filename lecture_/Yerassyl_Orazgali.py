# Performance Evaluation and Optimization in Parallel Computing
# Yerassyl Orazgali | Student ID: 230103272
# Google Colab / Python 3 — standard library only
#
# QUICK_MODE=True uses reduced workloads so the full lab can finish in Colab.
# Set QUICK_MODE=False for workloads closer to the PDF (may take a very long time).
# The PDF describes local/offline laptop execution; Colab results are not laptop measurements.

import os
import sys
import time
import math
import ctypes
import hashlib
import platform
import statistics
import multiprocessing as mp
from multiprocessing import shared_memory

STUDENT_NAME = "Yerassyl Orazgali"
STUDENT_ID = 230103272
QUICK_MODE = True

# Colab usually exposes 2 logical CPUs. Avoid oversubscribing it.
PLOG = max(1, mp.cpu_count())
PPHYS = max(1, PLOG // 2)
if PLOG == 2:
    PPHYS = 1
WORKERS = sorted(set([1, min(2, PLOG), PPHYS, PLOG]))
WORKERS = [p for p in WORKERS if p >= 1]

M_FS = 160_000 if QUICK_MODE else 16_000_000
N_STRONG = 60_000 if QUICK_MODE else 1_200_000
N0_WEAK = 10_000 if QUICK_MODE else 200_000
N_GEMM = 64 if QUICK_MODE else 256
N_SKEW = 500 if QUICK_MODE else 8_000
ROOF_BYTES = 4 * 1024 * 1024 if QUICK_MODE else 64 * 1024 * 1024
FMA_ITERS = 250_000 if QUICK_MODE else 3_750_000

MASK64 = (1 << 64) - 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

def fnv1a_hash(s):
    h = FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h

def verify_tag(phase, seed, m1, m2):
    sig = fnv1a_hash(f"{phase}|{seed}|{PLOG}|{m1:.4f}|{m2:.4f}")
    line = f"VERIFY_HASH[{phase}]: {sig:016X} (P_log={PLOG})"
    print(line)
    return line

def timed(fn, repeats=3):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    med = statistics.median(samples)
    q1 = samples[len(samples)//4]
    q3 = samples[(3*len(samples))//4]
    return samples[0], med, q3-q1

# ---------------- PHASE 0 ----------------
def phase0():
    print("\n" + "="*72 + "\nPHASE 0 — HARDWARE TOPOLOGY & TELEMETRY\n" + "="*72)
    calls = 100_000 if QUICK_MODE else 1_000_000
    t0 = time.perf_counter_ns()
    for _ in range(calls):
        time.perf_counter_ns()
    latency = (time.perf_counter_ns() - t0) / calls
    print("Student:", STUDENT_NAME, "| ID:", STUDENT_ID)
    print("OS:", platform.platform())
    print("Python:", sys.version.split()[0])
    print("CPU:", platform.processor() or platform.machine())
    print("Physical-core estimate:", PPHYS)
    print("Logical cores:", PLOG)
    print("Cache line / L1d / L2 / L3: not exposed reliably by this Colab runtime")
    print(f"perf_counter_ns average call latency: {latency:.2f} ns")
    verify_tag("PHASE0", STUDENT_ID, latency, float(PLOG))

# ---------------- PHASE 1: FALSE SHARING ----------------
def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & MASK64
        arr[slot_idx] = acc
    del arr
    shm.close()

def run_fs(p, stride, total):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    for i in range(512):
        arr[i] = i + 1
    del arr
    procs = []
    base, rem = divmod(total, p)
    t0 = time.perf_counter()
    try:
        for w in range(p):
            iters = base + (1 if w < rem else 0)
            proc = mp.Process(target=fs_worker, args=(shm.name, w*stride, iters))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError("False-sharing worker failed")
        return (time.perf_counter() - t0) * 1000
    finally:
        shm.close()
        shm.unlink()

def phase1():
    print("\n" + "="*72 + "\nPHASE 1 — FALSE SHARING & KAR P-FLATT\n" + "="*72)
    strides = [1, 2, 4, 8, 16, 32]
    stride_results = {}
    print("Stride sweep at PPHYS =", PPHYS, "| iterations =", M_FS)
    print("stride | byte offset | median ms | IQR ms | throughput Mops/s")
    for stride in strides:
        vals = [run_fs(PPHYS, stride, M_FS) for _ in range(3)]
        med = statistics.median(vals)
        iqr = max(vals) - min(vals)
        throughput = (M_FS / 1e6) / (med / 1000) if med else 0
        stride_results[stride] = med
        print(f"{stride:>6} | {stride*8:>11} | {med:>9.2f} | {iqr:>6.2f} | {throughput:>17.3f}")
    print("\nProcess scaling (stride 1 vs stride 16)")
    times = {}
    for stride in (1, 16):
        times[stride] = {}
        for p in WORKERS:
            t = statistics.median([run_fs(p, stride, M_FS) for _ in range(2)])
            times[stride][p] = t
        t1 = times[stride][1]
        print(f"\nstride={stride}")
        print("p | time ms | speedup | efficiency | Karp-Flatt e(p)")
        for p, t in times[stride].items():
            speed = t1/t if t else 0
            eff = speed/p
            e = ((1/speed)-(1/p))/(1-(1/p)) if p > 1 and speed else float("nan")
            print(f"{p:>1} | {t:>7.2f} | {speed:>7.3f} | {eff:>10.3f} | {e:>15.5f}")
    verify_tag("PHASE1", STUDENT_ID, times[1][PPHYS], times[16][PPHYS])

# ---------------- PHASE 2: STRONG / WEAK SCALING ----------------
def stage_a(n):
    total = 0.0
    for i in range(max(1, int(n*0.05))):
        x = (i + 1) * 0.00001
        for _ in range(10):
            total += math.sin(x) / math.sqrt(x + 1.0)
    return total

def compute_chunk(args):
    start, end, n = args
    acc = 0.0
    for i in range(start, end):
        x = (i + 1) * 0.000001
        # Repeated arithmetic/trigonometric work to create a CPU workload.
        for j in range(3):
            x = x * 1.0000001 + 0.0000003
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        acc += x
    return acc

def parallel_stage_b(n, p, pool):
    chunk = (n + p - 1)//p
    jobs = [(s, min(n, s+chunk), n) for s in range(0, n, chunk)]
    return sum(pool.map(compute_chunk, jobs))

def phase2():
    print("\n" + "="*72 + "\nPHASE 2 — STRONG & WEAK SCALING\n" + "="*72)
    strong = {}
    stage_a_ms = {}
    for p in WORKERS:
        with mp.Pool(processes=p) as pool:
            # Pool creation is intentionally outside the timed pipeline loop.
            def trial():
                stage_a(N_STRONG)
                parallel_stage_b(N_STRONG, p, pool)
            vals = []
            for _ in range(2):
                t0 = time.perf_counter()
                trial()
                vals.append((time.perf_counter()-t0)*1000)
            strong[p] = statistics.median(vals)
        t0 = time.perf_counter()
        stage_a(N_STRONG)
        stage_a_ms[p] = (time.perf_counter()-t0)*1000
    t1 = strong[1]
    s_fraction = min(1.0, stage_a_ms[1]/t1) if t1 else 1.0
    amdahl_cap = 1/s_fraction if s_fraction else float("inf")
    print(f"Strong scaling, N={N_STRONG}; estimated serial fraction={s_fraction:.4f}; Amdahl cap={amdahl_cap:.3f}")
    print("p | time ms | speedup | efficiency")
    for p in WORKERS:
        sp = t1/strong[p]
        print(f"{p} | {strong[p]:>8.2f} | {sp:>7.3f} | {sp/p:>10.3f}")
    weak = {}
    weak_a = {}
    for p in WORKERS:
        n = p*N0_WEAK
        with mp.Pool(processes=p) as pool:
            t0 = time.perf_counter()
            stage_a(n)
            parallel_stage_b(n, p, pool)
            weak[p] = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter()
        stage_a(n)
        weak_a[p] = (time.perf_counter()-t0)*1000
    print(f"\nWeak scaling, N0={N0_WEAK}")
    print("p | total N | time ms | weak efficiency | Gustafson speedup")
    for p in WORKERS:
        eff = weak[1]/weak[p] if weak[p] else 0
        alpha = weak_a[p]/weak[p] if weak[p] else 1
        gust = p-alpha*(p-1)
        print(f"{p} | {p*N0_WEAK:>7} | {weak[p]:>7.2f} | {eff:>15.3f} | {gust:>17.3f}")
    sorted_ps = sorted(strong)
    if PPHYS in strong:
        next_p = min([p for p in sorted_ps if p > PPHYS], default=PPHYS)
        marginal = (strong[PPHYS]/strong[next_p]) if next_p != PPHYS else 1.0
    else:
        marginal = 1.0
    print(f"Marginal speedup from PPHYS={PPHYS} to next tested worker count: {marginal:.3f}")
    verify_tag("PHASE2", STUDENT_ID, t1/strong[max(strong)], weak[1]/weak[max(weak)])

# ---------------- PHASE 3: ROOFLINE + GEMM ----------------
def copy_worker(shm_a, shm_b, start, end):
    a = shared_memory.SharedMemory(name=shm_a)
    b = shared_memory.SharedMemory(name=shm_b)
    av = a.buf
    bv = b.buf
    av[start:end] = bv[start:end]
    del av, bv
    a.close()
    b.close()

def fma_worker(iters):
    x = 1.000001
    a,b,c,d,e,f,g,h = 1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8
    for i in range(iters):
        x = x*a+b
        x = x*c+d
        x = x*e+f
        x = x*g+h
    return x

def gemm_worker(args):
    A, B, n, i0, i1, variant, bs = args
    rows = [[0.0]*n for _ in range(i1-i0)]
    if variant == 1:  # i-j-k
        for i in range(i0, i1):
            row = rows[i-i0]
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += A[i*n+k] * B[k*n+j]
                row[j] = acc
    elif variant == 2:  # i-k-j
        for i in range(i0, i1):
            row = rows[i-i0]
            base_i = i*n
            for k in range(n):
                aik = A[base_i+k]
                base_b = k*n
                for j in range(n):
                    row[j] += aik * B[base_b+j]
    else:  # cache blocked i-k-j
        for ii in range(i0, i1, bs):
            i_end = min(i1, ii+bs)
            for kk in range(0, n, bs):
                k_end = min(n, kk+bs)
                for jj in range(0, n, bs):
                    j_end = min(n, jj+bs)
                    for i in range(ii, i_end):
                        row = rows[i-i0]
                        base_i = i*n
                        for k in range(kk, k_end):
                            aik = A[base_i+k]
                            base_b = k*n
                            for j in range(jj, j_end):
                                row[j] += aik * B[base_b+j]
    return i0, rows

def run_gemm(n, p, variant, bs=16):
    # Flat 1D input buffers. Workers return their assigned output rows to the parent.
    A = [((i*7)%17)/17 for i in range(n*n)]
    B = [((i*11)%19)/19 for i in range(n*n)]
    C = [0.0]*(n*n)
    chunk = (n+p-1)//p
    jobs = [(A,B,n,s,min(n,s+chunk),variant,bs) for s in range(0,n,chunk)]
    t0 = time.perf_counter()
    with mp.Pool(p) as pool:
        pieces = pool.map(gemm_worker, jobs)
    for i0, rows in pieces:
        for offset, row in enumerate(rows):
            C[(i0+offset)*n:(i0+offset+1)*n] = row
    return (time.perf_counter()-t0)*1000, C

def phase3():
    print("\n" + "="*72 + "\nPHASE 3 — EMPIRICAL ROOFLINE & CACHE-BLOCKED GEMM\n" + "="*72)
    # Shared-memory copy bandwidth measurement.
    a = shared_memory.SharedMemory(create=True, size=ROOF_BYTES)
    b = shared_memory.SharedMemory(create=True, size=ROOF_BYTES)
    try:
        a.buf[:] = b'\x01' * ROOF_BYTES
        b.buf[:] = b'\x02' * ROOF_BYTES
        def do_copy():
            procs=[]
            chunk=(ROOF_BYTES+PPHYS-1)//PPHYS
            for w in range(PPHYS):
                s=w*chunk; e=min(ROOF_BYTES,s+chunk)
                q=mp.Process(target=copy_worker,args=(a.name,b.name,s,e))
                q.start(); procs.append(q)
            for q in procs:
                q.join()
        _, copy_ms, _ = timed(do_copy, 2)
        bw = (2*ROOF_BYTES/1e9)/(copy_ms/1000) if copy_ms else 0
    finally:
        a.close(); a.unlink(); b.close(); b.unlink()
    t0=time.perf_counter()
    fma_worker(FMA_ITERS)
    fma_ms=(time.perf_counter()-t0)*1000
    # 8 multiply/add pairs, approximately 16 FLOPs per loop.
    gflops=(FMA_ITERS*16)/(fma_ms/1000)/1e9 if fma_ms else 0
    ridge=gflops/bw if bw else 0
    print(f"Memory copy: {2*ROOF_BYTES/1e6:.1f} MB transferred in {copy_ms:.2f} ms => {bw:.3f} GB/s")
    print(f"Scalar compute estimate: {gflops:.4f} GFLOP/s; ridge point={ridge:.4f} FLOP/byte")
    n=N_GEMM
    print(f"\nGEMM N={n}, workers={PPHYS}")
    results={}
    for variant in (1,2,3):
        bs=16
        ms,C=run_gemm(n,PPHYS,variant,bs)
        results[variant]=(ms,C)
        g=(2*n**3)/(ms/1000)/1e9 if ms else 0
        print(f"Variant {variant}: {ms:.2f} ms | {g:.5f} GFLOP/s")
    # Validate variants against the reordered implementation.
    c2=results[2][1]
    c3=results[3][1]
    err=max(abs(x-y) for x,y in zip(c2,c3))
    print(f"Max |C_variant2 - C_variant3| = {err:.3e}")
    tile_sweep=[]
    for bs in ([8,16,32] if QUICK_MODE else [8,16,32,64,96,128]):
        ms,_=run_gemm(n,PPHYS,3,bs)
        tile_sweep.append((bs,ms))
    best_bs,best_ms=min(tile_sweep,key=lambda x:x[1])
    print("Tile sweep (BS, ms):", tile_sweep, "| best BS =",best_bs)
    wtile=3*best_bs*best_bs*8
    intensity=best_bs/12
    print(f"Best tile footprint={wtile} bytes; approximate operational intensity={intensity:.3f} FLOP/byte")
    g1=2*n**3/(results[1][0]/1000)/1e9
    g3=2*n**3/(best_ms/1000)/1e9
    verify_tag("PHASE3", STUDENT_ID, g1, g3)

# ---------------- PHASE 4: DYNAMIC SCHEDULING ----------------
def item_cost(i):
    return 1 + int((i/40)**2)

def do_item(i):
    x=(i+1)*0.001
    for j in range(item_cost(i)):
        x=math.sin(x)+math.cos(x)+math.sqrt(abs(x)+1.0)
    return x

def schedule_worker_static(worker_id, indices, runtime_arr, result_arr):
    t0=time.perf_counter()
    acc=0.0
    for i in indices:
        acc += do_item(i)
    runtime_arr[worker_id]=(time.perf_counter()-t0)
    result_arr[worker_id]=acc

def dynamic_worker(worker_id, counter, lock, chunk, n, runtime_arr, result_arr):
    t0=time.perf_counter()
    acc=0.0
    while True:
        with lock:
            start=counter.value
            counter.value += chunk
        if start >= n:
            break
        for i in range(start,min(n,start+chunk)):
            acc += do_item(i)
    runtime_arr[worker_id]=(time.perf_counter()-t0)
    result_arr[worker_id]=acc

def run_schedule(mode, p, n, chunk=1):
    ctx=mp.get_context()
    runtimes=ctx.Array('d',p)
    results=ctx.Array('d',p)
    procs=[]
    t0=time.perf_counter()
    if mode=="block":
        size=(n+p-1)//p
        for w in range(p):
            inds=range(w*size,min(n,(w+1)*size))
            q=ctx.Process(target=schedule_worker_static,args=(w,inds,runtimes,results))
            q.start();procs.append(q)
    elif mode=="cyclic":
        for w in range(p):
            q=ctx.Process(target=schedule_worker_static,args=(w,range(w,n,p),runtimes,results))
            q.start();procs.append(q)
    else:
        counter=ctx.Value('i',0)
        lock=ctx.Lock()
        for w in range(p):
            q=ctx.Process(target=dynamic_worker,args=(w,counter,lock,chunk,n,runtimes,results))
            q.start();procs.append(q)
    for q in procs:
        q.join()
        if q.exitcode != 0:
            raise RuntimeError("Scheduling worker failed")
    wall=time.perf_counter()-t0
    rt=list(runtimes[:])
    avg=sum(rt)/p if p else 0
    imbalance=max(rt)/avg if avg else 0
    return wall*1000, imbalance, rt

def phase4():
    print("\n" + "="*72 + "\nPHASE 4 — SKEWED WORKLOADS & DYNAMIC CHUNK QUEUE\n" + "="*72)
    print(f"Skewed items N={N_SKEW}, workers={PPHYS}")
    block=run_schedule("block",PPHYS,N_SKEW)
    cyclic=run_schedule("cyclic",PPHYS,N_SKEW)
    print(f"Block static:  wall={block[0]:.2f} ms, imbalance lambda={block[1]:.3f}")
    print(f"Cyclic static: wall={cyclic[0]:.2f} ms, imbalance lambda={cyclic[1]:.3f}")
    dyn={}
    for c in (1,8,64,512):
        dyn[c]=run_schedule("dynamic",PPHYS,N_SKEW,c)
        print(f"Dynamic C={c:<3}: wall={dyn[c][0]:.2f} ms, imbalance lambda={dyn[c][1]:.3f}")
    best=min(dyn,key=lambda c:dyn[c][0])
    print(f"Best measured dynamic chunk C*={best}")
    print("Trade-off: small chunks balance work finely but increase lock/counter contention;")
    print("large chunks reduce synchronization but can leave a long uneven tail.")
    verify_tag("PHASE4", STUDENT_ID, block[0], dyn[best][0])

def main():
    print("#"*72)
    print("PERFORMANCE EVALUATION AND OPTIMIZATION IN PARALLEL COMPUTING")
    print(STUDENT_NAME, "| Student ID:", STUDENT_ID)
    print(f"QUICK_MODE={QUICK_MODE} | Logical cores={PLOG} | Physical-core estimate={PPHYS}")
    print("Worker counts:", WORKERS)
    print("#"*72)
    hashes=[]
    for fn in (phase0,phase1,phase2,phase3,phase4):
        fn()
    print("\n" + "="*72)
    print("RUN COMPLETE")
    print("Copy all five VERIFY_HASH lines and measured tables into your lab report.")
    print("Note: cache sizes and physical-core count may not be exposed by hosted Colab.")
    print("="*72)

if __name__ == "__main__":
    mp.freeze_support()
    main()
