"""TASK 8 — Parallel Tile Reduction (CPU path, no dedicated GPU).

1024 x 1024 matrix multiplication C = A @ B across CPU threads:
  * non-tiled : each task computes one full row  C[i,:] = A[i,:] @ B
                (streams the whole 8 MB matrix B for every row -> poor cache reuse)
  * tiled 16  : C split into 16x16 tiles, k-loop over 16x16 blocks
  * tiled 64  : C split into 64x64 tiles, k-loop over 64x64 blocks (fits in L1/L2)
BLAS is forced to 1 thread so that ONLY our worker threads provide parallelism.
"""
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import time                                            # noqa: E402
import numpy as np                                     # noqa: E402
from concurrent.futures import ThreadPoolExecutor      # noqa: E402
# ---------------- helpers (student ID, hardware info, output log) ----------------
import os
import sys
import platform
import subprocess

STUDENT_ID = 230103272


class Tee:
    """Duplicates everything printed to the console into taskN_output.txt (UTF-8)."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, s):
        self.console.write(s)
        self.file.write(s)

    def flush(self):
        self.console.flush()
        self.file.flush()


def start_log(filename):
    """Call ONLY inside main(); output file is saved next to this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    sys.stdout = Tee(path)


def cpu_model():
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def logical_cores():
    return os.cpu_count() or 1


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return logical_cores()  # fallback if psutil is not installed


def header(task_title):
    print("=" * 64)
    print(task_title)
    print("=" * 64)
    print(f"Student ID     : {STUDENT_ID}")
    print(f"OS             : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform       : {platform.platform()}")
    print(f"CPU model      : {cpu_model()}")
    print(f"Logical cores  : {logical_cores()}  (os.cpu_count())")
    print(f"Physical cores : {physical_cores()}")
    print(f"Python         : {platform.python_version()}")
    print("-" * 64)
# ---------------------------------------------------------------------------------


N = 1024


def non_tiled(A, B, C, workers):
    def rows(lo, hi):
        for i in range(lo, hi):
            np.dot(A[i], B, out=C[i])                  # one row at a time
    bounds = np.linspace(0, N, workers * 4 + 1, dtype=int)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(rows, bounds[:-1], bounds[1:]))


def tiled(A, B, C, workers, bs):
    def tile(ij):
        i, j = ij
        acc = np.zeros((bs, bs))
        tmp = np.empty((bs, bs))
        for k in range(0, N, bs):
            np.dot(A[i:i + bs, k:k + bs], B[k:k + bs, j:j + bs], out=tmp)
            acc += tmp
        C[i:i + bs, j:j + bs] = acc
    tiles = [(i, j) for i in range(0, N, bs) for j in range(0, N, bs)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(tile, tiles, chunksize=1))


def main():
    start_log("task8_output.txt")
    header("TASK 8 — Compute Geometry: Tiled Matrix Multiplication (CPU)")

    workers = logical_cores()
    rng = np.random.default_rng(STUDENT_ID)
    A = rng.random((N, N))
    B = rng.random((N, N))
    ref = A @ B
    flops = 2 * N ** 3
    print(f"Matrix {N}x{N} float64 | {workers} worker threads | BLAS threads = 1")
    print("No dedicated NVIDIA GPU path used -> CPU blocked/tiled variant\n")

    variants = [
        ("Non-tiled (row-wise)", lambda C: non_tiled(A, B, C, workers)),
        ("Tiled 16 x 16", lambda C: tiled(A, B, C, workers, 16)),
        ("Tiled 64 x 64", lambda C: tiled(A, B, C, workers, 64)),
    ]
    results = []
    for name, fn in variants:
        best = float("inf")
        for _ in range(2):
            C = np.zeros((N, N))
            t0 = time.perf_counter()
            fn(C)
            best = min(best, time.perf_counter() - t0)
        ok = np.allclose(C, ref)
        results.append((name, best, ok))

    base = results[0][1]
    print(f"{'Variant':<22} | {'Runtime (ms)':>12} | {'GFLOP/s':>8} | {'vs non-tiled':>12} | Correct")
    print("-" * 76)
    for name, t, ok in results:
        print(f"{name:<22} | {t * 1000:>12.1f} | {flops / t / 1e9:>8.2f} | "
              f"{base / t:>11.2f}x | {ok}")

    t = {name: tm for name, tm, _ in results}
    best_name = min(results, key=lambda r: r[1])[0]
    print(f"\nFastest geometry on this CPU: {best_name}")
    print(f"Tiled 64x64 vs Tiled 16x16: {t['Tiled 16 x 16'] / t['Tiled 64 x 64']:.1f}x faster")
    print("Interpretation:")
    print("  * 16x16 tiles: each block is only 2 KB, so the work per call is tiny; 262,144")
    print("    block products -> per-call/scheduling overhead dominates over useful FLOPs.")
    print("  * 64x64 tiles: three 32 KB blocks fit in L1/L2 and each loaded block is reused")
    print("    64 times -> few cache misses per FLOP and low overhead (4,096 calls per k-step).")
    print("  * Non-tiled rows: matrix B (8 MB) exceeds L2 and is re-streamed for every row;")
    if t["Non-tiled (row-wise)"] < t["Tiled 64 x 64"]:
        print("    here the large L3 + hardware prefetcher hide that latency, and one long")
        print("    contiguous gemv call per row has the lowest overhead, so it still wins.")
    else:
        print("    the repeated cache misses on B make it slower than the 64x64 tiling.")
    print("  (Exact hardware cache-miss counters need `perf stat -e cache-misses` on Linux;")
    print("   runtime and GFLOP/s above are used as the proxy for miss latency.)")


if __name__ == "__main__":
    main()
