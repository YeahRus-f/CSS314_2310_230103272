"""TASK 5 — The Overhead Crossover Point.

y[i] = a * x[i] + b, serial vs. thread pool with all logical cores.
After the coarse sweep, the script bisects (geometrically) between the last
"slower" N and the first "faster" N to locate the tipping point precisely.
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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


A, B = 2.5, 1.0
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def axpb(x, y, lo, hi):
    np.multiply(x[lo:hi], A, out=y[lo:hi])
    np.add(y[lo:hi], B, out=y[lo:hi])


def serial(x, y):
    axpb(x, y, 0, len(x))


def parallel(x, y, pool, workers):
    n = len(x)
    bounds = np.linspace(0, n, workers + 1, dtype=np.int64)
    futs = [pool.submit(axpb, x, y, int(bounds[k]), int(bounds[k + 1]))
            for k in range(workers)]
    for f in futs:
        f.result()


def bench(fn, n):
    """Median time per call; more repetitions for small N."""
    reps = max(5, min(2000, 20_000_000 // max(n, 1)))
    reps = min(reps, 2000)
    fn()                                    # warm-up
    samples = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(max(1, reps // 7)):
            fn()
        samples.append((time.perf_counter() - t0) / max(1, reps // 7))
    return float(np.median(samples))


def measure(n, pool, workers, rng):
    x = rng.random(n)
    y1 = np.empty_like(x)
    y2 = np.empty_like(x)
    ts = bench(lambda: serial(x, y1), n)
    tp = bench(lambda: parallel(x, y2, pool, workers), n)
    assert np.allclose(y1, y2)
    return ts, tp


def main():
    start_log("task5_output.txt")
    header("TASK 5 — The Overhead Crossover Point")

    workers = logical_cores()
    rng = np.random.default_rng(STUDENT_ID)
    print(f"Kernel: y = {A}*x + {B} | parallel = ThreadPool with {workers} workers\n")
    print(f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>13} | {'Speedup':>8} | Verdict")
    print("-" * 66)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for n in SIZES:
            ts, tp = measure(n, pool, workers, rng)
            faster = tp < ts
            rows.append((n, faster))
            print(f"{n:>12,} | {ts * 1000:>12.4f} | {tp * 1000:>13.4f} | {ts / tp:>8.2f} | "
                  f"{'PARALLEL FASTER' if faster else 'parallel slower'}")

        # crossover: last N where parallel is slower, followed only by "faster" sizes
        first_fast = None
        for idx in range(len(rows)):
            if all(f for _, f in rows[idx:]):
                first_fast = idx
                break

        print()
        if first_fast is None:
            print("Parallel never became consistently faster in the tested range "
                  f"(up to N = {SIZES[-1]:,}).")
            return
        if first_fast == 0:
            print(f"Parallel was already faster at the smallest N = {SIZES[0]:,}.")
            return

        lo, hi = rows[first_fast - 1][0], rows[first_fast][0]
        print(f"Coarse crossover: parallel SLOWER at N = {lo:,}, FASTER at N = {hi:,}")
        print("Refining by geometric bisection...")
        for _ in range(5):
            mid = int(round((lo * hi) ** 0.5))
            if mid in (lo, hi):
                break
            ts, tp = measure(mid, pool, workers, rng)
            verdict = "faster" if tp < ts else "slower"
            print(f"  N = {mid:>10,}: serial {ts * 1000:.4f} ms, parallel {tp * 1000:.4f} ms "
                  f"-> parallel {verdict}")
            if tp < ts:
                hi = mid
            else:
                lo = mid

    print("\nRESULT on this machine:")
    print(f"  Multi-threading is SLOWER than serial for N <= {lo:,}")
    print(f"  Multi-threading is FASTER than serial for N >= {hi:,}")
    print(f"  Overhead crossover point lies in the range N = ({lo:,}, {hi:,}]")


if __name__ == "__main__":
    main()
