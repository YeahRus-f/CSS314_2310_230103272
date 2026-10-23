"""TASK 7 — Arithmetic Intensity vs. Memory Bottlenecks.

Kernel A (memory-bound) : B[i] = A[i]
Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])
Both split into chunks over a thread pool (numpy releases the GIL).
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


N = 30_000_000
REPEATS = 3


def copy_kernel(a, b, lo, hi):
    np.copyto(b[lo:hi], a[lo:hi])


def trig_kernel(a, b, lo, hi):
    x = a[lo:hi]
    out = b[lo:hi]
    np.sin(x, out=out)
    out *= np.cos(x)
    out += np.tan(x)


def run(kernel, a, b, workers, pool):
    bounds = np.linspace(0, len(a), workers + 1, dtype=np.int64)
    futs = [pool.submit(kernel, a, b, int(bounds[k]), int(bounds[k + 1]))
            for k in range(workers)]
    for f in futs:
        f.result()


def bench(kernel, a, b, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        run(kernel, a, b, workers, pool)          # warm-up (also touches pages)
        best = float("inf")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            run(kernel, a, b, workers, pool)
            best = min(best, time.perf_counter() - t0)
    return best


def main():
    start_log("task7_output.txt")
    header("TASK 7 — Arithmetic Intensity vs. Memory Bottlenecks")

    a = np.random.default_rng(STUDENT_ID).random(N)
    b = np.empty_like(a)
    max_cores = logical_cores()
    counts = sorted({1, 2, max_cores})
    print(f"N = {N:,} float64 ({a.nbytes / 1e6:.0f} MB per array) | best of {REPEATS}\n")

    eff_at_max = {}
    for name, kernel, flop_note in [
        ("Kernel A (memory-bound copy)", copy_kernel, "0 FLOP / 16 B moved"),
        ("Kernel B (compute-bound trig)", trig_kernel, "~60+ FLOP / 16 B moved"),
    ]:
        print(f"{name}  [{flop_note}]")
        print(f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'GB/s':>7}")
        print("-" * 54)
        t1 = None
        for w in counts:
            t = bench(kernel, a, b, w)
            t1 = t if t1 is None else t1
            sp = t1 / t
            eff = sp / w
            gbs = 2 * a.nbytes / t / 1e9           # bytes read + written
            print(f"{w:>6} | {t * 1000:>10.2f} | {sp:>8.2f} | {eff * 100:>9.1f}% | {gbs:>7.2f}")
            eff_at_max[name] = eff
        print()

    np.testing.assert_allclose(b[:10], np.sin(a[:10]) * np.cos(a[:10]) + np.tan(a[:10]))

    stalled = min(eff_at_max, key=eff_at_max.get)
    other = max(eff_at_max, key=eff_at_max.get)
    print("VERDICT:")
    print(f"  {stalled} stalled in scaling: efficiency at {max_cores} cores = "
          f"{eff_at_max[stalled] * 100:.1f}% vs {eff_at_max[other] * 100:.1f}% for {other}.")
    print(f"  {stalled.split(' (')[0]} is limited by MEMORY CHANNEL SATURATION: adding cores "
          "does not add DRAM bandwidth, so the cores wait on RAM.")
    print(f"  {other.split(' (')[0]} has high arithmetic intensity, so each core stays busy "
          "computing and scaling stays close to linear.")


if __name__ == "__main__":
    main()
