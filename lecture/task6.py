"""TASK 6 — Synchronization Cost & Data Loss.

In CPython the GIL makes most `counter += 1` races invisible with threads,
so the 4 parallel workers are PROCESSES sharing one int64 in shared memory.
This gives genuine simultaneous read-modify-write on real cores.
"""
import time
import multiprocessing as mp
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


WORKERS = 4
ITERS = 2_000_000
EXPECTED = WORKERS * ITERS


def unsafe_worker(buf, iters, barrier):
    c = memoryview(buf).cast("B").cast("q")
    barrier.wait()
    for _ in range(iters):
        c[0] += 1                       # load -> add -> store, NOT atomic


def locked_worker(buf, iters, barrier, lock):
    c = memoryview(buf).cast("B").cast("q")
    barrier.wait()
    for _ in range(iters):
        with lock:                      # mutual exclusion (mutex)
            c[0] += 1


def run(target, extra=()):
    buf = mp.RawArray("q", 1)           # shared counter = 0
    barrier = mp.Barrier(WORKERS + 1)
    procs = [mp.Process(target=target, args=(buf, ITERS, barrier, *extra))
             for _ in range(WORKERS)]
    for p in procs:
        p.start()
    barrier.wait()                      # all workers ready -> start the clock
    t0 = time.perf_counter()
    for p in procs:
        p.join()
    return buf[0], time.perf_counter() - t0


def main():
    start_log("task6_output.txt")
    header("TASK 6 — Synchronization Cost & Data Loss")
    print(f"{WORKERS} parallel workers x {ITERS:,} increments each\n")

    actual, t_unsafe = run(unsafe_worker)
    lost = EXPECTED - actual
    print("[1] NO synchronization")
    print(f"  Expected final value : {EXPECTED:,}")
    print(f"  Actual final value   : {actual:,}")
    print(f"  Lost updates         : {lost:,} ({lost / EXPECTED * 100:.2f}%)")
    print(f"  Wall-clock time      : {t_unsafe * 1000:.1f} ms\n")

    lock = mp.Lock()
    actual_l, t_locked = run(locked_worker, (lock,))
    print("[2] WITH Lock (mutex around every increment)")
    print(f"  Expected final value : {EXPECTED:,}")
    print(f"  Actual final value   : {actual_l:,}")
    print(f"  Lost updates         : {EXPECTED - actual_l:,} "
          f"({(EXPECTED - actual_l) / EXPECTED * 100:.2f}%)")
    print(f"  Wall-clock time      : {t_locked * 1000:.1f} ms\n")

    print("Synchronization penalty:")
    print(f"  +{(t_locked - t_unsafe) * 1000:.1f} ms  ({t_locked / t_unsafe:.2f}x slower), "
          f"~{(t_locked - t_unsafe) / EXPECTED * 1e9:.0f} ns extra per increment")
    print("\nConclusion: without a lock, concurrent load-add-store sequences overwrite each "
          "other and updates are lost. The lock restores correctness but serializes the "
          "critical section and adds acquire/release + cache-line transfer cost.")


if __name__ == "__main__":
    main()
