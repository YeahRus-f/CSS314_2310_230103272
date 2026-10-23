"""TASK 4 — Work Stealing & Scheduling Under Skew.

Item i costs sleep(0.0001 * (i % 20)) seconds (irregular workload).
time.sleep releases the GIL, so 4 threads truly overlap.
  Strategy A: static chunks  — worker k gets items [k*250, (k+1)*250)
  Strategy B: dynamic queue  — idle workers grab the next chunk (size 1 or 5)
"""
import time
import threading
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


ITEMS = 1000
WORKERS = 4


def work(i):
    time.sleep(0.0001 * (i % 20))


def static_schedule():
    block = ITEMS // WORKERS
    finish = [0.0] * WORKERS
    done = [0] * WORKERS

    def worker(k):
        lo = k * block
        hi = ITEMS if k == WORKERS - 1 else lo + block
        for i in range(lo, hi):
            work(i)
        done[k] = hi - lo
        finish[k] = time.perf_counter()

    return _run(worker, finish, done)


def dynamic_schedule(chunk):
    finish = [0.0] * WORKERS
    done = [0] * WORKERS
    nxt = [0]
    lock = threading.Lock()

    def worker(k):
        while True:
            with lock:                      # take next chunk from shared queue
                lo = nxt[0]
                nxt[0] += chunk
            if lo >= ITEMS:
                break
            for i in range(lo, min(lo + chunk, ITEMS)):
                work(i)
                done[k] += 1
        finish[k] = time.perf_counter()

    return _run(worker, finish, done)


def _run(worker, finish, done):
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(WORKERS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    delta = max(finish) - min(finish)
    per_thread = [f - t0 for f in finish]
    assert sum(done) == ITEMS
    return total, delta, per_thread, done


def report(name, res):
    total, delta, per_thread, done = res
    print(f"{name}")
    print(f"  Total execution time        : {total * 1000:9.2f} ms")
    print(f"  First->last finish delta    : {delta * 1000:9.2f} ms")
    print("  Per-thread finish (ms)      : " + ", ".join(f"{t * 1000:.1f}" for t in per_thread))
    print("  Items processed per thread  : " + ", ".join(str(d) for d in done))
    print()


def main():
    start_log("task4_output.txt")
    header("TASK 4 — Work Stealing & Scheduling Under Skew")

    ideal = sum(0.0001 * (i % 20) for i in range(ITEMS))
    print(f"{ITEMS} items, {WORKERS} workers, cost(i) = 0.1 ms * (i % 20)")
    print(f"Total requested sleep = {ideal * 1000:.1f} ms -> ideal parallel time "
          f"= {ideal / WORKERS * 1000:.1f} ms\n")

    a = static_schedule()
    b1 = dynamic_schedule(1)
    b5 = dynamic_schedule(5)
    report("Strategy A: STATIC chunks (250 items per worker)", a)
    report("Strategy B: DYNAMIC / work-queue, chunk = 1", b1)
    report("Strategy B: DYNAMIC / work-queue, chunk = 5", b5)

    print("Summary (finish delta = load imbalance):")
    print(f"  Static  : total {a[0] * 1000:8.2f} ms | delta {a[1] * 1000:7.2f} ms")
    print(f"  Dyn(1)  : total {b1[0] * 1000:8.2f} ms | delta {b1[1] * 1000:7.2f} ms")
    print(f"  Dyn(5)  : total {b5[0] * 1000:8.2f} ms | delta {b5[1] * 1000:7.2f} ms")
    best = min([("Static", a), ("Dynamic chunk=1", b1), ("Dynamic chunk=5", b5)],
               key=lambda x: x[1][1])[0]
    print(f"Most balanced strategy (smallest finish delta): {best}")


if __name__ == "__main__":
    main()
