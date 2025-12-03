"""
A very small integration check that spawns coordinator (thread) and three workers (subprocesses)
and checks logs to verify masked gradient messages were received.
This is not a full pytest suite but a quick smoke test.
"""
import subprocess, time, os, signal, sys, json
def run_demo():
    # start coordinator
    coord = subprocess.Popen([sys.executable, "coordinator.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.6)
    # start 3 workers
    workers = []
    for i in [1,2,3]:
        w = subprocess.Popen([sys.executable, "worker.py", "--id", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        workers.append(w)
        time.sleep(0.2)
    # wait a bit for exchange
    time.sleep(5)
    # kill processes
    for w in workers:
        try:
            w.send_signal(signal.SIGINT)
            w.wait(timeout=1)
        except:
            w.kill()
    try:
        coord.send_signal(signal.SIGINT)
        coord.wait(timeout=1)
    except:
        coord.kill()
    # check coordinator logs for aggregated events
    logs = open("logs/adst_coord.log").read().splitlines()
    found = any("masked_gradient_received" in l for l in logs)
    print("masked_gradient_received in logs:", found)
    return 0 if found else 2

if __name__ == "__main__":
    sys.exit(run_demo())
