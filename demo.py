#!/usr/bin/env python3
"""
demo.py - orchestrator (coordinator + 4 workers + dashboard)
"""
import os, subprocess, time, signal, sys

NUM_WORKERS = 4
COORD_TCP_PORT = 9000
COORD_UDP_PORT = 9001

def run_process(cmd, name):
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    stdout = open(os.path.join(log_dir, f"{name}.log"), "w")
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stdout, shell=True)
    return proc, stdout

def main():
    print("🚀 Starting ADST 2.0 Federated Learning Demo\n")
    # coordinator
    coord_cmd = f"python coordinator.py --host 127.0.0.1 --tcp-port {COORD_TCP_PORT} --udp-port {COORD_UDP_PORT}"
    coord_proc, coord_out = run_process(coord_cmd, "coordinator")
    time.sleep(2)
    workers = []
    outs = []
    for i in range(1, NUM_WORKERS+1):
        cmd = f"python worker.py --id {i} --coord-host 127.0.0.1 --coord-port {COORD_TCP_PORT} --udp-port {COORD_UDP_PORT} --jobid 0x1122334455667788"
        p, out = run_process(cmd, f"worker_{i}")
        workers.append(p); outs.append(out); time.sleep(1)
    # dashboard (headless)
    dash_cmd = "streamlit run dashboard.py --server.headless true"
    dash_proc, dash_out = run_process(dash_cmd, "dashboard")
    print("✅ All started. Dashboard: http://localhost:8501")
    try:
        # wait for coordinator to finish (it exits when MAX_EPOCHS completed)
        while True:
            ret = coord_proc.poll()
            if ret is not None:
                print("[demo] Coordinator exited. Shutting down workers and dashboard.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("[demo] Interrupted by user.")
    # terminate all
    for p in workers:
        try: p.terminate()
        except: pass
    try: dash_proc.terminate()
    except: pass
    try: coord_proc.terminate()
    except: pass
    # close file handles
    coord_out.close()
    for o in outs: o.close()
    dash_out.close()
    print("[demo] All stopped.")
    sys.exit(0)

if __name__ == "__main__":
    main()
