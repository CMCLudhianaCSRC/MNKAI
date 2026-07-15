"""Stall watchdog for MNKAI-III training.

Closes the one gap the orchestrator can't: a training process that stays ALIVE
but stops making progress (e.g. a wedged CUDA context after the machine slept).
The orchestrator blocks in proc.wait() in that case and can't recover on its own.

How it works:
  * Every POLL_SEC, find the most recently modified fold*_train.log.
  * Healthy training rewrites that log every few seconds (tqdm progress bar).
  * If the active log's mtime is stale for > STALL_SEC AND a `scripts.train`
    python process is alive, that's a stall -> kill the training process.
  * Killing it makes the orchestrator's proc.wait() return; the orchestrator
    then re-evaluates, sees the fold incomplete, waits for the GPU to free,
    and relaunches (resumes from last.pt). Full self-heal, no human needed.

STALL_SEC is set generously (20 min) so a legitimate quiet stretch (end-of-epoch
validation, checkpoint save) never triggers a false kill. A real sleep-stall
lasts far longer. During an actual system sleep the watchdog is suspended too,
so it only ever acts after the machine is awake again.

Launch (background):
  nohup .venv/Scripts/python.exe scripts/_watchdog.py > logs/watchdog.log 2>&1 &
"""
from __future__ import annotations
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "logs"

POLL_SEC = 120          # how often to check
STALL_SEC = 1200        # 20 min with no log write = stalled
MIN_AGE_SEC = 90        # ignore a log younger than this (just-launched, no bar yet)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[watchdog {ts}] {msg}", flush=True)


def active_train_log():
    cands = list(LOGDIR.glob("*fold*_train.log"))   # phase1 foldN_ + phase3 phase3_foldN_
    cands = [c for c in cands if c.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def train_pids():
    """PIDs of live `python -m scripts.train` processes (Windows)."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'scripts.train' } | "
             "Select-Object -ExpandProperty ProcessId"],
            text=True, timeout=30)
        return [int(x) for x in out.split()]
    except Exception as e:
        log(f"warn: could not list train pids: {e}")
        return []


def kill_pids(pids):
    for p in pids:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p)],
                           capture_output=True, timeout=20)
        except Exception as e:
            log(f"warn: kill {p} failed: {e}")


def main():
    log(f"watchdog start; poll={POLL_SEC}s stall_threshold={STALL_SEC}s")
    while True:
        time.sleep(POLL_SEC)
        lg = active_train_log()
        if lg is None:
            continue
        try:
            age = time.time() - lg.stat().st_mtime
        except Exception:
            continue
        if age < MIN_AGE_SEC:
            continue
        pids = train_pids()
        if not pids:
            # no training running (between folds / paused) -> nothing to watch
            continue
        if age > STALL_SEC:
            log(f"STALL DETECTED: {lg.name} stale {age:.0f}s, train pids {pids} alive "
                f"-> killing so orchestrator can resume")
            kill_pids(pids)
            # give the OS a moment; orchestrator will relaunch on its next loop
            time.sleep(30)
        # else: healthy, log is fresh


if __name__ == "__main__":
    main()
