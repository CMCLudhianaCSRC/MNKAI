"""Autonomous Phase-3 (Indian 25-class) fold orchestrator.

Same self-healing design as scripts/_orchestrate_folds.py, but for Phase 3:
runs folds 0..4 of configs/phase3_indian_25class.yaml to completion, resuming
any crash from last.pt, and refusing to double-launch while the GPU is busy
(so a hand-started fold0 finishes before fold1 starts).

Launch (background):
  nohup .venv/Scripts/python.exe scripts/_orchestrate_phase3.py \
        > logs/orchestrator_phase3.log 2>&1 &
"""
from __future__ import annotations
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
CONFIG = "configs/phase3_indian_25class.yaml"
SEQUENCE = [0, 1, 2, 3, 4]
TARGET_EPOCHS = 30
OUT_FMT = "checkpoints/phase3_indian_fold{n}"

GPU_FREE_MIB = 1500
GPU_FREE_READS = 3
GPU_POLL_SEC = 30


def log(m):
    print(f"[orch-p3 {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def fold_dir(n): return ROOT / OUT_FMT.format(n=n)


def fold_epoch(n):
    ck = fold_dir(n) / "last.pt"
    if not ck.exists():
        return -1
    try:
        import torch
        return int(torch.load(ck, map_location="cpu", weights_only=False).get("epoch", -1))
    except Exception as e:
        log(f"warn: read {ck}: {e}")
        return -1


def fold_complete(n): return fold_epoch(n) >= TARGET_EPOCHS


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return max(int(x) for x in out.strip().splitlines())
    except Exception as e:
        log(f"warn: nvidia-smi: {e}; assume busy")
        return 999999


def wait_for_gpu_free(done_cb):
    idle = 0
    while True:
        if done_cb():
            return False
        if gpu_used_mib() < GPU_FREE_MIB:
            idle += 1
            if idle >= GPU_FREE_READS:
                return True
        else:
            idle = 0
        time.sleep(GPU_POLL_SEC)


def kill_stray_workers():
    """Kill any leftover training / DataLoader-worker python procs from a prior
    crash (Windows orphans workers when the parent dies). Prevents the RAM
    death-spiral that caused OpenBLAS OOM. Safe: runs BEFORE we launch, so it
    never touches our own future child."""
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'scripts.train|multiprocessing.spawn' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
            capture_output=True, timeout=30)
    except Exception as e:
        log(f"warn: stray-cleanup failed: {e}")


def launch_fold(n):
    kill_stray_workers()
    time.sleep(3)
    outdir = OUT_FMT.format(n=n)
    last = fold_dir(n) / "last.pt"
    cmd = [PY, "-m", "scripts.train", "--config", CONFIG,
           "--fold", str(n), "--out-dir", outdir]
    if last.exists():
        cmd += ["--resume-from", outdir + "/last.pt"]
        log(f"fold {n}: resume from epoch {fold_epoch(n)}")
    else:
        log(f"fold {n}: fresh start (init-from-backbone per config)")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    logf = ROOT / "logs" / f"phase3_fold{n}_train.log"
    with open(logf, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== orchestrator-p3 launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        f.flush()
        rc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                              stdout=f, stderr=subprocess.STDOUT).wait()
    log(f"fold {n}: exited rc={rc}")


def main():
    log(f"start; sequence={SEQUENCE} target={TARGET_EPOCHS}")
    for n in SEQUENCE:
        log(f"=== fold {n} (epoch={fold_epoch(n)}) ===")
        while not fold_complete(n):
            if not wait_for_gpu_free(lambda: fold_complete(n)):
                break
            if fold_complete(n):
                break
            launch_fold(n)
            time.sleep(20)
        log(f"fold {n}: COMPLETE (epoch {fold_epoch(n)})")
    log("ALL PHASE-3 FOLDS COMPLETE -> ready for Phase 3 eval + Phase 4 novelty.")


if __name__ == "__main__":
    main()
