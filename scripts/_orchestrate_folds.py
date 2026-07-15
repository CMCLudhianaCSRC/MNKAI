"""Autonomous fold orchestrator for MNKAI-III Phase 1e.

Runs folds 2 -> 3 -> 4 to completion (50 epochs each) with ZERO human input,
and auto-recovers from crashes / power loss.

Design (robust to the messy real world):
  * For each fold N in SEQUENCE:
      - if fold already complete (last.pt epoch >= TARGET_EPOCHS) -> skip.
      - otherwise WAIT until the GPU is free (handles the case where fold 2 is
        already running as a separate process started by hand). When the GPU is
        free and the fold is still incomplete, (re)launch it:
            * resume from last.pt if it exists  (crash/power recovery)
            * fresh start otherwise              (folds 3/4 first launch)
      - wait for that training process to exit, then re-evaluate.
  * A fold is only declared done when its last.pt reports epoch >= TARGET_EPOCHS.
    So a crash (process dies at epoch < 50) simply loops back and resumes.

This never double-launches: it refuses to start a fold while the GPU is busy,
which is exactly when an externally-started training run is active.

Launch (background):
  nohup .venv/Scripts/python.exe scripts/_orchestrate_folds.py \
        > logs/orchestrator.log 2>&1 &
"""
from __future__ import annotations
import subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
CONFIG = "configs/phase1_dinov2_mnkai3_folds1-4.yaml"
SEQUENCE = [2, 3, 4]
TARGET_EPOCHS = 50

GPU_FREE_MIB = 1500          # below this = GPU considered idle
GPU_FREE_READS = 3           # consecutive idle reads required
GPU_POLL_SEC = 30


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[orch {ts}] {msg}", flush=True)


def fold_dir(n: int) -> Path:
    return ROOT / "checkpoints" / f"phase1_dinov2_fold{n}"


def fold_epoch(n: int) -> int:
    """Last completed epoch stored in last.pt, or -1 if none/unreadable."""
    ck = fold_dir(n) / "last.pt"
    if not ck.exists():
        return -1
    try:
        import torch
        d = torch.load(ck, map_location="cpu", weights_only=False)
        return int(d.get("epoch", -1))
    except Exception as e:
        log(f"warn: could not read {ck}: {e}")
        return -1


def fold_complete(n: int) -> bool:
    return fold_epoch(n) >= TARGET_EPOCHS   # last.pt 'epoch' is 1-based; 50 == all epochs done


def gpu_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return max(int(x) for x in out.strip().splitlines())
    except Exception as e:
        log(f"warn: nvidia-smi failed: {e}; assuming busy")
        return 999999


def wait_for_gpu_free(skip_if_done):
    idle = 0
    while True:
        if skip_if_done():
            return False  # fold finished while we waited (external proc completed it)
        used = gpu_used_mib()
        if used < GPU_FREE_MIB:
            idle += 1
            log(f"gpu idle {used} MiB ({idle}/{GPU_FREE_READS})")
            if idle >= GPU_FREE_READS:
                return True
        else:
            if idle:
                log(f"gpu busy again ({used} MiB)")
            idle = 0
        time.sleep(GPU_POLL_SEC)


def launch_fold(n: int) -> int:
    outdir = f"checkpoints/phase1_dinov2_fold{n}"
    last = fold_dir(n) / "last.pt"
    cmd = [PY, "-m", "scripts.train", "--config", CONFIG,
           "--fold", str(n), "--out-dir", outdir]
    if last.exists():
        cmd += ["--resume-from", outdir + "/last.pt"]
        log(f"fold {n}: resuming from {last} (epoch {fold_epoch(n)})")
    else:
        log(f"fold {n}: fresh start")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    logf = ROOT / "logs" / f"fold{n}_train.log"
    log(f"fold {n}: launching -> {logf}")
    with open(logf, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== orchestrator launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        rc = proc.wait()
    log(f"fold {n}: training process exited rc={rc}")
    return rc


def main():
    log(f"orchestrator start; sequence={SEQUENCE} target_epochs={TARGET_EPOCHS}")
    for n in SEQUENCE:
        log(f"=== fold {n} (current epoch={fold_epoch(n)}) ===")
        while not fold_complete(n):
            became_free = wait_for_gpu_free(lambda: fold_complete(n))
            if not became_free:
                log(f"fold {n}: completed externally while waiting")
                break
            if fold_complete(n):
                break
            launch_fold(n)
            # brief pause so GPU memory is released before re-evaluating
            time.sleep(20)
        log(f"fold {n}: COMPLETE (epoch {fold_epoch(n)})")
    log("ALL FOLDS COMPLETE. Phase 1e done -> ready for Phase 1f ensemble.")


if __name__ == "__main__":
    main()
