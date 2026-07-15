"""Phase-3 scientific ablation driver (GPU). Launch ONLY when GPU is free.

Pipeline:
  STAGE 1  bake-off: train fold 0 for each single-mechanism fix (A/B/C),
           then evaluate each fold-0 model on the HELD-OUT external test set.
  STAGE 2  pick the winner by external balanced-accuracy, tie-broken by the mean
           recall of the previously-collapsed classes (eczema/psoriasis/tinea/
           viral_infection) -> the fix that best restores the common classes.
  STAGE 3  train the winner's remaining folds (1-4).
  STAGE 4  full 5-fold publication eval (OOF + external ensemble + calibration +
           conformal + fairness).

Self-heals like the orchestrator: waits for a free GPU before each launch and
resumes a crashed fold from last.pt. Logs decisions to logs/ablation_driver.log.

Launch:
  nohup .venv/Scripts/python.exe scripts/_phase3_ablation_driver.py \
        > logs/ablation_driver.log 2>&1 &
"""
from __future__ import annotations
import subprocess, sys, time, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
FIXES = {
    "A": "configs/phase3_fixA_ce_sampler.yaml",
    "B": "configs/phase3_fixB_balsoftmax.yaml",
    "C": "configs/phase3_fixC_cbfocal_only.yaml",
}
CKDIR = {k: f"checkpoints/phase3_fix{k}" for k in FIXES}
EVALDIR = {k: f"eval/phase3_fix{k}" for k in FIXES}
COLLAPSED = ["eczema", "psoriasis", "tinea", "viral_infection"]
TARGET_EPOCHS = 30
GPU_FREE_MIB, GPU_FREE_READS, GPU_POLL = 1500, 3, 30


def log(m): print(f"[abl {time.strftime('%H:%M:%S')}] {m}", flush=True)
def env():
    e = dict(os.environ); e["PYTHONPATH"] = "src" + os.pathsep + e.get("PYTHONPATH", ""); return e


def gpu_busy():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], text=True, timeout=15)
        return max(int(x) for x in out.split()) >= GPU_FREE_MIB
    except Exception:
        return True


def wait_gpu():
    idle = 0
    while idle < GPU_FREE_READS:
        idle = idle + 1 if not gpu_busy() else 0
        time.sleep(GPU_POLL)


def fold_done(ckdir, n):
    p = ROOT / ckdir / f"fold{n}" / "last.pt"
    if not p.exists():
        return False
    try:
        import torch
        return int(torch.load(p, map_location="cpu", weights_only=False).get("epoch", -1)) >= TARGET_EPOCHS
    except Exception:
        return False


def train_fold(cfg, ckdir, n):
    while not fold_done(ckdir, n):
        wait_gpu()
        if fold_done(ckdir, n):
            break
        outdir = f"{ckdir}/fold{n}"
        last = ROOT / outdir / "last.pt"
        cmd = [PY, "-m", "scripts.train", "--config", cfg, "--fold", str(n), "--out-dir", outdir]
        if last.exists():
            cmd += ["--resume-from", outdir + "/last.pt"]
        log(f"train {cfg} fold{n} -> {outdir}")
        # name ..._fold{n}_train.log so the stall-watchdog (glob *fold*_train.log) sees it
        lf = ROOT / "logs" / f"ablation_{Path(cfg).stem}_fold{n}_train.log"
        with open(lf, "a", encoding="utf-8") as f:
            subprocess.Popen(cmd, cwd=str(ROOT), env=env(), stdout=f, stderr=subprocess.STDOUT).wait()
        time.sleep(15)


def evaluate(cfg, ckdir, evaldir, folds, skip_external=False):
    wait_gpu()
    cmd = [PY, "scripts/_phase3_eval_full.py", "--config", cfg,
           "--ckpt-fmt", f"{ckdir}/fold{{n}}/best.pt", "--folds", folds, "--out", evaldir]
    if skip_external:
        cmd.append("--skip-external")   # keep the external test PRISTINE during selection
    log(f"eval {cfg} folds={folds} skip_external={skip_external} -> {evaldir}")
    lf = ROOT / "logs" / f"ablation_eval_{Path(cfg).stem}.log"
    with open(lf, "a", encoding="utf-8") as f:
        subprocess.Popen(cmd, cwd=str(ROOT), env=env(), stdout=f, stderr=subprocess.STDOUT).wait()
    try:
        return json.loads((ROOT / evaldir / "metrics_full.json").read_text())
    except Exception as e:
        log(f"eval read failed: {e}"); return None


MEL_FLOOR = 0.80   # melanoma-recall safety floor for model selection (reviewer catch #2)

def score(metrics):
    """Select the winner on the IN-DISTRIBUTION validation (OOF) split, NOT the
    external test (reviewer catch #1: keep the held-out external set pristine, used
    only for the final ensemble number). A melanoma-recall floor disqualifies any
    candidate that trades cancer safety for common-class recovery (catch #2)."""
    if not metrics:
        return -1, {}
    sel = metrics.get("OOF_cv")
    if not sel:
        return -1, {}
    bal = sel["balanced_accuracy"]
    pc = {d["class"]: d["recall"] for d in sel["per_class"]}
    coll = sum(pc.get(c, 0) for c in COLLAPSED) / len(COLLAPSED)
    mel = pc.get("mel", 0.0)
    s = bal + coll - (1.0 if mel < MEL_FLOOR else 0.0)   # heavy penalty if melanoma unsafe
    return s, {"sel_bal_acc": bal, "collapsed_recall": round(coll, 3),
               "mel_recall": round(mel, 3), "mel_floor_ok": bool(mel >= MEL_FLOOR),
               "oof_macro_f1": sel["macro_f1"]}


def main():
    log("STAGE 1 — fold-0 bake-off of fixes A/B/C (winner chosen on OOF validation)")
    results = {}
    for k, cfg in FIXES.items():
        train_fold(cfg, CKDIR[k], 0)
        # selection eval: OOF only, external test kept pristine
        m = evaluate(cfg, CKDIR[k], EVALDIR[k] + "_fold0", "0", skip_external=True)
        s, summary = score(m)
        results[k] = (s, summary)
        log(f"FIX {k}: score={s:.3f} {summary}")

    winner = max(results, key=lambda k: results[k][0])
    log(f"STAGE 2 — WINNER = FIX {winner} {results[winner][1]}. Training folds 1-4.")
    for n in (1, 2, 3, 4):
        train_fold(FIXES[winner], CKDIR[winner], n)

    log("STAGE 3 — full 5-fold publication eval of winner (external ensemble = the headline number)")
    final = evaluate(FIXES[winner], CKDIR[winner], EVALDIR[winner] + "_full", "0,1,2,3,4", skip_external=False)
    log(f"DONE. Winner FIX {winner}. Final external ensemble: "
        f"{final['EXTERNAL_TEST_ensemble'] if final else 'eval-failed'}")
    log(f"All fix fold-0 scores: { {k: results[k][1] for k in results} }")


if __name__ == "__main__":
    main()
