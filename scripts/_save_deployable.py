"""Train + save the deployable 25-class clinical head (the honest frozen-probe):
MLP + scaler + temperature + per-class bias + conformal qhat + coarse map. The app
loads this and applies it to DINOv2@518 features from a new image. No fine-tuning.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts._phase3_eval_full import fit_temperature
from scripts._optimize_thresholds import coordinate_ascent
from scripts._conformal_fix import raps_scores, qhat_of

z = np.load("eval/probe/feats_25class_518.npz", allow_pickle=True)
X = z["features"].astype(np.float32); y = z["y"]; split = z["split"].astype(str); fold = z["fold"]
classes = [str(c) for c in z["classes"]]; C = len(classes)
tv = split == "train_val"; Xtv, ytv, ftv = X[tv], y[tv], fold[tv]

# OOF for calibration/bias/conformal
oof = np.zeros((len(ytv), C))
for fk in sorted(set(ftv[ftv >= 0])):
    tr = ftv != fk; te = ftv == fk
    sc = StandardScaler().fit(Xtv[tr]); m = MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3).fit(sc.transform(Xtv[tr]), ytv[tr])
    oof[te] = m.predict_proba(sc.transform(Xtv[te]))
oof_log = np.log(np.clip(oof, 1e-9, 1))
T = fit_temperature(oof_log, ytv)
bias, _, _ = coordinate_ascent(log_softmax(oof_log, 1), ytv, C, rounds=3)
# conformal RAPS qhat (target 90%) on temp-scaled OOF
oof_cal = softmax(oof_log / T, 1)
np.random.seed(0); K_REG, LAM = 2, 0.05
qh = qhat_of(raps_scores(oof_cal, ytv, LAM, K_REG), 0.10)

# final head on ALL train_val
scaler = StandardScaler().fit(Xtv)
head = MLPClassifier((512,), max_iter=300, early_stopping=True, alpha=1e-3).fit(scaler.transform(Xtv), ytv)

out = Path("models"); out.mkdir(exist_ok=True)
with open(out / "clinical_25class.pkl", "wb") as f:
    pickle.dump({"classes": classes, "scaler": scaler, "head": head, "temperature": float(T),
                 "bias": bias.astype(np.float32), "conformal": {"qhat": float(qh), "k_reg": K_REG, "lambda": LAM, "alpha": 0.10},
                 "backbone": "vit_base_patch14_dinov2.lvd142m", "img_size": 518,
                 "note": "frozen-probe 25-class; honest deduped external macro-F1 ~0.573; ECE 0.016; conformal ~4-set@0.90"}, f)
print(f"saved models/clinical_25class.pkl | classes={C} T={T:.2f} qhat={qh:.3f}")
print(f"  bias range [{bias.min():.2f},{bias.max():.2f}]")
