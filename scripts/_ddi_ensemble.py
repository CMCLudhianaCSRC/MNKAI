"""DDI dark-skin cancer — FROZEN foundation-model ENSEMBLE (the #1 leaderboard lever).
Extract features from several strong frozen backbones, then ensemble (concat-features
and score-average) a class-balanced probe. 5-fold stratified CV; AUROC overall + per tone.
Baseline: single DINOv2@518 = 0.748 overall / 0.789 V-VI.

    python scripts/_ddi_ensemble.py
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")  # fail fast on a stalled HF socket instead of hanging forever
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import timm
from timm.data import resolve_data_config, create_transform

DDI = Path("D:/AIFINAL/images/ddi")
CACHE = ROOT / "eval" / "probe" / "ddi_backbones"; CACHE.mkdir(parents=True, exist_ok=True)
TONE = {12: "FST I-III", 34: "FST III-IV", 56: "FST V-VI"}
# NOTE: dinov2_l is LAST — it is the biggest download and least additive (another DINOv2);
# the ensemble gain comes from the DIVERSE backbones (eva02, convnextv2, siglip) which run first.
# If dinov2_l stalls/fails to download, the per-backbone try/except skips it and the ensemble
# still runs on whatever succeeded.
BACKBONES = [
    ("dinov2_b", "vit_base_patch14_dinov2.lvd142m", 518, 16),          # cached
    ("dinov2_reg", "vit_base_patch14_reg4_dinov2.lvd142m", 518, 16),   # cached
    ("eva02_b", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", 448, 12),
    ("convnextv2_b", "convnextv2_base.fcmae_ft_in22k_in1k", 384, 24),
    ("siglip_b", "vit_base_patch16_siglip_384.webli", 384, 24),
    ("dinov2_l", "vit_large_patch14_dinov2.lvd142m", 518, 6),          # biggest download, run last
]


@torch.no_grad()
def extract(name, model_id, img_size, bs, files, dev):
    cf = CACHE / f"{name}.npy"
    if cf.exists():
        return np.load(cf)
    m = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=img_size).to(dev).eval() \
        if "convnext" not in model_id else timm.create_model(model_id, pretrained=True, num_classes=0).to(dev).eval()
    tfm = create_transform(**resolve_data_config({"input_size": (3, img_size, img_size)}, model=m))
    feats = []
    for i in range(0, len(files), bs):
        imgs = torch.stack([tfm(Image.open(f).convert("RGB")) for f in files[i:i+bs]]).to(dev)
        feats.append(m(imgs).float().cpu().numpy())
    F = np.concatenate(feats); np.save(cf, F); del m; torch.cuda.empty_cache()
    print(f"  [{name}] {F.shape}", flush=True)
    return F


def cv_scores(X, y, strat):
    """5-fold stratified CV -> OOF P(malignant)."""
    s = np.zeros(len(y)); skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, strat):
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return s


def auroc_by_tone(score, y, tone):
    out = {"overall": round(roc_auc_score(y, score), 4)}
    for k, nm in TONE.items():
        m = tone == k
        try: out[nm] = round(roc_auc_score(y[m], score[m]), 4)
        except Exception: out[nm] = None
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(DDI / "ddi_metadata.csv")
    files = [str(DDI / f) for f in meta["DDI_file"]]
    y = meta["malignant"].astype(int).values; tone = meta["skin_tone"].values
    strat = pd.Series(tone).astype(str) + "_" + pd.Series(y).astype(str)

    feats = {}; per = {}
    print("=== extracting + probing each backbone ===", flush=True)
    for name, mid, sz, bs in BACKBONES:
        try:
            F = extract(name, mid, sz, bs, files, dev); feats[name] = F
            per[name] = auroc_by_tone(cv_scores(F, y, strat), y, tone)
            print(f"  {name:14s} overall {per[name]['overall']}  V-VI {per[name]['FST V-VI']}", flush=True)
        except Exception as e:
            print(f"  {name}: FAILED {str(e)[:80]}", flush=True)

    names = list(feats)
    # Ensemble A: concat features
    Xcat = np.concatenate([feats[n] for n in names], 1)
    ensA = auroc_by_tone(cv_scores(Xcat, y, strat), y, tone)
    # Ensemble B: average per-backbone OOF scores
    sc_stack = np.column_stack([cv_scores(feats[n], y, strat) for n in names])
    ensB = auroc_by_tone(sc_stack.mean(1), y, tone)

    res = {"per_backbone": per, "ensemble_concat": ensA, "ensemble_scoreavg": ensB,
           "baseline_dinov2_518": {"overall": 0.748, "FST V-VI": 0.789}}
    (ROOT / "eval" / "probe" / "ddi_ensemble.json").write_text(json.dumps(res, indent=2))
    print("\n===== DDI ENSEMBLE RESULT =====")
    print(f"  baseline single DINOv2@518 : overall 0.748  V-VI 0.789")
    print(f"  ensemble (concat features) : overall {ensA['overall']}  V-VI {ensA['FST V-VI']}")
    print(f"  ensemble (score average)   : overall {ensB['overall']}  V-VI {ensB['FST V-VI']}")
    print("  saved -> eval/probe/ddi_ensemble.json")


if __name__ == "__main__":
    main()
