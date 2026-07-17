"""ANALYSIS 5 — SUPERVISED-ImageNet CONTROL (de-confound the central SSL claim).

Is DINOv2 the better DDI dark-skin cancer backbone because it is SELF-SUPERVISED,
or just because it is a stronger representation? This control isolates the pretraining
OBJECTIVE by matching architecture (ViT-B) and resolution (224px):

    supervised  : vit_base_patch16_224.augreg2_in21k_ft_in1k  (IN21k->IN1k supervised)
    self-superv : vit_base_patch14_dinov2.lvd142m @224         (DINOv2 SSL)

Both -> FROZEN features -> the SAME 5-fold stratified class-balanced LR probe used
everywhere in this project. AUROC overall + per Fitzpatrick tone, stratified bootstrap
95% CIs, and a PAIRED bootstrap dAUROC (DINOv2 - supervised) with a one-sided p-value.

Baseline to reproduce: DINOv2-B@224 overall 0.7364 (eval/probe/ddi_dinov2.json).

    ./.venv/Scripts/python.exe scripts/_ddi_supervised_control.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
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

DDI = Path(os.environ.get("MNKAI_DDI_ROOT", "D:/AIFINAL/images/ddi"))
CACHE = ROOT / "eval" / "probe" / "ddi_backbones"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "eval" / "probe" / "ddi_supervised_baseline.json"
TONE = {12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"}
N_BOOT = 4000
RNG = np.random.default_rng(12345)

# (cache_name, timm_id, img_size) — matched arch(ViT-B) + res(224)
SUPERVISED = ("sup_vitb224", "vit_base_patch16_224.augreg2_in21k_ft_in1k", 224)
DINOV2_224 = ("dinov2_b224", "vit_base_patch14_dinov2.lvd142m", 224)


@torch.no_grad()
def extract(cache_name, model_id, img_size, files, dev):
    cf = CACHE / f"{cache_name}.npy"
    if cf.exists():
        F = np.load(cf)
        print(f"  [{cache_name}] cached {F.shape}", flush=True)
        return F
    m = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=img_size).to(dev).eval()
    tfm = create_transform(**resolve_data_config({"input_size": (3, img_size, img_size)}, model=m))
    feats, bs = [], 32
    for i in range(0, len(files), bs):
        imgs = torch.stack([tfm(Image.open(f).convert("RGB")) for f in files[i:i+bs]]).to(dev)
        feats.append(m(imgs).float().cpu().numpy())
    F = np.concatenate(feats); np.save(cf, F); del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  [{cache_name}] extracted {F.shape} -> {cf.name}", flush=True)
    return F


def oof_scores(X, y, strat, seed=0):
    s = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, strat):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return s


def auroc_by_tone(score, y, tone):
    out = {"overall": round(float(roc_auc_score(y, score)), 4)}
    for k, nm in TONE.items():
        m = tone == k
        out[nm] = round(float(roc_auc_score(y[m], score[m])), 4) if len(np.unique(y[m])) > 1 else None
    return out


def strat_boot_ci(score, y, tone, n=N_BOOT):
    key = tone.astype(int) * 10 + y.astype(int)
    cells = {c: np.where(key == c)[0] for c in np.unique(key)}
    keys = list(cells)
    boot = {k: [] for k in ["overall", *TONE.values()]}
    for _ in range(n):
        idx = np.concatenate([RNG.choice(cells[c], size=len(cells[c]), replace=True) for c in keys])
        bs, by, bt = score[idx], y[idx], tone[idx]
        boot["overall"].append(roc_auc_score(by, bs))
        for kk, nm in TONE.items():
            m = bt == kk
            if len(np.unique(by[m])) > 1:
                boot[nm].append(roc_auc_score(by[m], bs[m]))
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in boot.items() if v}


def paired_delta(score_a, score_b, y, tone, n=N_BOOT):
    """Paired bootstrap dAUROC = A - B on the SAME resampled indices. H0: A <= B."""
    key = tone.astype(int) * 10 + y.astype(int)
    cells = {c: np.where(key == c)[0] for c in np.unique(key)}
    keys = list(cells)
    out = {}
    for label, mask in [("overall", np.ones(len(y), bool))] + [(nm, tone == kk) for kk, nm in TONE.items()]:
        deltas = []
        for _ in range(n):
            idx = np.concatenate([RNG.choice(cells[c], size=len(cells[c]), replace=True) for c in keys])
            idx = idx[mask[idx]]
            if len(np.unique(y[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
        if not deltas:
            out[label] = None; continue
        deltas = np.asarray(deltas)
        out[label] = {
            "delta_auroc": round(float(np.mean(deltas)), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
            "p_one_sided": round(float(np.mean(deltas <= 0)), 4),
        }
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(DDI / "ddi_metadata.csv")
    files = [str(DDI / f) for f in meta["DDI_file"]]
    y = meta["malignant"].astype(int).to_numpy()
    tone = meta["skin_tone"].to_numpy()
    strat = (pd.Series(tone).astype(str) + "_" + pd.Series(y).astype(str)).to_numpy()

    print(f"=== ANALYSIS 5: supervised-ImageNet control (device={dev}) ===", flush=True)
    Xsup = extract(*SUPERVISED, files, dev)
    Xdino = extract(*DINOV2_224, files, dev)

    sup_s = oof_scores(Xsup, y, strat, seed=0)
    dino_s = oof_scores(Xdino, y, strat, seed=0)

    res = {
        "control": "Matched architecture (ViT-B) + resolution (224px); only the pretraining "
                   "OBJECTIVE differs. Supervised = vit_base_patch16_224.augreg2_in21k_ft_in1k; "
                   "SSL = vit_base_patch14_dinov2.lvd142m@224. Frozen features -> identical 5-fold "
                   "stratified class-balanced LR probe (Daneshjou et al. 2022 protocol).",
        "n": int(len(y)),
        "supervised_vitb224": {
            "timm_id": SUPERVISED[1], "feat_dim": int(Xsup.shape[1]),
            "auroc": auroc_by_tone(sup_s, y, tone),
            "bootstrap_ci95": strat_boot_ci(sup_s, y, tone),
        },
        "ssl_dinov2_b224": {
            "timm_id": DINOV2_224[1], "feat_dim": int(Xdino.shape[1]),
            "auroc": auroc_by_tone(dino_s, y, tone),
            "bootstrap_ci95": strat_boot_ci(dino_s, y, tone),
            "reference_ddi_dinov2_json_overall": 0.7364,
        },
        "delta_ssl_minus_supervised": paired_delta(dino_s, sup_s, y, tone),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
