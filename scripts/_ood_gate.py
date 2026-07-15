"""OOD / input-quality gate (trust) — Mahalanobis detector on DINOv2 embeddings.

Extracts the backbone CLS embedding for the in-distribution reference (train_val)
and the external test, fits class-conditional Gaussians with tied covariance
(Lee et al. 2018), and scores inputs by min Mahalanobis distance to any class mean.
Non-skin / unlike-training inputs get a high distance -> the app abstains ("this
doesn't look like a skin lesion I was trained on").

Validates with synthetic OOD (uniform noise, solid colours) which SHOULD score OOD.
Saves the fitted detector (means + precision + threshold) for the app.

    python scripts/_ood_gate.py --config configs/phase3_fixC_cbfocal_only.yaml \
        --ckpt checkpoints/phase3_fixC/fold0/best.pt --out eval/phase3_fixC_full/ood_gate.npz
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from mnkai.data.datasets import SkinDataset, TabularSpec
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.utils.config import load_config
from scripts._phase3_eval_full import build_model_for
from torch.utils.data import DataLoader


@torch.no_grad()
def embed(model, df, classes, tab, tfm, device, bs=48, limit=None):
    if limit:
        df = df.sample(min(limit, len(df)), random_state=0).reset_index(drop=True)
    ds = SkinDataset(df, classes=classes, transform=tfm, images_dir=None, tabular=tab)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=False)
    bb = model.backbone; out = []
    for b in dl:
        x = b["image"].to(device, non_blocking=True)
        f = bb.forward_features(x)
        cls = f[:, 0] if f.dim() == 3 else (f.mean(dim=(2, 3)) if f.dim() == 4 else f)
        out.append(cls.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_batch(model, x, device):
    bb = model.backbone
    f = bb.forward_features(x.to(device))
    return (f[:, 0] if f.dim() == 3 else f.mean(dim=(2, 3))).float().cpu().numpy()


def fit_mahalanobis(X, y, C):
    d = X.shape[1]; means = np.zeros((C, d))
    Xc = X.copy().astype(np.float64)
    for c in range(C):
        m = X[y == c]
        means[c] = m.mean(0) if len(m) else 0.0
        Xc[y == c] -= means[c]
    cov = np.cov(Xc.T) + 1e-3 * np.eye(d)      # tied covariance + ridge
    prec = np.linalg.inv(cov)
    return means, prec


def maha_score(X, means, prec):
    """Return min Mahalanobis distance^2 to any class mean (lower = in-dist)."""
    n, C = X.shape[0], means.shape[0]
    dists = np.zeros((n, C))
    for c in range(C):
        diff = X - means[c]
        dists[:, c] = np.einsum("ij,jk,ik->i", diff, prec, diff)
    return dists.min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-limit", type=int, default=12000, help="cap reference embeddings for speed")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ood] device={device}", flush=True)
    cfg = load_config(args.config)
    classes = list(cfg["classes"]["names"]); C = len(classes); cidx = {c: i for i, c in enumerate(classes)}
    tab_cfg = cfg.get("tabular", {}) or {}
    tab = TabularSpec(use_sex=tab_cfg.get("use_sex", True), use_age=tab_cfg.get("use_age", True),
                      use_site=tab_cfg.get("use_site", True),
                      use_fitzpatrick=tab_cfg.get("use_fitzpatrick", True)) if tab_cfg.get("enabled", False) else None

    norm = LabelNormalizer(classes=classes)
    manifest = normalize_manifest(pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False), normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    manifest = manifest.merge(splits[["filepath", "split", "fold"]], on="filepath", how="inner")
    aug = AugConfig(img_size=int(cfg["data"]["img_size"]),
                    shades_of_gray_p=0.0, hair_removal=False)
    tfm = build_valid(aug)

    ref_df = manifest[manifest["split"] == "train_val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "external_test"].reset_index(drop=True)

    model = build_model_for(cfg, classes, tab, device)
    obj = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = obj.get("ema_state") or obj.get("model_state") or obj
    model.load_state_dict(sd, strict=False)

    t0 = time.time()
    ref_sub = ref_df.sample(min(args.ref_limit, len(ref_df)), random_state=0).reset_index(drop=True)
    Xref = embed(model, ref_sub, classes, tab, tfm, device)
    yref = ref_sub["label_name"].map(cidx).values.astype(int)
    Xtest = embed(model, test_df, classes, tab, tfm, device)
    print(f"[ood] embedded ref={len(Xref)} test={len(Xtest)} in {(time.time()-t0)/60:.1f}m", flush=True)

    # split ref: 80% fit, 20% in-dist calibration
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xref)); cut = int(0.8 * len(Xref))
    fit_i, cal_i = idx[:cut], idx[cut:]
    means, prec = fit_mahalanobis(Xref[fit_i], yref[fit_i], C)

    s_cal = maha_score(Xref[cal_i], means, prec)     # in-dist calibration
    s_test = maha_score(Xtest, means, prec)          # external (in-dist, mild shift)
    thr = float(np.percentile(s_cal, 95))            # flag top 5% of in-dist as boundary

    # synthetic OOD: uniform noise + solid colours (simulate non-skin garbage)
    sz = int(cfg["data"]["img_size"])
    g = torch.Generator().manual_seed(0)
    noise = torch.rand(32, 3, sz, sz, generator=g)
    solid = torch.stack([torch.full((3, sz, sz), v) for v in np.linspace(0, 1, 16)])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    ood_x = torch.cat([noise, solid], 0)
    ood_x = (ood_x - mean) / std
    s_ood = maha_score(embed_batch(model, ood_x, device), means, prec)

    res = {
        "threshold_p95_indist": round(thr, 2),
        "in_dist_cal": {"median": round(float(np.median(s_cal)), 1), "p95": round(thr, 1)},
        "external": {"median": round(float(np.median(s_test)), 1),
                     "flagged_ood_frac": round(float((s_test > thr).mean()), 3)},
        "synthetic_ood": {"median": round(float(np.median(s_ood)), 1),
                          "detected_frac": round(float((s_ood > thr).mean()), 3)},
    }
    np.savez_compressed(args.out, means=means, prec=prec, threshold=thr, classes=np.array(classes))
    Path(str(args.out).replace(".npz", ".json")).write_text(json.dumps(res, indent=2))
    print("\n===== OOD gate (Mahalanobis on DINOv2 CLS) =====")
    print(f"  in-dist cal : median {res['in_dist_cal']['median']}  (threshold p95 = {thr:.0f})")
    print(f"  external    : median {res['external']['median']}  flagged-OOD {res['external']['flagged_ood_frac']:.1%} (should be ~5%)")
    print(f"  synthetic OOD: median {res['synthetic_ood']['median']}  detected {res['synthetic_ood']['detected_frac']:.1%} (should be ~100%)")
    print(f"  saved detector -> {args.out}\n")


if __name__ == "__main__":
    main()
