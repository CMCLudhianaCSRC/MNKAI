"""Linear-probe a frozen backbone on DDI (5-fold CV) — the make-or-break test of
the deep-research finding that 0.58 is a BACKBONE problem, not a data ceiling.

Extract frozen features for all 656 DDI images, then 5-fold stratified logistic
regression (malignant vs benign), report AUROC overall + per Fitzpatrick tone.
Compare backbones: DINOv2 (our current, ImageNet) vs PanDerm (clinical derm FM).

    python scripts/_probe_backbone.py --backbone dinov2 --out eval/probe/ddi_dinov2.json
    python scripts/_probe_backbone.py --backbone panderm --ckpt <panderm.pth> --out eval/probe/ddi_panderm.json
"""
from __future__ import annotations
import argparse, json, os, sys
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

# DDI image root. Override with --ddi-root or the MNKAI_DDI_ROOT env var;
# the original hardcoded D: path remains the default so existing runs are
# unchanged, but the path is no longer baked into the source.
DEFAULT_DDI_ROOT = os.environ.get("MNKAI_DDI_ROOT", "D:/AIFINAL/images/ddi")
TONE = {12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"}


def load_backbone(name, ckpt, device, img_size=224):
    if name == "dinov2":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=img_size)
    elif name == "panderm":
        # PanDerm ViT-B: load our timm ViT-B skeleton, then overlay the checkpoint's state dict
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("model") or obj.get("state_dict") or obj
        sd = {k.replace("backbone.", "").replace("module.", ""): v for k, v in sd.items()}
        miss, unexp = m.load_state_dict(sd, strict=False)
        print(f"[panderm] loaded; missing={len(miss)} unexpected={len(unexp)}", flush=True)
    else:
        raise ValueError(name)
    return m.to(device).eval()


@torch.no_grad()
def extract(model, files, tfm, device, bs=32):
    feats = []
    for i in range(0, len(files), bs):
        batch = torch.stack([tfm(Image.open(f).convert("RGB")) for f in files[i:i+bs]]).to(device)
        feats.append(model(batch).float().cpu().numpy())
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--ddi-root", default=DEFAULT_DDI_ROOT,
                    help="DDI image root (default: $MNKAI_DDI_ROOT or D:/AIFINAL/images/ddi)")
    a = ap.parse_args()
    np.random.seed(0)
    ddi = Path(a.ddi_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(a.backbone, a.ckpt, dev, a.img_size)
    cfg = resolve_data_config({"input_size": (3, a.img_size, a.img_size)}, model=model); tfm = create_transform(**cfg)

    meta = pd.read_csv(ddi / "ddi_metadata.csv")
    files = [str(ddi / f) for f in meta["DDI_file"]]
    y = meta["malignant"].astype(int).values
    tone = meta["skin_tone"].values
    print(f"[probe] extracting {a.backbone} features for {len(files)} DDI images...", flush=True)
    X = extract(model, files, tfm, dev)
    print(f"[probe] features {X.shape}", flush=True)

    # 5-fold stratified CV, class-balanced LR; aggregate held-out scores
    strat = pd.Series(tone).astype(str) + "_" + pd.Series(y).astype(str)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = np.zeros(len(y))
    for tr, te in skf.split(X, strat):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        scores[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    res = {"backbone": a.backbone, "n": len(y), "feat_dim": int(X.shape[1]),
           "overall_auroc": round(roc_auc_score(y, scores), 4)}
    for k, nm in TONE.items():
        m = tone == k
        try: res[nm] = round(roc_auc_score(y[m], scores[m]), 4)
        except Exception: res[nm] = None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n===== LINEAR PROBE on DDI ({a.backbone}) =====")
    print(f"  OVERALL AUROC {res['overall_auroc']}   (old fine-tune: 0.58 | target ~0.73-0.78)")
    for nm in TONE.values():
        print(f"  {nm:12s} {res[nm]}")
    print(f"  saved -> {a.out}\n")


if __name__ == "__main__":
    main()
