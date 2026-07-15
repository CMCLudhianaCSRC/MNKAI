"""Cache FROZEN backbone features for a config's manifest (the winning pattern:
frozen features + light head, which beat fine-tuning on DDI). Extract once, probe
many times. Saves features + labels + split/fold + fitzpatrick to .npz.

    python scripts/_cache_features.py --config configs/phase3_indian_25class.yaml \
        --backbone dinov2 --img-size 224 --out eval/probe/feats_25class_224.npz
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.utils.config import load_config
import timm
from timm.data import resolve_data_config, create_transform


def build(name, img_size, device):
    if name == "dinov2":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=img_size)
    elif name == "dinov2_reg":
        m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=img_size)
    else:
        raise ValueError(name)
    return m.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config)
    classes = [c.lower() for c in cfg["classes"]["names"]]; cidx = {c: i for i, c in enumerate(classes)}

    norm = LabelNormalizer(classes=classes)
    manifest = normalize_manifest(pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False), normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    key = "filepath" if "filepath" in splits.columns else "isic_id"
    manifest = manifest.merge(splits[[key, "split", "fold"]], on=key, how="inner").reset_index(drop=True)

    model = build(a.backbone, a.img_size, dev)
    dcfg = resolve_data_config({"input_size": (3, a.img_size, a.img_size)}, model=model)
    tfm = create_transform(**dcfg)

    fps = manifest["filepath"].tolist()
    y = manifest["label_name"].map(cidx).values.astype(int)
    fst = pd.to_numeric(manifest.get("fitzpatrick", pd.Series([np.nan]*len(manifest))), errors="coerce").values
    split = manifest["split"].values; fold = pd.to_numeric(manifest["fold"], errors="coerce").fillna(-1).astype(int).values
    print(f"[cache] {len(fps)} imgs, {a.backbone}@{a.img_size}, dev={dev}", flush=True)

    feats = np.zeros((len(fps), model.num_features), dtype=np.float32)
    t0 = time.time(); ok = np.ones(len(fps), bool)
    for i in range(0, len(fps), a.bs):
        chunk = fps[i:i+a.bs]; imgs = []
        for f in chunk:
            try: imgs.append(tfm(Image.open(f).convert("RGB")))
            except Exception: imgs.append(torch.zeros(3, a.img_size, a.img_size)); ok[i+len(imgs)-1] = False
        x = torch.stack(imgs).to(dev)
        feats[i:i+len(chunk)] = model(x).float().cpu().numpy()
        if (i // a.bs) % 100 == 0:
            print(f"[cache] {i}/{len(fps)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, features=feats, y=y, split=split, fold=fold, fst=fst.astype(np.float32),
                        classes=np.array(classes), ok=ok)
    print(f"[cache] saved {feats.shape} -> {a.out} ({(time.time()-t0)/60:.1f}m, {ok.sum()} valid)", flush=True)


if __name__ == "__main__":
    main()
