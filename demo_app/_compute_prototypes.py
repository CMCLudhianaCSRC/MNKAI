"""Build the latent-space prototypes for the demo's embedding explorer.

Embeds a per-class sample of training images with the demo model, PCA-projects
to 2-D, and saves coords + per-class centroids + the PCA basis (so an uploaded
image can be projected into the same space). Run later (uses GPU/CPU inference):

  PYTHONPATH=src .venv/Scripts/python.exe demo_app/_compute_prototypes.py \
      --manifest data/manifests/master_clean.csv --per-class 60
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from inference import engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "data/manifests/master_clean.csv"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--out", default=str(HERE / "prototypes.npz"))
    args = ap.parse_args()
    eng = engine()
    df = pd.read_csv(args.manifest, low_memory=False)
    df = df[df["label_name"].isin(eng.classes)]
    embs, labs = [], []
    for c in eng.classes:
        sub = df[df["label_name"] == c]
        if len(sub) > args.per_class:
            sub = sub.sample(n=args.per_class, random_state=42)
        for fp in sub["filepath"]:
            try:
                img_t, _, _ = eng.preprocess(Image.open(fp))
                embs.append(eng.embedding(img_t)); labs.append(c)
            except Exception:
                continue
        print(f"[proto] {c}: {labs.count(c)} embeddings", flush=True)
    E = np.stack(embs); labs = np.array(labs)
    mean = E.mean(0); Ec = E - mean
    # PCA via SVD -> top 2 components
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    comp = Vt[:2]                          # (2, D)
    coords = Ec @ comp.T                   # (N, 2)
    cents = np.stack([E[labs == c].mean(0) for c in eng.classes])
    np.savez(args.out, coords=coords, labels=labs, classes=np.array(eng.classes),
             centroids=cents, pca_mean=mean, pca_comp=comp, method=np.array("pca"))
    print(f"[proto] saved {len(labs)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
