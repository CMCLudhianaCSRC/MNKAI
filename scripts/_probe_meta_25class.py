"""Metadata-fused frozen-probe for the 25-class: concat 518 image features with
encoded clinical metadata (age / sex / Fitzpatrick), then probe on the de-duplicated
external. Tests whether the metadata the cRT model used adds signal to the probe.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import log_softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts._optimize_thresholds import coordinate_ascent
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.utils.config import load_config


def encode_meta(df):
    age = pd.to_numeric(df.get("age_approx"), errors="coerce").values
    age_norm = np.nan_to_num(age / 100.0); age_miss = np.isnan(age).astype(float)
    sex = df.get("sex", pd.Series([None]*len(df))).astype(str).str.lower().values
    male = (sex == "male").astype(float); fem = (sex == "female").astype(float); sunk = 1 - male - fem
    fst = pd.to_numeric(df.get("fitzpatrick"), errors="coerce").values
    fcols = [(fst == k).astype(float) for k in range(1, 7)]; funk = np.isnan(fst).astype(float)
    return np.column_stack([age_norm, age_miss, male, fem, sunk, *fcols, funk]).astype(np.float32)


def main():
    z = np.load("eval/probe/feats_25class_518.npz", allow_pickle=True)
    Ximg = z["features"].astype(np.float32); y = z["y"]; split = z["split"].astype(str); fold = z["fold"]; classes = [str(c) for c in z["classes"]]; C = len(classes)
    cfg = load_config("configs/phase3_indian_25class.yaml")
    man = normalize_manifest(pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False), LabelNormalizer(classes=classes))
    sp = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    k = "filepath" if "filepath" in sp.columns else "isic_id"
    man = man.merge(sp[[k, "split", "fold"]], on=k, how="inner").reset_index(drop=True)
    assert len(man) == len(y), f"align mismatch {len(man)} vs {len(y)}"
    assert (man["label_name"].map({c: i for i, c in enumerate(classes)}).values == y).mean() > 0.99, "label misalign"
    M = encode_meta(man)
    for name, X in [("image-only", Ximg), ("image+meta", np.concatenate([Ximg, M], 1))]:
        tv = split == "train_val"; ex = split == "external_test"
        Xtv, ytv, ftv = X[tv], y[tv], fold[tv]; Xex, yex = X[ex], y[ex]
        # dedup mask (from 518 image feats)
        i5 = Ximg[tv]; e5 = Ximg[ex]
        tvn = i5/(np.linalg.norm(i5,axis=1,keepdims=True)+1e-8); exn = e5/(np.linalg.norm(e5,axis=1,keepdims=True)+1e-8)
        mx = np.zeros(len(exn))
        for i in range(0,len(exn),256): mx[i:i+256]=(exn[i:i+256]@tvn.T).max(1)
        clean = mx <= 0.99
        oof = np.zeros((len(ytv), C))
        for fk in sorted(set(ftv[ftv>=0])):
            tr=ftv!=fk; te=ftv==fk; sc=StandardScaler().fit(Xtv[tr]); m=MLPClassifier((512,),max_iter=300,early_stopping=True,alpha=1e-3).fit(sc.transform(Xtv[tr]),ytv[tr]); oof[te]=m.predict_proba(sc.transform(Xtv[te]))
        bias,_,_=coordinate_ascent(log_softmax(np.log(np.clip(oof,1e-9,1)),1),ytv,C,rounds=3)
        sc=StandardScaler().fit(Xtv); m=MLPClassifier((512,),max_iter=300,early_stopping=True,alpha=1e-3).fit(sc.transform(Xtv),ytv)
        p=(log_softmax(np.log(np.clip(m.predict_proba(sc.transform(Xex)),1e-9,1)),1)+bias).argmax(1)
        print(f"  {name:12s}: CLEAN macro-F1 {f1_score(yex[clean],p[clean],average='macro'):.4f}  top1 {accuracy_score(yex[clean],p[clean]):.3f}", flush=True)


if __name__ == "__main__":
    print("=== metadata fusion (25-class, clean external) ===", flush=True)
    main()
