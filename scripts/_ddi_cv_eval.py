"""Cross-validated DDI evaluation. Each DDI image is scored ONLY by the fold-model
that held it out (no leakage), then aggregated -> honest CV dark-skin cancer AUROC,
overall and per skin tone. Compare to the old 25-class model's 0.58.

    python scripts/_ddi_cv_eval.py --config configs/cancer_cv.yaml \
        --ckpt-fmt checkpoints/cancer_cv/fold{n}/best.pt --out eval/cancer_cv/ddi_cv.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.utils.config import load_config
from scripts._phase3_eval_full import build_model_for, infer

TONE = {2: "FST I-III (light)", 4: "FST III-IV (mid)", 6: "FST V-VI (dark)"}


def ss(score, y, thr):
    p = (score >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    return {"sens": round(tp / (tp + fn + 1e-9), 3), "spec": round(tn / (tn + fp + 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt-fmt", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config); classes = list(cfg["classes"]["names"]); mal_i = classes.index("malignant")
    tfm = build_valid(AugConfig(img_size=int(cfg["data"]["img_size"]), shades_of_gray_p=0.0, hair_removal=False))

    man = pd.read_csv(cfg["data"]["manifest_csv"])
    sp = pd.read_csv(cfg["data"]["splits_csv"])
    d = man.merge(sp[["filepath", "fold"]], on="filepath", how="inner")
    ddi = d[d["source"] == "DDI"].reset_index(drop=True)
    ddi["y"] = (ddi["label_name"] == "malignant").astype(int)
    ddi["tone"] = pd.to_numeric(ddi["fitzpatrick"], errors="coerce").fillna(0).astype(int)
    print(f"[cv] DDI images: {len(ddi)} (malignant {int(ddi.y.sum())})", flush=True)

    score = np.full(len(ddi), np.nan)
    for n in folds:
        sub = ddi[ddi["fold"] == n]
        if len(sub) == 0:
            continue
        cfg["train"]["fold"] = n
        m = build_model_for(cfg, classes, None, dev)
        o = torch.load(a.ckpt_fmt.format(n=n), map_location=dev, weights_only=False)
        m.load_state_dict(o.get("ema_state") or o.get("model_state") or o, strict=False)
        lg = infer(m, sub.reset_index(drop=True), classes, None, tfm, dev)
        score[sub.index] = F.softmax(torch.from_numpy(lg), dim=-1).numpy()[:, mal_i]
        del m
        if dev.type == "cuda": torch.cuda.empty_cache()
        print(f"[cv] fold {n}: scored {len(sub)} held-out DDI", flush=True)

    ok = ~np.isnan(score); s = score[ok]; y = ddi["y"].values[ok]; tone = ddi["tone"].values[ok]

    def blk(mask, name):
        ss_, yy = s[mask], y[mask]
        d_ = {"group": name, "n": int(mask.sum()), "n_mal": int(yy.sum())}
        try: d_["auroc"] = round(roc_auc_score(yy, ss_), 4)
        except Exception: d_["auroc"] = None
        d_["at_0.5"] = ss(ss_, yy, 0.5); d_["at_0.3"] = ss(ss_, yy, 0.3)
        return d_
    res = {"model": "clinical cancer 5-fold CV (CleanPatrick+DDI+PAD)", "n_scored": int(ok.sum()),
           "overall": blk(np.ones(len(s), bool), "overall"),
           "by_tone": [blk(tone == k, TONE[k]) for k in (2, 4, 6)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2))
    o_ = res["overall"]
    print("\n===== CROSS-VALIDATED DDI (new clinical cancer detector) =====")
    print(f"  OVERALL AUROC {o_['auroc']}  (old 25-class: 0.58, naive clinical: 0.46)")
    print(f"          @0.5 sens {o_['at_0.5']['sens']} spec {o_['at_0.5']['spec']} | @0.3 sens {o_['at_0.3']['sens']} spec {o_['at_0.3']['spec']}")
    for b in res["by_tone"]:
        print(f"  {b['group']:20s} AUROC {b['auroc']}  (n={b['n']}, mal={b['n_mal']})")
    print(f"  saved -> {a.out}\n")


if __name__ == "__main__":
    main()
