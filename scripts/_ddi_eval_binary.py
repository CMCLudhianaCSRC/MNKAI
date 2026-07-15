"""Evaluate the binary clinical-photo cancer detector on DDI (held-out, biopsy).
Malignant score = softmax(logits)[:, 'malignant']. Reports AUROC overall + per
skin tone, vs the old 25-class model's 0.58 baseline.

    python scripts/_ddi_eval_binary.py --config configs/cancer_clinical_binary.yaml \
        --ckpt checkpoints/cancer_clinical/fold0/best.pt --out eval/cancer_clinical/ddi_binary.json
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

DDI = Path("D:/AIFINAL/images/ddi")
TONE = {12: "FST I-III (light)", 34: "FST III-IV (mid)", 56: "FST V-VI (dark)"}


def ss(score, y, thr):
    p = (score >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    return {"sens": round(tp / (tp + fn + 1e-9), 3), "spec": round(tn / (tn + fp + 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(a.config); classes = list(cfg["classes"]["names"])
    mal_i = classes.index("malignant")
    tfm = build_valid(AugConfig(img_size=int(cfg["data"]["img_size"]), shades_of_gray_p=0.0, hair_removal=False))
    meta = pd.read_csv(DDI / "ddi_metadata.csv")
    df = pd.DataFrame({"filepath": [str(DDI / f) for f in meta["DDI_file"]], "label_name": "benign"})
    y = meta["malignant"].astype(int).values; tone = meta["skin_tone"].values

    m = build_model_for(cfg, classes, None, dev)         # tab=None -> image-only
    o = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(o.get("ema_state") or o.get("model_state") or o, strict=False)
    logits = infer(m, df, classes, None, tfm, dev)
    score = F.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, mal_i]

    def blk(mask, name):
        s, yy = score[mask], y[mask]
        d = {"group": name, "n": int(mask.sum()), "n_mal": int(yy.sum())}
        try: d["auroc"] = round(roc_auc_score(yy, s), 4)
        except Exception: d["auroc"] = None
        d["at_0.5"] = ss(s, yy, 0.5); d["at_0.3"] = ss(s, yy, 0.3)
        return d
    res = {"model": "clinical binary cancer (ckpt best)", "ckpt_epoch": o.get("epoch"),
           "overall": blk(np.ones(len(df), bool), "overall"),
           "by_tone": [blk(tone == k, TONE[k]) for k in (12, 34, 56)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2))
    o_ = res["overall"]
    print("\n===== DDI — NEW clinical binary cancer detector =====")
    print(f"  OVERALL AUROC {o_['auroc']}  (old 25-class model: 0.58)  @0.5 sens {o_['at_0.5']['sens']} spec {o_['at_0.5']['spec']}")
    for b in res["by_tone"]:
        print(f"  {b['group']:20s} AUROC {b['auroc']}  (n={b['n']}, mal={b['n_mal']})")
    print(f"  saved -> {a.out}\n")


if __name__ == "__main__":
    main()
