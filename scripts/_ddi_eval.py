"""DDI as a 2nd EXTERNAL test + powered dark-skin fairness eval.

Runs the cRT 5-fold ensemble (+ per-class bias) on the 656 DDI images (never
trained on) and reports the binary malignant-vs-benign performance OVERALL and
PER SKIN TONE (12=FST I-II, 34=III-IV, 56=V-VI). DDI's `malignant` label is the
ground truth. Our malignant score = P(mel)+P(bcc)+P(scc)+P(ak).

Note: DDI's malignant set is broader than our 4 malignant classes (it includes
mycosis fungoides, SCC-in-situ, etc.), so measured sensitivity is a conservative
lower bound — an honest, hard external test.

    python scripts/_ddi_eval.py --config configs/phase3_crt_a1_full.yaml \
        --ckpt-fmt checkpoints/phase3_crt_full/fold{n}/best.pt \
        --bias eval/phase3_crt_full/threshold_opt_5fold.json \
        --out eval/phase3_crt_full/ddi_eval.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from mnkai.data.datasets import TabularSpec
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.utils.config import load_config
from scripts._phase3_eval_full import build_model_for, infer

DDI_DIR = Path("D:/AIFINAL/images/ddi")
MALIGNANT = ["mel", "bcc", "scc", "ak"]
TONE = {12: "FST I-III (light)", 34: "FST III-IV (mid)", 56: "FST V-VI (dark)"}
TONE_FST = {12: 2, 34: 4, 56: 6}


def sens_spec(score, y, thr):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"sensitivity": round(tp / (tp + fn + 1e-9), 3), "specificity": round(tn / (tn + fp + 1e-9), 3),
            "n_malignant": tp + fn, "n_benign": tn + fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-fmt", required=True)
    ap.add_argument("--bias", default=None)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [int(x) for x in args.folds.split(",")]
    cfg = load_config(args.config)
    classes = list(cfg["classes"]["names"]); C = len(classes)
    tab_cfg = cfg.get("tabular", {}) or {}
    tab = TabularSpec(use_sex=tab_cfg.get("use_sex", True), use_age=tab_cfg.get("use_age", True),
                      use_site=tab_cfg.get("use_site", True),
                      use_fitzpatrick=tab_cfg.get("use_fitzpatrick", True)) if tab_cfg.get("enabled", False) else None
    tfm = build_valid(AugConfig(img_size=int(cfg["data"]["img_size"]), shades_of_gray_p=0.0, hair_removal=False))

    # DDI dataframe (mirror training manifest columns)
    meta = pd.read_csv(DDI_DIR / "ddi_metadata.csv")
    df = pd.DataFrame({
        "filepath": [str(DDI_DIR / f) for f in meta["DDI_file"]],
        "label_name": "normal",                       # placeholder; binary label from metadata
        "fitzpatrick": meta["skin_tone"].map(TONE_FST).astype(float),
        "sex": np.nan, "age_approx": np.nan,
    })
    y_mal = meta["malignant"].astype(int).values
    tone = meta["skin_tone"].values
    print(f"[ddi] {len(df)} images; malignant={int(y_mal.sum())}", flush=True)

    # 5-fold ensemble inference
    prob_sum = np.zeros((len(df), C))
    for n in folds:
        cfg["train"]["fold"] = n
        m = build_model_for(cfg, classes, tab, device)
        obj = torch.load(args.ckpt_fmt.format(n=n), map_location=device, weights_only=False)
        sd = obj.get("ema_state") or obj.get("model_state") or obj
        m.load_state_dict(sd, strict=False)
        lg = infer(m, df, classes, tab, tfm, device)
        prob_sum += F.softmax(torch.from_numpy(lg), dim=-1).numpy()
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"[ddi] fold {n} done", flush=True)
    probs = prob_sum / len(folds)

    # optional per-class bias
    if args.bias and Path(args.bias).is_file():
        d = json.loads(Path(args.bias).read_text())
        bmap = {c: float(b) for c, b in zip([str(x).lower() for x in d["classes"]], d["bias"])}
        bias = np.array([bmap.get(c, 0.0) for c in classes])
        logit = np.log(np.clip(probs, 1e-9, 1)) + bias
        probs = np.exp(logit - logit.max(1, keepdims=True))
        probs = probs / probs.sum(1, keepdims=True)

    mal_idx = [classes.index(c) for c in MALIGNANT if c in classes]
    score = probs[:, mal_idx].sum(1)

    def block(mask, name):
        s, yy = score[mask], y_mal[mask]
        out = {"group": name, "n": int(mask.sum()), "n_malignant": int(yy.sum())}
        try: out["auroc"] = round(roc_auc_score(yy, s), 4)
        except Exception: out["auroc"] = None
        out["at_0.30"] = sens_spec(s, yy, 0.30)
        out["at_0.20"] = sens_spec(s, yy, 0.20)
        return out

    res = {"dataset": "DDI", "n": len(df), "model": "cRT 5-fold + bias",
           "overall": block(np.ones(len(df), bool), "overall"),
           "by_skin_tone": [block(tone == k, TONE[k]) for k in (12, 34, 56)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    print("\n===== DDI external cancer-triage eval (cRT 5-fold + bias) =====")
    o = res["overall"]; print(f"  OVERALL: AUROC {o['auroc']} | @0.20 sens {o['at_0.20']['sensitivity']} spec {o['at_0.20']['specificity']} (n={o['n']}, mal={o['n_malignant']})")
    print("  --- per skin tone (the powered fairness result) ---")
    for b in res["by_skin_tone"]:
        print(f"  {b['group']:20s}: AUROC {b['auroc']} | @0.20 sens {b['at_0.20']['sensitivity']} spec {b['at_0.20']['specificity']} (n={b['n']}, mal={b['n_malignant']})")
    print(f"  saved -> {args.out}\n")


if __name__ == "__main__":
    main()
