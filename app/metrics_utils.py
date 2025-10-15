# app/metrics_utils.py
import json
from pathlib import Path

def load_valid_report(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def extract_scalar_metrics(vrpt: dict) -> dict:
    """Pull top-level scalars safely (loss, accuracy, macro F1, macro AUC, ECE)."""
    if not vrpt:
        return {}
    scalars = {"val_loss": float(vrpt.get("loss", 0.0))}
    rep = vrpt.get("report", {}) if isinstance(vrpt.get("report", {}), dict) else {}
    acc = rep.get("accuracy", vrpt.get("accuracy", None))
    if acc is not None:
        scalars["accuracy"] = float(acc)
    macro = rep.get("macro avg", {})
    if isinstance(macro, dict) and "f1-score" in macro:
        scalars["macro_f1"] = float(macro["f1-score"])
    if "macro_auc_ovr" in vrpt and vrpt["macro_auc_ovr"] is not None:
        scalars["macro_auc_ovr"] = float(vrpt["macro_auc_ovr"])
    if "ece" in vrpt and vrpt["ece"] is not None:
        scalars["ece"] = float(vrpt["ece"])
    return scalars

def per_class_table(vrpt: dict):
    """Return rows: [class, precision, recall, f1, support]."""
    rows = []
    if not vrpt or "report" not in vrpt or not isinstance(vrpt["report"], dict):
        return rows
    for cls, vals in vrpt["report"].items():
        if cls in ("accuracy", "macro avg", "weighted avg", "micro avg") or not isinstance(vals, dict):
            continue
        rows.append([
            str(cls),
            float(vals.get("precision", 0.0)),
            float(vals.get("recall", 0.0)),
            float(vals.get("f1-score", 0.0)),
            int(vals.get("support", 0)),
        ])
    return rows

def pretty_markdown(scalars: dict, path: str) -> str:
    if not scalars:
        return f"**No report found** at `{path}`."
    parts = [f"**Report:** `{path}`\n"]
    def fmt(k, label):
        return f"- **{label}:** {scalars[k]:.4f}" if k in scalars else None
    for k, lbl in [("val_loss", "Val Loss"),
                   ("accuracy", "Accuracy"),
                   ("macro_f1", "Macro F1"),
                   ("macro_auc_ovr", "Macro AUC (OvR)"),
                   ("ece", "ECE")]:
        s = fmt(k, lbl)
        if s: parts.append(s)
    return "\n".join(parts)
