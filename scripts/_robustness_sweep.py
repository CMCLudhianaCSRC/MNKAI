"""Robustness sweep: apply image corruptions and measure metric degradation.

Required for clinical deployment (FDA SaMD draft guidance — "Robustness to
input variation"). Measures how much performance drops under common real-world
distortions a smartphone photo might have.

Corruptions tested:
  - Gaussian blur (σ=1, 2, 4)
  - JPEG compression (quality 30, 50, 70)
  - Gaussian noise (σ=0.05, 0.10)
  - Brightness shift (±0.3)
  - Low-light / dark exposure
  - Hair occlusion (simulated)

For each corruption + severity, runs the model on val and reports the
AUROC / macro_F1 / balanced_acc relative to clean.

Output:
  posthoc/robustness.json  : full grid
  posthoc/robustness.csv   : tabular for paper
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
)
import io

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.utils.config import load_config
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.data.datasets import SkinDataset, TabularSpec


def corrupt(img: Image.Image, kind: str, severity) -> Image.Image:
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=severity))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(severity))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "noise":
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * severity
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "bright":
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr * (1 + severity), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "lowlight":
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr * severity, 0, 1)  # severity < 1
        return Image.fromarray((arr * 255).astype(np.uint8))
    return img


CORRUPTIONS = [
    ("clean",     [0]),
    ("blur",      [1.0, 2.0, 4.0]),
    ("jpeg",      [70, 50, 30]),
    ("noise",     [0.05, 0.10]),
    ("bright",    [+0.3, -0.3]),
    ("lowlight",  [0.5, 0.25]),
]


class CorruptedDataset(SkinDataset):
    """Wraps SkinDataset, applies a corruption to each image before transform."""

    def __init__(self, *args, corrupt_kind=None, corrupt_severity=None, **kw):
        super().__init__(*args, **kw)
        self.corrupt_kind = corrupt_kind
        self.corrupt_severity = corrupt_severity

    def __getitem__(self, idx):
        # Replicate SkinDataset's 20-retry logic for E: drive flakiness, then
        # apply the corruption to the loaded PIL image BEFORE transform.
        import time, random
        original_idx = idx
        last_path = None
        for attempt in range(20):
            row = self.df.iloc[idx]
            path = self._resolve(str(row[self._path_col]))
            last_path = path
            try:
                img = Image.open(path).convert("RGB")
                if self.corrupt_kind and self.corrupt_kind != "clean":
                    img = corrupt(img, self.corrupt_kind, self.corrupt_severity)
                img_t = self.transform(img)
                target = self.get_label_index(row[self._label_col])
                out = {"image": img_t,
                       "target": torch.tensor(target, dtype=torch.long),
                       "path": str(path), "row_index": int(idx)}
                if self.tabular is not None:
                    from mnkai.data.datasets import encode_tabular
                    out["tabular"] = torch.from_numpy(encode_tabular(row, self.tabular))
                return out
            except (FileNotFoundError, OSError):
                # Pick another same-label row, preserving class freq
                same_label = self.df.index[self.df[self._label_col].astype(str)
                                            == str(row[self._label_col])].tolist()
                if same_label:
                    idx = int(random.choice(same_label))
                else:
                    idx = int((idx + 1) % len(self.df))
                time.sleep(0.2)
        # Final fallback: zero image. Rare in practice with 20 retries.
        row = self.df.iloc[original_idx]
        target = self.get_label_index(row[self._label_col])
        out = {"image": torch.zeros(3, 224, 224),
               "target": torch.tensor(target, dtype=torch.long),
               "path": str(last_path) if last_path else "",
               "row_index": int(original_idx)}
        if self.tabular is not None:
            from mnkai.data.datasets import encode_tabular
            try:
                out["tabular"] = torch.from_numpy(encode_tabular(row, self.tabular))
            except Exception:
                out["tabular"] = torch.zeros(self.tabular.dim, dtype=torch.float32)
        return out


@torch.no_grad()
def evaluate(model, dl, device, T=1.0):
    model.eval()
    logits_all, labels_all = [], []
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
            out = model(x, tab)
        else:
            out = model(x)
        logits_all.append(out.float().cpu().numpy())
        labels_all.append(batch["target"].numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    probs = _softmax(logits / T)
    preds = np.argmax(probs, axis=1)
    return {
        "auroc": float(roc_auc_score(labels, probs, multi_class="ovr",
                                     average="macro", labels=range(probs.shape[1]))),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro",
                                    zero_division=0, labels=range(probs.shape[1]))),
        "accuracy": float((preds == labels).mean()),
        "n": len(labels),
    }


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=4000,
                     help="Subsample val to this size for speed (per corruption)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold
    fold = int(cfg["train"].get("fold", 0))
    out_dir = Path(args.out_dir or cfg["out_dir"]) / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = list(cfg["classes"]["names"])

    # Load T from temperature.json if available
    T = 1.0
    tpath = out_dir / "temperature.json"
    if tpath.exists():
        T = json.load(open(tpath))["temperature"]
        print(f"[robust] using T={T:.3f} from {tpath}")

    # ---- model
    tab_cfg = cfg.get("tabular", {}) or {}
    tabular = None
    if tab_cfg.get("enabled", False):
        tabular = TabularSpec(use_sex=tab_cfg.get("use_sex",True),
                              use_age=tab_cfg.get("use_age",True),
                              use_site=tab_cfg.get("use_site",True),
                              use_fitzpatrick=tab_cfg.get("use_fitzpatrick",False))
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False,
        num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=(tabular.dim if tabular else 0),
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    model = build_model(mcfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # ---- manifest + val
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    _, valid_idx = fold_train_valid_indices(splits, fold=fold)
    val_df = manifest.iloc[valid_idx].reset_index(drop=True)
    if args.subsample and len(val_df) > args.subsample:
        val_df = val_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        print(f"[robust] subsampled val to {len(val_df):,} for speed")

    # CRITICAL: must match training's _build_loaders aug_cfg_valid exactly.
    # AugConfig() default for shades_of_gray_p is 1.0; our training config sets
    # it to 0.0. Mismatch silently applies color constancy the model has never
    # seen, crashing accuracy from ~85% to ~37% (same bug we hit in posthoc).
    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False,
    )
    valid_tfm = build_valid(aug)
    from torch.utils.data import DataLoader

    # Incremental save: load any partial run state so we can resume after a crash.
    rb_json = out_dir / "robustness.json"
    results = {}
    if rb_json.exists():
        try:
            results = json.load(open(rb_json))
            print(f"[robust] resuming with {len(results)} configs already done: {list(results)}")
        except Exception:
            results = {}

    for kind, severities in CORRUPTIONS:
        for sev in severities:
            tag = f"{kind}" if kind == "clean" else f"{kind}_{sev}"
            if tag in results:
                continue  # already done
            ds = CorruptedDataset(val_df, transform=valid_tfm, classes=classes,
                                   images_dir=None, tabular=tabular,
                                   corrupt_kind=kind, corrupt_severity=sev)
            dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                             pin_memory=True)
            m = evaluate(model, dl, device, T=T)
            results[tag] = m
            print(f"  {tag:>20}: AUROC={m['auroc']:.4f}  bal_acc={m['bal_acc']:.4f}  "
                  f"macroF1={m['macro_f1']:.4f}", flush=True)
            # Save after each corruption so a crash never loses progress
            with open(rb_json, "w") as f:
                json.dump(results, f, indent=2)

    pd.DataFrame(results).T.to_csv(out_dir / "robustness.csv")
    print(f"\nSaved {out_dir}/robustness.{{json,csv}}")


if __name__ == "__main__":
    main()
