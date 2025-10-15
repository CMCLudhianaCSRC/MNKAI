import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from torch.utils.data import DataLoader
import yaml

from src.model import build_model
from src.dataset import SkinDataset
from src.transforms import AugConfig, build_valid
from src.metrics import metrics_from_preds


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("filepath") or cols.get("image") or cols.get("image_path") or cols.get("filename")
    label_col = cols.get("label_name") or cols.get("label") or cols.get("diagnosis") or cols.get("target")
    return path_col, label_col


def build_label_map(cfg: Dict) -> Tuple[List[str], Dict[str, str], set]:
    classes = [str(x).strip().lower() for x in cfg["classes"]["names"]]
    alias = cfg["classes"].get("aliases", {})
    table = {c: c for c in classes}
    for key, values in alias.items():
        key = str(key).strip().lower()
        for value in values:
            table[str(value).strip().lower()] = key
    ignore = {str(s).lower() for s in cfg["classes"].get("ignore", [])}
    return classes, table, ignore


def normalize_label(raw: str, table: Dict[str, str], ignore: set) -> str:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in ignore:
        return None
    if token in table:
        return table[token]
    token2 = " ".join(token.replace("-", " ").replace("_", " ").split())
    return table.get(token2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ISIC model with optional TTA.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--no-tta", action="store_true", help="Disable simple flip TTA.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to dump metrics JSON.")
    return parser.parse_args()


def prepare_validation(cfg, classes):
    data_cfg = cfg.get("data", {})
    seed = int(cfg.get("seed", 42))
    manifest_csv = data_cfg.get("manifest_csv")
    if not manifest_csv:
        raise RuntimeError("config.data.manifest_csv is required for evaluation.")

    raw = pd.read_csv(manifest_csv)
    pcol, lcol = detect_cols(raw)
    if pcol is None or lcol is None:
        raise RuntimeError("Could not detect filepath/label columns in manifest.")

    strat_col = data_cfg.get("stratify_col", "label")
    group_col = data_cfg.get("group_column", "")

    classes_clean, alias_table, ignore = build_label_map(cfg)

    df = pd.DataFrame(
        {
            "filepath": raw[pcol].astype(str),
            "raw_label": raw[lcol],
        }
    )
    if strat_col and strat_col in raw.columns:
        df[strat_col] = raw[strat_col]
    if group_col and group_col in raw.columns:
        df[group_col] = raw[group_col]

    df["label"] = df["raw_label"].apply(lambda s: normalize_label(s, alias_table, ignore))
    df = df[df["label"].isin(classes_clean)].copy()
    df = df[df["filepath"].map(lambda p: Path(p).is_file())].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise RuntimeError("No valid images found after filtering manifest.")

    strat_values = df["label"]
    if strat_col and strat_col in df.columns:
        strat_values = df[strat_col].fillna(df["label"])

    groups = None
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("").astype(str).values

    val_fraction = float(data_cfg.get("val_fraction", 0.2))
    if groups is not None and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        _, val_idx = next(splitter.split(df.index.values, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        _, val_idx = next(splitter.split(df.index.values, strat_values))

    valid_df = df.iloc[val_idx].reset_index(drop=True)
    return valid_df


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_classes = checkpoint.get("classes")
    classes = [str(c).strip().lower() for c in (ckpt_classes or cfg["classes"]["names"])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    valid_df = prepare_validation(cfg, classes)

    data_cfg = cfg.get("data", {})
    img_size = int(data_cfg.get("img_size", 336))
    aug_cfg = AugConfig(
        img_size=img_size,
        mean=tuple(data_cfg.get("normalize_mean", (0.485, 0.456, 0.406))),
        std=tuple(data_cfg.get("normalize_std", (0.229, 0.224, 0.225))),
    )
    center_crop = bool(cfg.get("augmentation", {}).get("valid", {}).get("center_crop", True))
    valid_transform = build_valid(aug_cfg, center_crop=center_crop)

    dataset = SkinDataset(
        valid_df,
        classes,
        img_size=img_size,
        transform=valid_transform,
        hair_removal=False,
        images_dir=data_cfg.get("images_dir", ""),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=True if device.type == "cuda" else False,
    )

    model_name = cfg["train"]["model_name"]
    model = build_model(
        model_name,
        num_classes=len(classes),
        pretrained=False,
        compile_model=False,
    )
    missing, unexpected = model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}, Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()

    use_tta = not args.no_tta
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device)
            logits = model(inputs)
            if use_tta:
                logits = 0.5 * (logits + model(torch.flip(inputs, dims=[3])))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(targets.cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    preds = probs_all.argmax(axis=1)

    report = metrics_from_preds(labels_all, preds, probs_all, classes)
    metrics_json = json.dumps(report["report"], indent=2)
    print(metrics_json)
    if "macro_auc_ovr" in report:
        print(f"macro_auc_ovr: {report['macro_auc_ovr']}")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(metrics_json, encoding="utf-8")


if __name__ == "__main__":
    main()
