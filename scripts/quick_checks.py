import os
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils import set_seed as project_set_seed
from src.transforms import AugConfig, build_train, build_valid
from src.dataset import SkinDataset
from src.model import build_model
import yaml


CFG_PATH = "configs/mnkai_isic.yaml"


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower(): c for c in df.columns}
    path_col = colmap.get("filepath") or colmap.get("image") or colmap.get("image_path") or colmap.get("filename")
    label_col = colmap.get("label") or colmap.get("label_name") or colmap.get("diagnosis") or colmap.get("target")
    if path_col is None or label_col is None:
        raise RuntimeError("Manifest must contain filepath + label columns.")
    out = df.rename(columns={path_col: "filepath", label_col: "label"}).copy()
    return out


def filter_existing(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["filepath"].map(lambda p: Path(p).is_file())
    missing = (~mask).sum()
    if missing:
        print(f"[CHECK] dropping {missing} rows with missing files.")
    return df[mask].reset_index(drop=True)


def split_train_valid(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedShuffleSplit

    seed = int(cfg.get("seed", 42))
    frac = float(cfg["data"].get("val_fraction", 0.2))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    y = df["label"].astype(str).values
    idxs = np.arange(len(df))
    tr_idx, va_idx = next(splitter.split(idxs, y))
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    valid_df = df.iloc[va_idx].reset_index(drop=True)
    print(f"[CHECK] split -> train={len(train_df)} valid={len(valid_df)}")
    return train_df, valid_df


def dataset_class_order(cfg: Dict, sample_df: pd.DataFrame):
    classes = [str(x) for x in cfg["classes"]["names"]]
    aug_cfg = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"]["normalize_mean"]),
        std=tuple(cfg["data"]["normalize_std"]),
    )
    ds = SkinDataset(
        sample_df,
        classes,
        img_size=int(cfg["data"]["img_size"]),
        transform=build_valid(aug_cfg),
        hair_removal=False,
        images_dir=cfg["data"].get("images_dir", ""),
    )
    ordering = sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])
    ordered_labels = [label for label, _ in ordering]
    print("\n[CHECK] YAML class order:", classes)
    print("[CHECK] dataset.class_to_idx:", ds.class_to_idx)
    if ordered_labels != classes:
        raise AssertionError("Label index order mismatch between dataset and YAML.")
    print("[CHECK] dataset class order matches YAML.")


def build_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: Dict,
    classes: List[str],
    train_batch: int = 16,
    valid_batch: int = 64,
):
    img_size = int(cfg["data"]["img_size"])
    aug_conf = AugConfig(
        img_size=img_size,
        mean=tuple(cfg["data"]["normalize_mean"]),
        std=tuple(cfg["data"]["normalize_std"]),
    )
    ds_train = SkinDataset(
        train_df,
        classes,
        img_size,
        build_train(aug_conf),
        hair_removal=False,
        images_dir=cfg["data"].get("images_dir", ""),
    )
    ds_valid = SkinDataset(
        valid_df,
        classes,
        img_size,
        build_valid(aug_conf),
        hair_removal=False,
        images_dir=cfg["data"].get("images_dir", ""),
    )

    train_counts = train_df["label"].value_counts().reindex(classes).fillna(0).values
    weights = 1.0 / np.maximum(train_counts, 1)
    sampler_weights = train_df["label"].map(dict(zip(classes, weights))).values
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(sampler_weights), replacement=True)

    nworkers = int(cfg["hardware"].get("num_workers", 4))
    pin = bool(cfg["hardware"].get("pin_memory", True))
    prefetch = cfg["hardware"].get("prefetch_factor", 2) if nworkers > 0 else None

    dl_train = DataLoader(
        ds_train,
        batch_size=train_batch,
        sampler=sampler,
        num_workers=nworkers,
        pin_memory=pin,
        prefetch_factor=prefetch,
        persistent_workers=(nworkers > 0),
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=valid_batch,
        shuffle=False,
        num_workers=nworkers,
        pin_memory=pin,
        prefetch_factor=prefetch,
        persistent_workers=(nworkers > 0),
    )
    return dl_train, dl_valid


@torch.no_grad()
def eval_once(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: List[str],
    report_hist: bool = True,
):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)
        logits = model(images)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())
    if not preds:
        return 0.0, {}
    preds_arr = np.concatenate(preds)
    targets_arr = np.concatenate(targets)
    acc = float((preds_arr == targets_arr).mean())
    hist = dict(sorted(Counter(preds_arr.tolist()).items(), key=lambda kv: kv[0]))
    if report_hist:
        print("[CHECK] prediction histogram:", hist)
    return acc, hist


def quick_overfit(train_df: pd.DataFrame, cfg: Dict, classes: List[str]) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = train_df.sample(n=min(256, len(train_df)), random_state=123).reset_index(drop=True)
    dl_train, dl_valid = build_loaders(subset, subset, cfg, classes, train_batch=8, valid_batch=16)
    model = make_model(cfg, classes, pretrained=False, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    print(f"[CHECK] tiny-overfit on {len(subset)} samples")
    for epoch in range(1, 13):
        model.train()
        for batch in dl_train:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        acc, _ = eval_once(model, dl_valid, device, classes, report_hist=False)
        print(f"[CHECK] tiny-overfit epoch {epoch}: acc={acc:.3f}")
        if acc >= 0.88:
            print("[CHECK] tiny-overfit passed")
            return True
    print("[CHECK] tiny-overfit did not cross 0.88 accuracy - inspect labels/order if issues persist.")
    return False


def make_model(cfg: Dict, classes: List[str], pretrained: bool, device: torch.device):
    model = build_model(
        cfg["train"]["model_name"],
        num_classes=len(classes),
        pretrained=pretrained,
        compile_model=False,
    )
    return model.to(device)


def main():
    cfg = load_yaml(CFG_PATH)
    project_set_seed(int(cfg.get("seed", 42)))
    manifest = cfg["data"].get("manifest_csv") or cfg["data"].get("metadata_csv")
    if not manifest:
        raise RuntimeError("config.data.manifest_csv (or metadata_csv) is required.")
    df = ensure_cols(pd.read_csv(manifest))
    df = filter_existing(df)
    classes = [str(x) for x in cfg["classes"]["names"]]

    train_df, valid_df = split_train_valid(df, cfg)
    dataset_class_order(cfg, train_df)

    _, valid_loader = build_loaders(train_df, valid_df, cfg, classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg, classes, pretrained=False, device=device)
    _, pred_hist = eval_once(model, valid_loader, device, classes, report_hist=True)
    if pred_hist:
        total = sum(pred_hist.values())
        dominant_cls, dominant_count = max(pred_hist.items(), key=lambda kv: kv[1])
        if dominant_count / total > 0.7:
            pct = dominant_count / total
            print(
                f"[CHECK] warning: {pct:.1%} of predictions map to class index {dominant_cls} on "
                "an untrained model. Double-check class order / label mapping."
            )

    quick_overfit(train_df, cfg, classes)


if __name__ == "__main__":
    main()
