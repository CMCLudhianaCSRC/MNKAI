"""
Utility script to export per-image class probabilities from a trained checkpoint.

Example:
    python scripts/text.py \
        --checkpoint checkpoints/run_final/best.pt \
        --manifest data/manifest_unified_labeled.csv \
        --out-prefix outputs/val
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import timm
import yaml


DEFAULT_CLASSES = ["mel", "nv", "bcc", "scc", "df", "bkl", "ak", "vasc"]


# -------------------------
# Helpers
# -------------------------
def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    path_keys = ("filepath", "path", "image", "image_path", "file_path", "filename")
    label_keys = ("label", "label_name", "target", "diagnosis", "class")

    path_col = next((cols[k] for k in path_keys if k in cols), None)
    label_col = next((cols[k] for k in label_keys if k in cols), None)
    return path_col, label_col


def resolve_path(raw: str, images_dir: Optional[Path]) -> Path:
    if not raw:
        return Path(raw)
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    if images_dir:
        candidate = images_dir / raw
        if candidate.exists():
            return candidate
    return p


def build_transform(img_size: int, mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        classes: List[str],
        images_dir: Optional[Path],
        tfm: T.Compose,
    ):
        self.df = frame.reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.images_dir = images_dir
        self.transform = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = resolve_path(row["filepath"], self.images_dir)
        with Image.open(path) as im:  # type: ignore[name-defined]
            image = im.convert("RGB")
        x = self.transform(image)
        target = self.class_to_idx[row["label"]]
        return {"image": x, "target": target, "path": str(path)}


# Delay PIL import until after dataset definition to keep typing happy
from PIL import Image  # noqa: E402  pylint: disable=wrong-import-position


# -------------------------
# Core export logic
# -------------------------
@torch.no_grad()
def export_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: List[str],
    tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_paths: List[str] = []

    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = torch.as_tensor(batch["target"]).to(device)

        logits = model(images)
        if tta:
            flips = [
                torch.flip(images, dims=[3]),  # horizontal
                torch.flip(images, dims=[2]),  # vertical
                torch.flip(images, dims=[2, 3]),
            ]
            for aug in flips:
                logits = logits + model(aug)
            logits = logits / (len(flips) + 1)

        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())
        all_paths.extend(batch["path"])

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return y_prob, y_true, all_paths


def create_model(
    checkpoint: Path,
    model_name: Optional[str],
    classes: List[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    if model_name is None:
        model_name = (
            cfg.get("train", {}).get("model_name")
            or ckpt.get("model_name")
        )
    if not model_name:
        raise ValueError("Model name not provided. Pass --model-name or include it in the checkpoint.")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    return model, cfg


def load_classes(
    checkpoint: Path,
    cfg: Optional[Dict],
    explicit_classes: Optional[List[str]],
) -> List[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if ckpt.get("classes"):
        return [str(c).strip().lower() for c in ckpt["classes"]]
    if explicit_classes:
        return [str(c).strip().lower() for c in explicit_classes]
    if cfg and cfg.get("classes", {}).get("names"):
        return [str(c).strip().lower() for c in cfg["classes"]["names"]]
    return DEFAULT_CLASSES


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export probabilities from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", help="CSV manifest; defaults to config.data.manifest_csv if not provided")
    parser.add_argument("--config", help="Model config YAML (optional; falls back to checkpoint config)")
    parser.add_argument("--model-name", help="Model architecture name (timm)")
    parser.add_argument(
        "--out-prefix",
        default="outputs/val",
        help="Path prefix for npy outputs (e.g. outputs/val -> val_probs.npy)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--tta", action="store_true", help="Enable simple flip TTA (4-way)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--classes", nargs="+", help="Override class list")
    parser.add_argument("--images-dir", help="Optional root directory to prepend to relative paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    cfg_from_file = load_yaml(args.config) if args.config else None
    ckpt_path = Path(args.checkpoint)
    classes = load_classes(ckpt_path, cfg_from_file, args.classes)

    model, cfg_in_ckpt = create_model(ckpt_path, args.model_name, classes, device)

    config = cfg_from_file or cfg_in_ckpt or {}
    mean = config.get("data", {}).get("normalize_mean", [0.485, 0.456, 0.406])
    std = config.get("data", {}).get("normalize_std", [0.229, 0.224, 0.225])

    manifest_path = args.manifest or config.get("data", {}).get("manifest_csv")
    if manifest_path is None:
        raise ValueError("Manifest CSV not provided. Use --manifest or specify data.manifest_csv in the config.")
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    path_col, label_col = detect_columns(df)
    if path_col is None or label_col is None:
        raise ValueError("Manifest must contain columns for image paths and labels.")
    df = df.rename(columns={path_col: "filepath", label_col: "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(classes)]
    if df.empty:
        raise ValueError("No rows remaining after filtering by class list.")

    images_dir = Path(args.images_dir).resolve() if args.images_dir else None
    tfm = build_transform(args.img_size, mean, std)
    dataset = ManifestDataset(df, classes, images_dir, tfm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    probs, labels, paths = export_probs(model, loader, device, classes, tta=args.tta)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{prefix}_probs.npy", probs)
    np.save(f"{prefix}_labels.npy", labels)
    with open(f"{prefix}_paths.json", "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)

    print(f"Saved probabilities to {prefix}_probs.npy (shape {probs.shape})")
    print(f"Saved labels to {prefix}_labels.npy (shape {labels.shape})")
    print(f"Saved paths to {prefix}_paths.json")


if __name__ == "__main__":
    main()
