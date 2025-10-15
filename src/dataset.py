# src/dataset.py
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SkinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        classes: List[str],
        img_size: int,
        transform=None,
        hair_removal: bool = False,
        images_dir: Optional[str] = ""
    ):
        self.df = df.reset_index(drop=True)
        self.classes = [str(c).lower() for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.img_size = img_size
        self.transform = transform
        self.hair_removal = bool(hair_removal)
        self.images_dir = images_dir or ""
        cols = {c.lower(): c for c in self.df.columns}
        self.path_col = cols.get("filepath") or cols.get("image") or cols.get("image_path") or cols.get("filename")
        self.label_col = cols.get("label") or cols.get("label_name") or cols.get("diagnosis") or cols.get("class")
        if self.path_col is None or self.label_col is None:
            raise ValueError("Dataset CSV must have filepath and label columns")

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        P = Path(p)
        if P.is_absolute():
            return P
        if self.images_dir:
            return (Path(self.images_dir) / P).resolve()
        return P.resolve()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = self._resolve_path(str(row[self.path_col]))
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        lab = str(row[self.label_col]).strip().lower()
        target = self.class_to_idx.get(lab, -1)
        if target < 0:
            raise RuntimeError(f"Unknown label '{lab}' at row {idx}")
        return {"image": img, "target": torch.tensor(target, dtype=torch.long), "path": str(path)}
