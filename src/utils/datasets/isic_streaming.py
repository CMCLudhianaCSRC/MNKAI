import csv, io, time
from pathlib import Path
from PIL import Image
import requests
import torch
from torch.utils.data import Dataset

ISIC_API = "https://api.isic-archive.com/api/v2"

class ISICStreaming(Dataset):
    def __init__(self, manifest_csv: str, cache_dir: str, transforms=None, filter_modality: str = 'dermoscopic'):
        self.rows = []
        with open(manifest_csv, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                if (filter_modality is None) or (row.get('modality','')==filter_modality):
                    self.rows.append(row)
        self.transforms = transforms
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        labs = sorted({(x.get('label') or 'unknown') for x in self.rows})
        self.label_to_idx = {l:i for i,l in enumerate(labs)}

    def _cache_path(self, isic_id: str) -> Path:
        sub = isic_id.split('_')[1][:3] if '_' in isic_id else isic_id[:3]
        d = self.cache_dir / sub; d.mkdir(parents=True, exist_ok=True)
        return d / f"{isic_id}.jpg"

    def _download(self, isic_id: str) -> bytes:
        url = f"{ISIC_API}/image/{isic_id}/download"
        for i in range(5):
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code==200: return r.content
                if r.status_code in (429,500,502,503,504): time.sleep(0.5*(2**i)); continue
                r.raise_for_status()
            except Exception:
                time.sleep(0.5*(2**i))
        raise RuntimeError(f"Failed to fetch {isic_id}")

    def _load_img(self, isic_id: str, path: Path):
        if not path.exists():
            b = self._download(isic_id)
            Image.open(io.BytesIO(b)).verify()
            tmp = path.with_suffix('.jpg.part'); tmp.write_bytes(b); tmp.replace(path)
        return Image.open(path).convert('RGB')

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]; isic_id = row['isic_id']
        y = self.label_to_idx.get((row.get('label') or 'unknown'), 0)
        img = self._load_img(isic_id, self._cache_path(isic_id))
        if self.transforms: img = self.transforms(img)
        return img, y, isic_id, row.get('source','ISIC')
