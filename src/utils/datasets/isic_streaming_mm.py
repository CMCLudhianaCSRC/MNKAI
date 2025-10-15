import csv, io, time, numpy as np
from pathlib import Path
from PIL import Image
import requests
import torch
from torch.utils.data import Dataset

ISIC_API = "https://api.isic-archive.com/api/v2"
SEX_MAP = {"male":0, "female":1, "":2, "unknown":2}

class ISICStreamingMM(Dataset):
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
        labs = sorted({(x.get('label') or 'unknown') for x in self.rows}); self.label_to_idx = {l:i for i,l in enumerate(labs)}
        sites = sorted({((x.get('anatom_site_general') or '').strip().lower()) for x in self.rows}); self.site_to_idx = {s:i for i,s in enumerate(sites)}
        self.tab_dim = 1 + 3 + len(self.site_to_idx)

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

    def _encode_tab(self, row):
        try: age = float(row.get('age_approx') or 0)/100.0
        except: age = 0.0
        sex_vec = np.zeros(3, dtype=np.float32); sex_vec[SEX_MAP.get((row.get('sex') or '').lower(),2)] = 1.0
        site = (row.get('anatom_site_general') or '').strip().lower(); site_vec = np.zeros(len(self.site_to_idx), dtype=np.float32)
        site_vec[self.site_to_idx.get(site, 0)] = 1.0
        tab = np.concatenate([[age], sex_vec, site_vec]).astype(np.float32)
        return torch.from_numpy(tab)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]; isic_id = row['isic_id']
        y = self.label_to_idx.get((row.get('label') or 'unknown'), 0)
        img = self._load_img(isic_id, self._cache_path(isic_id))
        if self.transforms: img = self.transforms(img)
        tab = self._encode_tab(row); src = row.get('source','ISIC')
        return img, tab, y, isic_id, src
