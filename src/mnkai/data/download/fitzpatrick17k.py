"""Fitzpatrick17k downloader.

NOTE: the CSV at github.com/mattgroh/fitzpatrick17k points to dermaamin.com
and atlasdermatologico URLs that have ROTTED — they all return 404 as of 2025.
We use the Hugging Face mirror `ZYXue/Fitzpatrick_17k` which bundles all images
into Parquet files (~88 MB total).

`download_dataset_hf()`  -> recommended, fast, works.
`download_dataset()`     -> legacy URL-based, kept for reference but broken.
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from loguru import logger
from PIL import Image

CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"


def fetch_metadata(out_csv: str | Path) -> pd.DataFrame:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(CSV_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(out_csv, index=False)
    logger.info(f"[fitz17k] wrote {len(df)} rows to {out_csv}")
    return df


def _shard_path(root: Path, key: str) -> Path:
    a, b = key[:2], key[2:4]
    d = root / a / b
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.jpg"


async def _fetch_one(session: aiohttp.ClientSession, url: str, target: Path,
                     sem: asyncio.Semaphore, stats: dict) -> None:
    if target.exists() and target.stat().st_size > 0:
        stats["skipped"] += 1
        return
    delay = 0.5
    for _ in range(5):
        async with sem:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status == 200:
                        data = await r.read()
                        try:
                            Image.open(io.BytesIO(data)).verify()
                        except Exception:
                            stats["failed"] += 1
                            return
                        tmp = target.with_suffix(".jpg.part")
                        tmp.write_bytes(data)
                        tmp.replace(target)
                        stats["downloaded"] += 1
                        return
                    if r.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(delay); delay = min(delay * 2, 16.0); continue
                    stats["failed"] += 1
                    return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(delay); delay = min(delay * 2, 16.0)
    stats["failed"] += 1


async def _download_async(rows: pd.DataFrame, cache_dir: Path, concurrency: int = 8,
                          url_col: str = "url", key_col: str = "md5hash") -> dict:
    stats = {"requested": len(rows), "skipped": 0, "downloaded": 0, "failed": 0}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=120)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for _, row in rows.iterrows():
            url = str(row.get(url_col, "") or "")
            key = str(row.get(key_col, "") or "")
            if not url or not key:
                stats["failed"] += 1
                continue
            tasks.append(_fetch_one(session, url, _shard_path(cache_dir, key), sem, stats))
        for fut in asyncio.as_completed(tasks):
            await fut
    return stats


def download_dataset(meta_csv: str | Path, cache_dir: str | Path,
                     concurrency: int = 6) -> dict:
    """Download all images referenced by the Fitzpatrick17k CSV."""
    df = pd.read_csv(meta_csv)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_download_async(df, cache_dir, concurrency=concurrency))


def download_dataset_hf(image_root: str | Path, *, meta_out: str | Path,
                        repo: str = "ZYXue/Fitzpatrick_17k") -> dict:
    """Download from Hugging Face mirror (Parquet) and unpack to sharded JPEGs.

    Each Parquet row is expected to have at least: `image` (dict with bytes
    or PIL-style structure), and FST + label columns. The exact column names
    vary per uploader; we autodetect.
    """
    import io as _io
    import pandas as pd
    image_root = Path(image_root); image_root.mkdir(parents=True, exist_ok=True)
    meta_out = Path(meta_out); meta_out.parent.mkdir(parents=True, exist_ok=True)

    base = f"https://huggingface.co/datasets/{repo}/resolve/main/data"
    parquet_files = ["train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"]

    rows_meta: list[dict] = []
    n_written = n_skipped = n_failed = 0
    for pq in parquet_files:
        url = f"{base}/{pq}"
        logger.info(f"[fitz17k-hf] downloading {pq}")
        local = image_root / f".{pq}"
        if not local.exists() or local.stat().st_size < 1000:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for buf in r.iter_content(1 << 20):
                        if buf: f.write(buf)
        df = pd.read_parquet(local)
        logger.info(f"[fitz17k-hf] {pq}: {len(df)} rows, columns={list(df.columns)}")
        # autodetect image column
        img_col = next((c for c in df.columns
                        if c.lower() in ("image", "img", "picture")), None)
        if img_col is None:
            logger.warning(f"[fitz17k-hf] no image column in {pq}; cols={list(df.columns)}")
            continue
        # autodetect label/FST columns
        cols_lc = {c.lower(): c for c in df.columns}
        fst_col = cols_lc.get("fitzpatrick_scale") or cols_lc.get("fitzpatrick") or cols_lc.get("fst")
        label_col = cols_lc.get("label") or cols_lc.get("nine_partition_label") or cols_lc.get("three_partition_label")
        hash_col = cols_lc.get("md5hash") or cols_lc.get("hash") or cols_lc.get("filename")
        for i, row in df.iterrows():
            img_obj = row[img_col]
            data = None
            if isinstance(img_obj, dict):
                data = img_obj.get("bytes")
            elif isinstance(img_obj, bytes):
                data = img_obj
            if not data:
                n_failed += 1
                continue
            key = str(row[hash_col]) if hash_col else f"img_{i:06d}"
            if not key:
                key = f"img_{i:06d}"
            shard_a = key[:2] if len(key) >= 2 else "00"
            shard_b = key[2:4] if len(key) >= 4 else "00"
            d = image_root / shard_a / shard_b
            d.mkdir(parents=True, exist_ok=True)
            target = d / f"{key}.jpg"
            if target.exists() and target.stat().st_size > 0:
                n_skipped += 1
            else:
                try:
                    Image.open(_io.BytesIO(data)).convert("RGB").save(target, "JPEG", quality=95)
                    n_written += 1
                except Exception as e:
                    logger.warning(f"[fitz17k-hf] decode err for {key}: {e}")
                    n_failed += 1
                    continue
            md = {"md5hash": key, "filepath": str(target)}
            if fst_col: md["fitzpatrick_scale"] = row[fst_col]
            if label_col: md["label_name"] = row[label_col]
            for extra in ("nine_partition_label", "three_partition_label", "qc"):
                if extra in df.columns: md[extra] = row[extra]
            rows_meta.append(md)
        try:
            local.unlink()
        except OSError:
            pass

    out = pd.DataFrame(rows_meta)
    out.to_csv(meta_out, index=False)
    logger.info(f"[fitz17k-hf] wrote {len(out)} rows to {meta_out}; "
                f"images: written={n_written} skipped={n_skipped} failed={n_failed}")
    return {"rows": len(out), "written": n_written, "skipped": n_skipped, "failed": n_failed}


__all__ = ["fetch_metadata", "download_dataset", "download_dataset_hf", "CSV_URL"]
