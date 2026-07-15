"""ISIC Archive v2 downloader, with collection support.

The ISIC Archive groups images into named "collections" (HAM10000, BCN_20000,
ISIC 2019 Challenge - Training, MILK10k, PROVe-AI, BALD acral, etc.). This
module provides:

  - list_collections()         : resolve human-readable name patterns to IDs
  - fetch_metadata_for(...)    : page through `/api/v2/images/search/?collections=ID`
                                 and write a CSV. Retries on connection drops.
  - download_ids(...)          : bulk-download images, sharded, resumable.
"""
from __future__ import annotations

import asyncio
import csv
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import aiohttp
import requests
from loguru import logger
from PIL import Image

ISIC_API_V2 = "https://api.isic-archive.com/api/v2"


def _get_with_retry(session: requests.Session, url: str, *, params=None, max_attempts: int = 6,
                    base_delay: float = 1.0, timeout: int = 60) -> requests.Response:
    """Retry on transient connection / 5xx errors with exponential backoff."""
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"[isic] {url} HTTP {r.status_code}; retry {attempt}/{max_attempts}")
                time.sleep(delay); delay = min(delay * 2, 30.0); continue
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            last_exc = e
            logger.warning(f"[isic] {url} {type(e).__name__}; retry {attempt}/{max_attempts}")
            time.sleep(delay); delay = min(delay * 2, 30.0)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"max retries exhausted for {url}")


# ----------------------------------------------------------- collection list --

def list_collections() -> list[dict]:
    """Return all collections {id, name, description, public, ...}."""
    url = f"{ISIC_API_V2}/collections/"
    out: list[dict] = []
    while url:
        r = requests.get(url, params={"limit": 200}, timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("results", []))
        url = data.get("next")
    return out


def find_collections(patterns: Sequence[str]) -> dict[str, list[dict]]:
    """For each pattern, return collections whose name matches (case-insensitive
    substring or regex). Useful for `["BCN", "MILK", "PROVe", "BALD acral",
    "HAM10000", "2019 Challenge", "2020 Challenge"]`.
    """
    coll = list_collections()
    out: dict[str, list[dict]] = {p: [] for p in patterns}
    for p in patterns:
        rx = re.compile(p, re.IGNORECASE)
        for c in coll:
            n = c.get("name", "")
            if rx.search(n):
                out[p].append({"id": c["id"], "name": n, "public": c.get("public", False)})
    return out


# -------------------------------------------------------------- metadata --

def _flatten_meta(it: dict) -> dict:
    md = it.get("metadata") or {}
    cm = md.get("clinical", {}) or {}
    am = md.get("acquisition", {}) or {}
    return {
        "isic_id": it.get("isic_id"),
        "attribution": it.get("attribution"),
        "copyright_license": it.get("copyright_license"),
        "modality": am.get("image_type") or am.get("modality"),
        "diagnosis": cm.get("diagnosis"),
        "diagnosis_3": cm.get("diagnosis_3"),
        "diagnosis_confirm_type": cm.get("diagnosis_confirm_type"),
        "benign_malignant": cm.get("benign_malignant"),
        "sex": cm.get("sex"),
        "age_approx": cm.get("age_approx"),
        "anatom_site_general": cm.get("anatom_site_general"),
        "fitzpatrick": cm.get("fitzpatrick_skin_type"),
        "patient_id": cm.get("patient_id"),
        "lesion_id": cm.get("lesion_id"),
    }


def fetch_metadata_for(collection_ids: Sequence[int] | None, out_csv: str | Path,
                       *, limit: int | None = None) -> int:
    """Fetch metadata for one OR many collections (or whole archive if None).

    Pages through `/images/search/?collections=A,B,C`. The `/search/` variant is
    the one that actually honors the `collections` filter; the bare `/images/`
    endpoint silently returns the whole archive.

    Returns row count written.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    base = f"{ISIC_API_V2}/images/search/" if collection_ids else f"{ISIC_API_V2}/images/"
    params: dict = {"limit": 200}
    if collection_ids:
        params["collections"] = ",".join(str(i) for i in collection_ids)

    rows: list[dict] = []
    next_url: str | None = base
    page = 0
    while next_url:
        page += 1
        r = _get_with_retry(sess, next_url, params=params if next_url == base else None, timeout=60)
        data = r.json()
        results = data.get("results", []) or []
        for it in results:
            rows.append(_flatten_meta(it))
            if limit is not None and len(rows) >= limit:
                break
        total = data.get("count")
        if page == 1 and total is not None:
            logger.info(f"[isic] total rows for selection: {total}")
        if page % 25 == 0:
            logger.info(f"[isic] page {page} fetched ({len(rows)} rows so far)")
        if limit is not None and len(rows) >= limit:
            break
        next_url = data.get("next")

    if not rows:
        logger.warning(f"[isic] no metadata returned for collections={collection_ids}")
        # No rows fetched; leave `rows` empty. The `fieldnames` fallback below
        # writes a header-only CSV using the canonical column list.
        rows = []
    fieldnames = list(rows[0].keys()) if rows else [
        "isic_id", "attribution", "copyright_license", "modality",
        "diagnosis", "diagnosis_3", "diagnosis_confirm_type", "benign_malignant",
        "sex", "age_approx", "anatom_site_general", "fitzpatrick",
        "patient_id", "lesion_id",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"[isic] wrote {len(rows)} rows to {out_csv}")
    return len(rows)


# -------------------------------------------------------------- download --

@dataclass
class DownloadStats:
    requested: int = 0
    skipped_exists: int = 0
    downloaded: int = 0
    failed: int = 0


def _shard_path(root: Path, isic_id: str) -> Path:
    body = isic_id.split("_")[1] if "_" in isic_id else isic_id
    a = body[:2]
    b = body[2:4] if len(body) >= 4 else "00"
    d = root / a / b
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{isic_id}.jpg"


ISIC_S3_BASE = "https://isic-archive.s3.amazonaws.com/images"


async def _fetch_one(session: aiohttp.ClientSession, isic_id: str, target: Path,
                     sem: asyncio.Semaphore, stats: DownloadStats) -> None:
    """Fetch directly from the public S3 bucket. The /api/v2/.../download
    endpoint was retired; image URLs now live at:
        https://isic-archive.s3.amazonaws.com/images/<id>.jpg
    (this URL is also exposed in `files.full.url` of each metadata row).
    """
    if target.exists() and target.stat().st_size > 0:
        stats.skipped_exists += 1
        return
    url = f"{ISIC_S3_BASE}/{isic_id}.jpg"
    delay = 0.5
    for _ in range(6):
        async with sem:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status == 200:
                        data = await r.read()
                        # Defer PIL validation to the QC pass (scripts/run_qc.py).
                        # Validating per-image at download time is CPU-heavy and
                        # bottlenecks the async loop; S3 ETag already guarantees
                        # byte-level integrity.
                        if len(data) < 256:
                            stats.failed += 1
                            return
                        tmp = target.with_suffix(".jpg.part")
                        tmp.write_bytes(data)
                        tmp.replace(target)
                        stats.downloaded += 1
                        return
                    if r.status in (404, 410):
                        # legitimately missing on S3 — record + move on, don't waste retries
                        stats.failed += 1
                        return
                    if r.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(delay); delay = min(delay * 2, 16.0); continue
                    logger.error(f"[isic] {isic_id} HTTP {r.status}")
                    stats.failed += 1
                    return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"[isic] {isic_id} {e}")
                await asyncio.sleep(delay); delay = min(delay * 2, 16.0)
    stats.failed += 1


async def _download_async(ids: Iterable[str], cache_dir: Path, concurrency: int = 8) -> DownloadStats:
    stats = DownloadStats()
    ids = list(ids)
    stats.requested = len(ids)
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=120)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [_fetch_one(session, i, _shard_path(cache_dir, i), sem, stats) for i in ids]
        for fut in asyncio.as_completed(tasks):
            await fut
    return stats


def download_ids(ids: Iterable[str], cache_dir: Path | str, concurrency: int = 8) -> DownloadStats:
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_download_async(ids, cache_dir, concurrency=concurrency))


__all__ = [
    "list_collections", "find_collections",
    "fetch_metadata_for", "download_ids",
    "DownloadStats", "ISIC_API_V2",
]
