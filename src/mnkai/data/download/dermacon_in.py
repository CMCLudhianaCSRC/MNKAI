"""DermaCon-IN downloader (Harvard Dataverse).

DOI: 10.7910/DVN/W7OUZM. Public. The Dataverse API at
`/api/datasets/:persistentId/?persistentId=doi:...` returns metadata; each file
is fetched via `/api/access/datafile/<id>`.

`download_file` supports HTTP Range resume so a broken transfer continues
from where the .part file left off, with retries on `ChunkedEncodingError` and
similar transient breaks.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

DATAVERSE_BASE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/W7OUZM"


def get_dataset_files(doi: str = DOI) -> list[dict]:
    url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={doi}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("data", {}).get("latestVersion", {}).get("files", [])
    out = []
    for f in files:
        df = f.get("dataFile", {})
        out.append({
            "id": df.get("id"),
            "filename": df.get("filename"),
            "filesize": df.get("filesize"),
            "contentType": df.get("contentType"),
            "md5": df.get("md5"),
            "directoryLabel": f.get("directoryLabel", ""),
        })
    return out


def download_file(file_id: int, target: Path, *, chunk: int = 1 << 20,
                  max_attempts: int = 24, base_delay: float = 2.0) -> None:
    """Download with HTTP Range resume + transient-error retry.

    Robust to:
      - mid-stream ChunkedEncodingError / ConnectionError (resume from .part)
      - server ignoring our Range header (restart from byte 0)
      - the .part file disappearing between attempts (recreate dir, retry)
    """
    url = f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        logger.info(f"[dermacon] skip (exists): {target.name}")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        # be defensive: ensure the directory still exists every iteration
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            already = tmp.stat().st_size if tmp.exists() else 0
        except OSError:
            already = 0
        headers = {"Range": f"bytes={already}-"} if already > 0 else {}
        bytes_this_attempt = 0
        try:
            with requests.get(url, stream=True, timeout=600, headers=headers) as r:
                if already > 0 and r.status_code == 200:
                    logger.info(f"[dermacon] {target.name}: server ignored Range; restarting")
                    tmp.unlink(missing_ok=True)
                    already = 0
                elif r.status_code not in (200, 206):
                    r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + already
                mode = "ab" if already > 0 else "wb"
                with open(str(tmp), mode) as f, tqdm(
                    total=total, initial=already, unit="B", unit_scale=True,
                    desc=target.name, leave=False, mininterval=5.0,
                ) as bar:
                    for buf in r.iter_content(chunk):
                        if buf:
                            f.write(buf)
                            bytes_this_attempt += len(buf)
                            bar.update(len(buf))
                            f.flush()
            tmp.replace(target)
            return
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
                FileNotFoundError, OSError) as e:
            try:
                on_disk = tmp.stat().st_size if tmp.exists() else 0
            except OSError:
                on_disk = 0
            logger.warning(f"[dermacon] {target.name} {type(e).__name__} "
                           f"(on-disk={on_disk:,}, this-attempt={bytes_this_attempt:,}); "
                           f"retry {attempt}/{max_attempts} in {delay:.0f}s")
            time.sleep(delay); delay = min(delay * 1.5, 60.0)
    raise RuntimeError(f"download_file: max retries exhausted for {target}")


def unpack_zip(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            # avoid path traversal
            if ".." in name or name.startswith("/") or name.startswith("\\"):
                continue
            target = dest / name
            if target.exists():
                continue
            z.extract(name, dest)
            n += 1
    return n


def download_dataset(out_dir: str | Path, *, doi: str = DOI, unzip: bool = True,
                     exclude_exts: tuple[str, ...] = (),
                     include_only_exts: tuple[str, ...] = ()) -> dict:
    """Download all files (default) OR filter by extension.

    `exclude_exts`: drop any file whose suffix is in this tuple, e.g. (".pth",)
    `include_only_exts`: keep only files whose suffix is in this tuple. If both
    are given, exclude_exts wins.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = get_dataset_files(doi)
    logger.info(f"[dermacon] {len(files)} files in dataset")
    excl = {e.lower() for e in exclude_exts}
    incl = {e.lower() for e in include_only_exts}
    stats = {"files": 0, "skipped": 0, "bytes": 0, "extracted": 0}
    for f in files:
        name = f.get("filename") or ""
        suf = Path(name).suffix.lower()
        if suf in excl or (incl and suf not in incl):
            stats["skipped"] += 1
            logger.info(f"[dermacon] skip (filter): {name}")
            continue
        target = out_dir / (f.get("directoryLabel", "") or "") / name
        download_file(f["id"], target)
        stats["files"] += 1
        stats["bytes"] += int(f.get("filesize") or 0)
        if unzip and (target.suffix.lower() == ".zip" or (f.get("contentType") or "").endswith("/zip")):
            n = unpack_zip(target, target.with_suffix(""))
            stats["extracted"] += n
    return stats


__all__ = ["get_dataset_files", "download_file", "unpack_zip", "download_dataset", "DOI"]
