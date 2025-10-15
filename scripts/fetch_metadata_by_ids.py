#!/usr/bin/env python
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

API = "https://api.isic-archive.com/api/v2/images/"


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    lg = logging.getLogger("isic_by_ids")
    lg.setLevel(lvl)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


def session_with_retries() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "MNKAI-ISIC-BY-IDS/1.0"})
    retry = Retry(total=7, connect=5, read=5, status=5, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def load_needed_ids(ids_csv: str) -> List[str]:
    df = pd.read_csv(ids_csv)
    if "isic_id" not in df.columns:
        raise SystemExit("ids_csv must contain an 'isic_id' column")
    ids = df["isic_id"].dropna().astype(str).unique().tolist()
    return ids


def load_done(out_csv: Path) -> set:
    done = set()
    for p in (out_csv, out_csv.with_suffix(out_csv.suffix + ".part")):
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p, usecols=["isic_id"])
                done |= set(df["isic_id"].astype(str).tolist())
            except Exception:
                with p.open("r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if row.get("isic_id"):
                            done.add(str(row["isic_id"]))
    return done


def flatten_row(obj: dict) -> dict:
    meta = obj.get("metadata") or {}
    clin = meta.get("clinical") or {}
    acq = meta.get("acquisition") or {}
    modality = acq.get("image_type") or meta.get("image_type") or meta.get("modality") or ""
    return {
        "isic_id": str(obj.get("isic_id") or obj.get("id") or ""),
        "diagnosis": str(clin.get("diagnosis") or "").strip().lower(),
        "benign_malignant": str(clin.get("benign_malignant") or "").strip().lower(),
        "sex": str(clin.get("sex") or "").strip().lower(),
        "age_approx": clin.get("age_approx"),
        "anatom_site_general": str(clin.get("anatom_site_general") or "").strip().lower(),
        "modality": str(modality).strip().lower(),
    }


def flush_rows(rows: List[dict], part_file: Path, header_written: Dict[str, bool], log: logging.Logger):
    if not rows:
        return
    write_header = not part_file.exists() or not header_written.get(str(part_file), False)
    df = pd.DataFrame(rows)
    part_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(part_file, index=False, mode="a", header=write_header)
    header_written[str(part_file)] = True
    log.debug(f"[flush] +{len(rows)} rows -> '{part_file}'")
    rows.clear()


def finalize(part_file: Path, final_file: Path, log: logging.Logger):
    if not part_file.exists():
        log.warning("[finalize] no .part file; nothing to finalize")
        return
    df = pd.read_csv(part_file)
    if "isic_id" not in df.columns:
        log.error("[finalize] missing isic_id col in part file")
        return
    df = df.drop_duplicates(subset=["isic_id"]).reset_index(drop=True)
    tmp = final_file.with_suffix(final_file.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(final_file)
    log.info(f"[finalize] wrote {final_file} (rows={len(df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_csv", required=True, help="CSV with isic_id column (universe + order)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--status-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--log-file", default="logs/fetch_isic_metadata_by_ids.log")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    log = setup_logger(args.log_file, args.log_level)
    out_csv = Path(args.out_csv)
    part_csv = out_csv.with_suffix(out_csv.suffix + ".part")

    need_list = load_needed_ids(args.ids_csv)
    done = load_done(out_csv) if args.resume else set()
    pending = [iid for iid in need_list if iid not in done]
    log.info(f"need={len(need_list)} done={len(done)} pending={len(pending)}")
    if not pending:
        finalize(part_csv, out_csv, log)
        log.info("Nothing to do.")
        return

    sess = session_with_retries()
    header_written: Dict[str, bool] = {}
    buf: List[dict] = []
    started = time.time()
    accepted = 0
    errors = 0

    def fetch_one(isic_id: str):
        url = API + isic_id
        try:
            r = sess.get(url, params={"include": "metadata"}, timeout=(10, 60))
            if r.status_code == 404:
                return isic_id, None, "not-found"
            r.raise_for_status()
            obj = r.json()
            if not isinstance(obj, dict):
                return isic_id, None, "bad-json"
            return isic_id, flatten_row(obj), None
        except Exception as e:
            return isic_id, None, str(e)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_one, iid) for iid in pending]
            for i, fut in enumerate(as_completed(futs), 1):
                iid, row, err = fut.result()
                if row is not None:
                    buf.append(row)
                    accepted += 1
                    if accepted % args.save_every == 0:
                        flush_rows(buf, part_csv, header_written, log)
                else:
                    errors += 1
                    if errors % 50 == 0:
                        log.warning(f"[errors] total={errors} (latest for {iid}: {err})")

                if accepted and accepted % args.status_every == 0:
                    elapsed = time.time() - started
                    rate = accepted / max(elapsed, 1e-6)
                    log.info(f"[status] accepted={accepted} | rate={rate:.1f}/s | progress={accepted}/{len(pending)}")

        # flush remaining
        if buf:
            flush_rows(buf, part_csv, header_written, log)

        finalize(part_csv, out_csv, log)
        elapsed = time.time() - started
        log.info(f"=== Completed: accepted={accepted} errors={errors} in {elapsed:.0f}s ===")

    except KeyboardInterrupt:
        if buf:
            flush_rows(buf, part_csv, header_written, log)
        log.warning("[interrupt] partial progress saved (.part). Re-run to continue.")
        sys.exit(1)


if __name__ == "__main__":
    main()

