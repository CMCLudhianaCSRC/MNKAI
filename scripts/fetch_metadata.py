#!/usr/bin/env python
import argparse, csv, json, logging, os, sys, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

API = "https://api.isic-archive.com/api/v2/images/"

def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    lg = logging.getLogger("isic_bulk")
    lg.setLevel(lvl); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(lvl); sh.setFormatter(fmt); lg.addHandler(sh)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setLevel(lvl); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg

def session_with_retries() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "MNKAI-ISIC-BULK/1.0"})
    retry = Retry(total=7, connect=5, read=5, status=5, backoff_factor=0.5,
                  status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

def load_needed_ids(ids_csv: Optional[str]) -> Optional[set]:
    if not ids_csv: return None
    df = pd.read_csv(ids_csv)
    if "isic_id" not in df.columns:
        raise SystemExit("ids_csv must contain an 'isic_id' column")
    return set(df["isic_id"].dropna().astype(str).tolist())

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
    # v2 /images returns per-image dicts; clinical nested in metadata.clinical
    meta = obj.get("metadata") or {}
    clin = meta.get("clinical") or {}
    acq = meta.get("acquisition") or {}
    # image_type is under metadata.acquisition.image_type in v2
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

def flush_rows(rows: List[dict], part_file: Path, header_written: Dict[str,bool], log: logging.Logger):
    if not rows: return
    write_header = not part_file.exists() or not header_written.get(str(part_file), False)
    df = pd.DataFrame(rows)
    part_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(part_file, index=False, mode="a", header=write_header)
    header_written[str(part_file)] = True
    log.debug(f"[flush] +{len(rows)} rows → {part_file}")
    rows.clear()

def finalize(part_file: Path, final_file: Path, log: logging.Logger):
    if not part_file.exists():
        log.warning("[finalize] no .part file; nothing to finalize"); return
    df = pd.read_csv(part_file)
    if "isic_id" not in df.columns:
        log.error("[finalize] missing isic_id col in part file"); return
    df = df.drop_duplicates(subset=["isic_id"]).reset_index(drop=True)
    tmp = final_file.with_suffix(final_file.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(final_file)
    log.info(f"[finalize] wrote {final_file} (rows={len(df)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_csv", default="data/manifest_unified.csv",
                    help="If present, filter to these IDs only; must contain isic_id col")
    ap.add_argument("--out_csv", default="data/isic_meta.csv")
    ap.add_argument("--limit", type=int, default=200, help="page size (100–500 good)")
    ap.add_argument("--status-every", type=int, default=100, help="log every N accepted rows")
    ap.add_argument("--save-every", type=int, default=100, help="flush every N accepted rows")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--cursor_file", default=None, help="custom cursor file (defaults to out_csv.cursor)")
    ap.add_argument("--log-file", default="logs/fetch_isic_metadata_bulk.log")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--max-pages", type=int, default=0, help="stop after N pages (debug)")
    ap.add_argument("--dermo-only", action="store_true", default=False,
                    help="filter to dermoscopic via server-side query if available")
    args = ap.parse_args()

    log = setup_logger(args.log_file, args.log_level)
    out_csv = Path(args.out_csv)
    part_csv = out_csv.with_suffix(out_csv.suffix + ".part")
    cursor_file = Path(args.cursor_file) if args.cursor_file else out_csv.with_suffix(out_csv.suffix + ".cursor")

    need = load_needed_ids(args.ids_csv) if args.ids_csv else None
    done = load_done(out_csv) if args.resume else set()
    if done: log.info(f"[resume] already have {len(done)} rows; will skip those IDs")

    # Read saved cursor if resuming a previous bulk run
    cursor = None
    if args.resume and cursor_file.exists():
        cursor = cursor_file.read_text(encoding="utf-8").strip() or None
        if cursor: log.info(f"[resume] resuming from saved cursor")

    sess = session_with_retries()
    params = {"limit": args.limit, "include": "metadata"}
    if args.dermo_only:
        # server-side filter; API supports metadata.* filters; keep tolerant
        params["metadata.acquisition.image_type"] = "dermoscopic"

    accepted = 0
    page = 0
    buf: List[dict] = []
    header_written = {}
    started = time.time()

    try:
        while True:
            q = dict(params)
            if cursor: q["cursor"] = cursor
            r = sess.get(API, params=q, timeout=(10, 60))  # connect/read timeouts
            if r.status_code != 200:
                log.warning(f"[page] HTTP {r.status_code}; sleeping 2s and retrying…")
                time.sleep(2); continue
            j = r.json()
            results = j.get("results") or j.get("images") or j.get("data") or []
            page += 1
            got = 0
            for obj in results:
                row = flatten_row(obj)
                isid = row.get("isic_id") or ""
                if not isid: continue
                if need is not None and isid not in need:  # not requested
                    continue
                if isid in done:  # already have
                    continue
                buf.append(row); done.add(isid)
                accepted += 1; got += 1
                if accepted % args.status_every == 0:
                    elapsed = time.time() - started
                    rate = accepted / max(elapsed, 1e-6)
                    log.info(f"[status] accepted={accepted} | rate={rate:.1f}/s | page={page} (got={got})")
                if accepted % args.save_every == 0:
                    flush_rows(buf, part_csv, header_written, log)
                    # persist cursor for robust resume
                    cursor_file.write_text(cursor or "", encoding="utf-8")

            # flush at end of page too so you see the file quickly
            if buf:
                flush_rows(buf, part_csv, header_written, log)
                cursor_file.write_text(cursor or "", encoding="utf-8")

            # Next page?
            next_url = j.get("next") or (j.get("links", {}) if isinstance(j.get("links", {}), dict) else {}).get("next") or j.get("next_url")
            if next_url:
                # extract cursor param from next_url, else keep previous
                try:
                    import urllib.parse as up
                    qs = dict(up.parse_qsl(up.urlsplit(next_url).query))
                    cursor = qs.get("cursor", cursor)
                except Exception:
                    pass
            else:
                log.info("[done] no next page")
                break

            if args.max_pages and page >= args.max_pages:
                log.info(f"[debug] stopping after max-pages={args.max_pages}")
                break

        # finalize
        finalize(part_csv, out_csv, log)
        if cursor_file.exists():
            cursor_file.unlink(missing_ok=True)
        elapsed = time.time() - started
        log.info(f"=== Completed: {accepted} rows in {elapsed:.0f}s ===")

    except KeyboardInterrupt:
        if buf:
            flush_rows(buf, part_csv, header_written, log)
        cursor_file.write_text(cursor or "", encoding="utf-8")
        log.warning("[interrupt] partial progress saved (.part + .cursor). Re-run with --resume to continue.")
        sys.exit(1)

if __name__ == "__main__":
    main()
