import os, csv, json, time, argparse, random
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple, List
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

API_LIST = "https://api.isic-archive.com/api/v2/images"
API_IMG = "https://api.isic-archive.com/api/v2/images/{isic_id}/download"

# ---------------- Sleep prevention (Windows) ----------------
def set_keep_awake(enable: bool):
    if not enable or os.name != "nt":
        return lambda: None
    try:
        import ctypes
        ES_CONTINUOUS       = 0x80000000
        ES_SYSTEM_REQUIRED  = 0x00000001
        ES_AWAYMODE_REQUIRED= 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        def restore():
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("[keepawake] ON")
        return restore
    except Exception as e:
        print(f"[keepawake] failed: {e}")
        return lambda: None

# ---------------- HTTP helpers ----------------
def robust_get(session: requests.Session, url: str, params: Optional[dict]=None,
               timeout: int=60, retries: int=7, backoff: float=0.6):
    """
    GET with backoff; retries on network errors & 429/5xx.
    Returns (response, next_url_from_headers_or_json)
    """
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code in (429,500,502,503,504):
                raise requests.HTTPError(f"Transient {r.status_code}")
            r.raise_for_status()
            # Pull Link: rel="next" if present
            next_url = None
            if 'Link' in r.headers:
                from requests.utils import parse_header_links
                links = parse_header_links(r.headers['Link'].rstrip('>').replace('>,', ',<'))
                for lk in links:
                    if lk.get('rel') == 'next':
                        next_url = lk.get('url')
                        break
            # Some deployments also provide JSON next/next_url
            if (next_url is None) and 'application/json' in r.headers.get('Content-Type',''):
                try:
                    jj = r.json()
                    if isinstance(jj, dict):
                        next_url = jj.get('next') or jj.get('next_url')
                except Exception:
                    pass
            return r, next_url
        except Exception as e:
            sleep_s = backoff*(2**i) + random.uniform(0,0.4)
            print(f"[http] GET fail ({i+1}/{retries}) {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"GET failed after {retries} retries: {url}")

def ensure_filter_in_url(url: str) -> str:
    """Ensure the dermoscopic filter is present on the URL query string."""
    try:
        pr = urlparse(url)
        q = dict(parse_qsl(pr.query, keep_blank_values=True))
        key = 'metadata.acquisition.image_type'
        if key not in q:
            q[key] = 'dermoscopic'
            new_q = urlencode(q, doseq=True)
            pr = pr._replace(query=new_q)
            return urlunparse(pr)
        return url
    except Exception:
        return url

def parse_page_json(resp: requests.Response) -> List[dict]:
    """
    ISIC v2 returns a list for /images. Be strict and defensive.
    """
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Non-JSON response: {e}")
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get('results'), list):
        return data['results']
    raise RuntimeError(f"Unexpected JSON shape: {type(data)}")

def fetch_filtered_total(session: requests.Session) -> Optional[int]:
    """One-time fetch to get total count of dermoscopic images for ETA."""
    try:
        r = session.get(API_LIST, params={'limit': 1, 'metadata.acquisition.image_type': 'dermoscopic'}, timeout=30)
        r.raise_for_status()
        jj = r.json()
        if isinstance(jj, dict) and isinstance(jj.get('count'), int):
            return jj['count']
    except Exception:
        pass
    return None

def verify_img_bytes(b: bytes):
    Image.open(BytesIO(b)).verify()

# ---------------- Manifest / state ----------------
def load_have_ids(manifest: Path) -> set:
    have = set()
    if manifest.exists():
        with manifest.open('r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                iid = (r.get('isic_id') or '').strip()
                if iid:
                    have.add(iid)
    return have

def append_manifest_row(manifest: Path, row: list):
    # [isic_id, diagnosis, benign_malignant, modality, source, filepath]
    with manifest.open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

def save_state(path: Path, state: dict):
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    tmp.replace(path)

# ---------------- Download one ----------------
def download_one(session: requests.Session, root_img_dir: Path, img_meta: dict) -> Tuple[str, Optional[str], Optional[str], bool]:
    """
    Returns (isic_id, saved_path_or_None, err_msg_or_None)
    """
    isic_id = (img_meta.get('isic_id') or '').strip()
    if not isic_id:
        return "", None, "missing isic_id"

    acq = (img_meta.get('metadata') or {}).get('acquisition', {}) or {}
    modality = (acq.get('image_type') or '').strip().lower()
    if modality != 'dermoscopic':
        return isic_id, None, None  # skip others

    sub = isic_id.split('_')[1][:3] if '_' in isic_id else isic_id[:3]
    out_dir = root_img_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{isic_id}.jpg"
    part = out.with_suffix('.jpg.part')

    if out.exists():
        # File already on disk
        return isic_id, str(out), None, True

    # Prefer direct file URL from listing when available
    file_url = None
    try:
        files = (img_meta.get('files') or {})
        full = (files.get('full') or {})
        url_from_meta = full.get('url')
        if isinstance(url_from_meta, str) and url_from_meta.startswith('http'):
            file_url = url_from_meta
    except Exception:
        pass

    if not file_url:
        # Without a direct file URL, the API download endpoint may return 404 on v2.
        # Treat as non-fatal and let the caller count as error.
        return isic_id, None, "no-file-url", False
    url = file_url
    for i in range(6):
        try:
            r = session.get(url, timeout=60)
            if r.status_code in (429,500,502,503,504):
                raise requests.HTTPError(f"Transient {r.status_code}")
            r.raise_for_status()
            b = r.content
            verify_img_bytes(b)
            with open(part, 'wb') as f:
                f.write(b)
            part.replace(out)
            return isic_id, str(out), None, False
        except Exception as e:
            time.sleep(0.4*(2**i) + random.uniform(0,0.3))
    return isic_id, None, "download-failed", False

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\AIFINAL', help='Root containing images/, logs/, manifest.csv')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--max-images', type=int, default=None, help='Cap new downloads this run')
    ap.add_argument('--keepawake', action='store_true')
    args = ap.parse_args()

    ROOT = Path(args.root)
    IMG_DIR  = ROOT / 'images'
    LOG_DIR  = ROOT / 'logs'
    MANIFEST = ROOT / 'manifest.csv'
    STATE    = LOG_DIR / 'state.json'
    ERRLOG   = LOG_DIR / 'errors.log'
    IMG_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure manifest header exists
    if not MANIFEST.exists():
        with MANIFEST.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['isic_id','diagnosis','benign_malignant','modality','source','filepath'])

    # Known IDs: read once (no folder scan)
    have = load_have_ids(MANIFEST)

    # Load state or init
    state = {
        'offset': 0,
        'downloaded': len(have),
        'skipped': 0,
        'errors': 0,
        'next_url': None
    }
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text(encoding='utf-8'))
            state.update({k:v for k,v in prev.items() if k in state or k=='next_url'})
        except Exception:
            pass

    # Build first request (cursor-first if present)
    current_url = state.get('next_url') or API_LIST
    if state.get('next_url'):
        # Sanitize persisted cursor to ensure filter sticks
        current_url = ensure_filter_in_url(current_url)
        state['next_url'] = current_url
        save_state(STATE, state)
    params = None if state.get('next_url') else {'limit': args.limit, 'offset': state.get('offset',0),
                                                 'metadata.acquisition.image_type':'dermoscopic'}

    prevent_sleep_restore = set_keep_awake(args.keepawake)
    session = requests.Session()

    print(f"Start: offset={state['offset']} downloaded={state['downloaded']} skipped={state['skipped']} errors={state['errors']}")
    have_start = len(have)
    print(f"Known IDs in manifest: {have_start}")
    print(f"Saving to: {IMG_DIR}")
    print(f"Cursor mode: {'YES' if state.get('next_url') else 'NO'}")
    dermo_total = fetch_filtered_total(session)
    if dermo_total is not None:
        remaining_est = max(0, dermo_total - len(have))
        print(f"Dermoscopy total={dermo_total} | remaining={remaining_est}")

    new_this_run = 0
    start_ts = time.time()

    try:
        while True:
            # 1) GET a listing page
            resp, next_from_headers = robust_get(session, current_url, params=params)
            page = parse_page_json(resp)
            page_len = len(page) if isinstance(page, list) else 0
            if not page:
                print("done (empty page).")
                state['next_url'] = None
                save_state(STATE, state)
                break

            # 2) Build meta rows for this page, skip known IDs and non-dermoscopic
            metas = []
            skip_pre = 0
            exist_known = 0
            non_dermo = 0
            in_have = 0
            missing_id = 0
            for img in page:
                if not isinstance(img, dict):
                    state['skipped'] += 1
                    skip_pre += 1
                    continue
                isic_id = (img.get('isic_id') or '').strip()
                if not isic_id:
                    state['skipped'] += 1
                    skip_pre += 1
                    missing_id += 1
                    continue
                # If already known, count as existing on disk when present and skip scheduling
                if isic_id in have:
                    in_have += 1
                    sub = isic_id.split('_')[1][:3] if '_' in isic_id else isic_id[:3]
                    out_path = (IMG_DIR / sub / f"{isic_id}.jpg")
                    if out_path.exists():
                        exist_known += 1
                    state['skipped'] += 1
                    skip_pre += 1
                    continue
                # client-side modality guard to avoid pointless jobs if server forgot filter
                acq = (img.get('metadata') or {}).get('acquisition', {}) or {}
                modality = (acq.get('image_type') or '').strip().lower()
                if modality != 'dermoscopic':
                    state['skipped'] += 1
                    skip_pre += 1
                    non_dermo += 1
                    continue
                metas.append(img)

            # 3) Download this page in parallel
            ok, skip, err, exist = 0, 0, 0, 0
            def one(meta):
                iid, saved, emsg, existed = download_one(session, IMG_DIR, meta)
                return (iid, saved, emsg, existed, meta)

            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                futs = [ex.submit(one, m) for m in metas]
                for fut in as_completed(futs):
                    iid, saved, emsg, existed, meta = fut.result()
                    if not iid:
                        skip += 1; continue
                    if emsg is None and saved:
                        # Write manifest
                        md = meta.get('metadata') or {}
                        clin = (md.get('clinical') or {})
                        acq  = (md.get('acquisition') or {})
                        row = [
                            iid,
                            clin.get('diagnosis',''),
                            clin.get('benign_malignant',''),
                            acq.get('image_type',''),
                            meta.get('attribution',''),
                            saved
                        ]
                        append_manifest_row(MANIFEST, row)
                        have.add(iid)
                        if existed:
                            exist += 1
                        else:
                            ok += 1
                            new_this_run += 1
                        if args.max_images and new_this_run >= args.max_images:
                            break
                    elif emsg is None and saved is None:
                        skip += 1
                    else:
                        with ERRLOG.open('a', encoding='utf-8') as f:
                            f.write(f"[IMG] {iid}: {emsg}\n")
                        state['errors'] += 1
                        err += 1
                # hard cap check
                if args.max_images and new_this_run >= args.max_images:
                    pass

            state['downloaded'] += ok
            state['skipped']    += skip
            # include existing known on disk too
            exist_all = exist + exist_known
            state['existing']   = state.get('existing', 0) + exist_all

            # 4) Advance pagination: prefer cursor
            next_url = next_from_headers
            if (next_url is None) and 'application/json' in resp.headers.get('Content-Type',''):
                try:
                    jj = resp.json()
                    next_url = jj.get('next') or jj.get('next_url')
                except Exception:
                    next_url = None

            if next_url:
                current_url, params = ensure_filter_in_url(next_url), None
                state['next_url'] = current_url
            else:
                # Fallback to numeric offset only when cursor not provided
                state['offset'] = state.get('offset',0) + args.limit
                current_url = API_LIST
                params = {'limit': args.limit, 'offset': state['offset'],
                          'metadata.acquisition.image_type':'dermoscopic'}
                state['next_url'] = None

            save_state(STATE, state)
            mode = 'cursor' if state['next_url'] else f"offset={state['offset']}"
            elapsed = max(1e-6, time.time() - start_ts)
            rate = new_this_run / elapsed
            if args.max_images and rate > 0 and new_this_run < args.max_images:
                remain = args.max_images - new_this_run
                eta_sec = int(remain / rate)
                hrs, rem = divmod(eta_sec, 3600)
                mins, secs = divmod(rem, 60)
                eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                eta_str = "--"
            # ETA to finish entire dermoscopic set
            if dermo_total is not None and rate > 0:
                remain_all = max(0, dermo_total - (have_start + new_this_run))
                eta_all_sec = int(remain_all / rate) if remain_all > 0 else 0
                hrs, rem = divmod(eta_all_sec, 3600)
                mins, secs = divmod(rem, 60)
                eta_all = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                eta_all = "--"
            skip_total = skip_pre + skip
            # Additional debug to help diagnose zero counters
            dbg = f"page={page_len} queued={len(metas)} in_have={in_have} non_dermo={non_dermo} missing_id={missing_id}"
            print(f"[page] ok={ok} exist={exist_all} skip={skip_total} err={err} | total_ok={state['downloaded']} exist_total={state.get('existing',0)} | next={mode} | rate={rate:.2f}/s | eta={eta_str} | eta_all={eta_all} | {dbg}")

            if args.max_images and new_this_run >= args.max_images:
                print("Reached --max-images cap; stopping.")
                break

            time.sleep(0.15)  # gentle pacing between pages

    finally:
        try: prevent_sleep_restore()
        except: pass

if __name__ == '__main__':
    main()

