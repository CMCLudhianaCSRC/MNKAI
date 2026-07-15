"""Lightweight phone-friendly live status server for MNKAI-III training.

Serves an auto-refreshing HTML page showing:
  - the latest training log (tail)
  - parsed progress bar (SSL steps or fold epochs)
  - live GPU utilisation / VRAM / temp / power

stdlib only — no Flask, no installs. Read-only. Safe to leave running.

Usage:
  python -m scripts._status_server               # port 8765
  python -m scripts._status_server --port 9000

View from any device on the same WiFi:  http://<PC-LAN-IP>:8765
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, time, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "logs"
STATE_FILE = LOGDIR / "pipeline_state.json"

# progress line patterns
RE_SSL = re.compile(r"\[ssl\]\s+e(\d+)\s+step\s+(\d+)/(\d+)\s+loss=([\d.]+)")
RE_EPOCH = re.compile(r"\[epoch\s+(\d+)\]\s+loss=([\d.]+).*?macroF1=([\d.]+)", re.I)
# full per-epoch metrics line:
# [epoch 4] loss=0.5470 | val_loss=1.3077 acc=0.5627 bal_acc=0.4949 macroF1=0.3171 AUROC=0.9295
RE_EPOCH_FULL = re.compile(
    r"\[epoch (\d+)\] loss=([\d.]+) \| val_loss=([\d.]+) acc=([\d.]+) "
    r"bal_acc=([\d.]+) macroF1=([\d.]+) AUROC=([\d.nan]+)")

# fold-0 reference (completed run) for side-by-side comparison
FOLD0_REF = {1: (0.005, 0.787), 2: (0.479, 0.905), 3: (0.559, 0.931),
             4: (0.603, 0.939), 5: (0.630, 0.943), 6: (0.679, 0.946),
             7: (0.708, 0.949), 8: (0.721, 0.950), 9: (0.723, 0.950),
             10: (0.714, 0.951), 11: (0.716, 0.950), 12: (0.722, 0.950)}


def latest_log() -> Path | None:
    """Most recently modified .log / .log.err under logs/."""
    cands = list(LOGDIR.glob("*.log")) + list(LOGDIR.glob("*.log.err"))
    cands = [c for c in cands if c.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def tail(path: Path, n: int = 30) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except Exception as e:
        return [f"(could not read log: {e})"]


def gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        u, mu, mt, t, p = [x.strip() for x in out.split(",")]
        return {"util": u, "mem_used": mu, "mem_total": mt, "temp": t, "power": p}
    except Exception:
        return {}


def parse_progress(lines: list[str]) -> dict:
    """Scan log lines bottom-up for the latest progress marker."""
    for ln in reversed(lines):
        m = RE_SSL.search(ln)
        if m:
            ep, step, total, loss = int(m[1]), int(m[2]), int(m[3]), float(m[4])
            ssl_epochs = 2  # current SSL run is configured for 2 epochs
            done = (ep - 1) * total + step
            grand = total * ssl_epochs
            return {"kind": "SSL pretraining",
                    "label": f"epoch {ep}/{ssl_epochs} · step {step}/{total}",
                    "loss": loss, "frac": min(1.0, done / max(1, grand))}
        m = RE_EPOCH.search(ln)
        if m:
            ep, loss, f1 = int(m[1]), float(m[2]), float(m[3])
            fold_epochs = 50  # full proper fold training
            return {"kind": "Fold training",
                    "label": f"epoch {ep}/{fold_epochs} · macroF1 {f1:.4f}",
                    "loss": loss, "frac": min(1.0, ep / fold_epochs)}
    return {}


def bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def pipeline_html(live_frac: float | None) -> str:
    """Render the full-pipeline infographic from pipeline_state.json.
    The phase marked 'running' uses live_frac (parsed from the log) if given."""
    try:
        st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return ""
    rows = ""
    for ph in st.get("phases", []):
        status = ph.get("status", "pending")
        pct = ph.get("pct", 0)
        if status == "running" and live_frac is not None:
            pct = round(live_frac * 100, 1)
        if status == "done":
            dot, col, w = "&#9679;", "#3fb950", 100
        elif status == "running":
            dot, col, w = "&#9654;", "#d29922", pct
        else:
            dot, col, w = "&#9675;", "#6e7681", 0
        rows += (
            f"<div class='ph'><div class='phh'>"
            f"<span style='color:{col}'>{dot}</span> "
            f"<b>{html.escape(ph.get('name',''))}</b>"
            f"<span class='phpct'>{pct if status!='pending' else ''}"
            f"{'%' if status!='pending' else ''}</span></div>"
            f"<div class='phbarbg'><div class='phbar' "
            f"style='width:{w}%;background:{col}'></div></div>"
            f"<div class='phnote'>{html.escape(ph.get('note',''))}</div></div>")
    return (
        "<div class='card'><div class='k'>&#128293; Full pipeline &middot; "
        f"deadline {html.escape(st.get('deadline','-'))} &middot; "
        f"projected {html.escape(st.get('projected_done','-'))}</div>"
        f"{rows}</div>")


def epoch_table() -> str:
    """Per-epoch score table for the current fold, with fold-0 reference."""
    lg = latest_log()
    if not lg:
        return ""
    try:
        text = lg.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    rows = RE_EPOCH_FULL.findall(text)
    if not rows:
        return ""
    body = ""
    for ep, loss, vloss, acc, bal, f1, auroc in rows[-50:]:
        epn = int(ep)
        ref = FOLD0_REF.get(epn)
        ref_txt = f"{ref[0]:.3f}/{ref[1]:.3f}" if ref else "&mdash;"
        body += (f"<tr><td>{ep}</td><td><b>{float(acc):.3f}</b></td>"
                 f"<td>{float(bal):.3f}</td><td>{float(f1):.3f}</td>"
                 f"<td>{float(auroc):.3f}</td><td>{float(loss):.2f}</td>"
                 f"<td class='ref'>{ref_txt}</td></tr>")
    return (
        "<div class='card'><div class='k'>&#128202; Per-epoch scores &middot; "
        f"current fold ({len(rows)} epochs)</div>"
        "<table class='et'><tr><th>ep</th><th>acc</th><th>bal</th>"
        "<th>mF1</th><th>AUROC</th><th>loss</th><th>f0 ref</th></tr>"
        f"{body}</table>"
        "<div class='phnote'>f0 ref = fold-0 acc/AUROC at same epoch &middot; "
        "fold-0 final: acc 0.854 &middot; macroF1 0.640 &middot; AUROC 0.952</div></div>")


def render() -> str:
    lg = latest_log()
    lines = tail(lg, 30) if lg else ["(no log files found yet)"]
    prog = parse_progress(lines)
    g = gpu_stats()
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    prog_html = ""
    if prog:
        pct = prog["frac"] * 100
        prog_html = (
            f"<div class='card'><div class='k'>{html.escape(prog['kind'])}</div>"
            f"<div class='bar'>{bar(prog['frac'])}</div>"
            f"<div class='big'>{pct:.1f}%</div>"
            f"<div class='sub'>{html.escape(prog['label'])} &middot; "
            f"loss {prog['loss']:.3f}</div></div>")

    gpu_html = "<div class='card'><div class='k'>GPU not queryable</div></div>"
    if g:
        mempct = 0
        try:
            mempct = 100 * float(g['mem_used']) / float(g['mem_total'])
        except Exception:
            pass
        tcol = "#3fb950" if float(g.get("temp", 0) or 0) < 75 else "#f85149"
        gpu_html = (
            "<div class='card'><div class='k'>GPU</div>"
            f"<div class='row'><span>Utilisation</span><b>{g['util']}%</b></div>"
            f"<div class='row'><span>VRAM</span><b>{g['mem_used']}/{g['mem_total']} MiB "
            f"({mempct:.0f}%)</b></div>"
            f"<div class='row'><span>Temp</span><b style='color:{tcol}'>{g['temp']} &deg;C</b></div>"
            f"<div class='row'><span>Power</span><b>{g['power']} W</b></div></div>")

    pipe_html = pipeline_html(prog.get("frac") if prog else None)
    etable = epoch_table()

    log_html = html.escape("".join(lines))
    fname = lg.name if lg else "-"

    return f"""<!doctype html><html><head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<meta http-equiv='refresh' content='8'>
<title>MNKAI-III status</title>
<style>
 body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,Segoe UI,monospace;
       margin:0;padding:14px;font-size:15px}}
 h1{{font-size:17px;margin:0 0 4px}}
 .t{{color:#8b949e;font-size:12px;margin-bottom:12px}}
 .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;
        padding:12px 14px;margin-bottom:10px}}
 .k{{color:#58a6ff;font-weight:700;margin-bottom:6px}}
 .bar{{font-family:monospace;font-size:18px;letter-spacing:1px;color:#3fb950}}
 .big{{font-size:30px;font-weight:800;margin:4px 0}}
 .sub{{color:#8b949e;font-size:13px}}
 .row{{display:flex;justify-content:space-between;padding:3px 0;
       border-bottom:1px solid #21262d}}
 .ph{{margin:9px 0}}
 .phh{{display:flex;justify-content:space-between;font-size:13px}}
 .phpct{{color:#8b949e;font-size:12px}}
 .phbarbg{{background:#21262d;border-radius:4px;height:7px;margin:3px 0}}
 .phbar{{height:7px;border-radius:4px}}
 .phnote{{color:#6e7681;font-size:11px;margin-top:4px}}
 table.et{{width:100%;border-collapse:collapse;font-size:12px;margin-top:4px}}
 table.et th{{color:#58a6ff;text-align:right;padding:3px 5px;
              border-bottom:1px solid #30363d}}
 table.et td{{text-align:right;padding:3px 5px;border-bottom:1px solid #21262d;
              font-variant-numeric:tabular-nums}}
 table.et td:first-child,table.et th:first-child{{text-align:left}}
 table.et td.ref{{color:#6e7681}}
 pre{{background:#010409;border:1px solid #30363d;border-radius:8px;padding:10px;
      overflow-x:auto;font-size:11px;line-height:1.4;color:#8b949e;white-space:pre-wrap}}
</style></head><body>
<h1>&#129504; MNKAI-III &middot; live status</h1>
<div class='t'>updated {now} &middot; auto-refresh 8s &middot; log: {html.escape(fname)}</div>
{prog_html}
{gpu_html}
{etable}
{pipe_html}
<div class='card'><div class='k'>Log tail</div><pre>{log_html}</pre></div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # silence

    def do_GET(self):
        body = render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[status] serving on http://0.0.0.0:{args.port}  "
          f"(open http://<PC-LAN-IP>:{args.port} from your phone)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
