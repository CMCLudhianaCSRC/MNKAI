"""Live overnight status dashboard for MNKAI-III.

Serves a self-refreshing web page at http://127.0.0.1:8790 that shows what the agent is
doing right now: the pipeline stepper, current step, background tasks, key results, and a
live log feed. Reads logs/live_status.json fresh on every poll, so it is always current.

    python scripts/live_dashboard.py            # port 8790
    python scripts/live_dashboard.py --port 8791
"""
from __future__ import annotations
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "logs" / "live_status.json"

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNKAI-III - Live</title>
<style>
:root{--bg:#0a0d14;--card:#121826;--card2:#0f1420;--line:#1e2636;--tx:#e6edf7;--mut:#8595ad;
--accent:#4da3ff;--ok:#38d39f;--run:#ffb454;--pend:#556074;--hot:#ff6b8b;}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#141c2e 0%,var(--bg) 60%);
color:var(--tx);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
-webkit-font-smoothing:antialiased;padding:22px;max-width:1080px;margin:0 auto}
h1{font-size:20px;margin:0;font-weight:650;letter-spacing:.2px}
.sub{color:var(--mut);font-size:13px;margin-top:2px}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.spacer{flex:1}
.live{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--ok);
background:#0e2019;border:1px solid #1c4636;padding:5px 10px;border-radius:20px;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 0 rgba(56,211,159,.7);
animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(56,211,159,.6)}70%{box-shadow:0 0 0 9px rgba(56,211,159,0)}100%{box-shadow:0 0 0 0 rgba(56,211,159,0)}}
.ago{color:var(--mut);font-size:12px}
.card{background:linear-gradient(180deg,var(--card) 0%,var(--card2) 100%);border:1px solid var(--line);
border-radius:14px;padding:16px 18px;margin-top:16px}
.now{border-color:#284066;background:linear-gradient(180deg,#15233b 0%,#101a2c 100%)}
.now .k{color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:1.4px;font-weight:700}
.now .v{font-size:18px;font-weight:600;margin-top:5px}
.now .d{color:var(--mut);font-size:13.5px;margin-top:6px}
.now .n{color:#a9c3e6;font-size:13px;margin-top:10px;border-top:1px dashed #26344d;padding-top:9px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.h{font-size:12px;text-transform:uppercase;letter-spacing:1.3px;color:var(--mut);font-weight:700;margin:0 0 4px}
.step{display:flex;gap:12px;padding:9px 0;border-bottom:1px solid #161d2b}
.step:last-child{border-bottom:0}
.ic{flex:none;width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font-size:12px;margin-top:1px}
.done .ic{background:#0e2019;color:var(--ok);border:1px solid #1c4636}
.running .ic{background:#241a08;color:var(--run);border:1px solid #4a3413;animation:pulse2 1.6s infinite}
@keyframes pulse2{0%,100%{opacity:1}50%{opacity:.45}}
.pending .ic{background:#141a26;color:var(--pend);border:1px solid #222c3e}
.error .ic{background:#2a1119;color:var(--hot);border:1px solid #582231}
.step .t{font-weight:560}
.running .t{color:#ffd9a0}
.pending .t{color:#93a1b8}
.step .sn{color:var(--mut);font-size:12.5px;margin-top:2px}
.task{display:flex;align-items:center;gap:9px;padding:7px 0;border-bottom:1px solid #161d2b}
.task:last-child{border-bottom:0}
.badge{font-size:11px;font-weight:700;padding:3px 8px;border-radius:6px}
.b-done{background:#0e2019;color:var(--ok)}.b-running{background:#241a08;color:var(--run)}
.b-pending{background:#141a26;color:var(--pend)}.b-error{background:#2a1119;color:var(--hot)}
.task .tn{font-weight:540}.task .td{color:var(--mut);font-size:12.5px}
.res{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid #161d2b}
.res:last-child{border-bottom:0}
.res .m{color:#c6d3e6;font-size:13.5px}.res .rn{color:var(--mut);font-size:12px}
.res .val{font-weight:700;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.res.hot .val{color:var(--ok)}
.logf{max-height:340px;overflow:auto}
.le{display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #141b28;font-size:13.5px}
.le:last-child{border-bottom:0}
.le .lt{color:var(--accent);font-variant-numeric:tabular-nums;flex:none}
.le .lm{color:#cdd8e8}
.next{color:#a9c3e6;font-size:13.5px}
.err{color:var(--hot)}
.flash{animation:fl .8s}@keyframes fl{from{background:#16263e}to{background:transparent}}
footer{color:#4a566b;font-size:11.5px;text-align:center;margin:20px 0 6px}
</style></head>
<body>
<div class="row">
  <div><h1>MNKAI-III &middot; Live Overnight Status</h1><div class="sub" id="title">-</div></div>
  <div class="spacer"></div>
  <span class="live"><span class="dot"></span> LIVE</span>
  <span class="ago" id="ago">-</span>
</div>

<div class="card now">
  <div class="k">What I'm doing right now</div>
  <div class="v" id="step">-</div>
  <div class="d" id="detail"></div>
  <div class="n" id="next"></div>
</div>

<div class="card">
  <div class="h">Pipeline</div>
  <div id="pipeline"></div>
</div>

<div class="grid">
  <div class="card"><div class="h">Background tasks</div><div id="tasks"></div></div>
  <div class="card"><div class="h">Key results</div><div id="results"></div></div>
</div>

<div class="card">
  <div class="h">Live log</div>
  <div class="logf" id="log"></div>
</div>

<footer>Auto-refreshing every 3s &middot; reads logs/live_status.json &middot; you can close this any time, work continues</footer>

<script>
const ICON={done:"✓",running:"●",pending:"○",error:"!"};
let lastStep="";
function ago(iso){try{const s=Math.max(0,(Date.now()-new Date(iso).getTime())/1000);
 if(s<60)return Math.round(s)+"s ago";if(s<3600)return Math.round(s/60)+"m ago";return Math.round(s/3600)+"h ago";}catch(e){return"-";}}
function esc(x){return(x==null?"":""+x).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
async function tick(){
 let d;try{d=await(await fetch("/status.json?t="+Date.now())).json();}catch(e){document.getElementById("ago").textContent="(offline)";return;}
 document.getElementById("title").textContent=d.session_title||"";
 document.getElementById("ago").textContent="updated "+ago(d.updated_at);
 const step=d.current_step||"-";
 const se=document.getElementById("step");se.textContent=step;
 if(step!==lastStep){se.parentElement.classList.remove("flash");void se.offsetWidth;se.parentElement.classList.add("flash");lastStep=step;}
 document.getElementById("detail").textContent=d.current_detail||"";
 document.getElementById("next").innerHTML=d.next?("<b>Next:</b> "+esc(d.next)):"";
 document.getElementById("pipeline").innerHTML=(d.pipeline||[]).map(p=>
  `<div class="step ${esc(p.status)}"><div class="ic">${ICON[p.status]||"○"}</div>
   <div><div class="t">${esc(p.title)}</div>${p.note?`<div class="sn">${esc(p.note)}</div>`:""}</div></div>`).join("");
 document.getElementById("tasks").innerHTML=(d.tasks||[]).map(t=>
  `<div class="task"><span class="badge b-${esc(t.status)}">${esc((t.status||"").toUpperCase())}</span>
   <div><div class="tn">${esc(t.name)}</div>${t.detail?`<div class="td">${esc(t.detail)}</div>`:""}</div></div>`).join("");
 document.getElementById("results").innerHTML=(d.results||[]).map(r=>
  `<div class="res ${r.highlight?"hot":""}"><div><div class="m">${esc(r.metric)}</div>${r.note?`<div class="rn">${esc(r.note)}</div>`:""}</div>
   <div class="val">${esc(r.value)}</div></div>`).join("");
 document.getElementById("log").innerHTML=(d.log||[]).map(l=>
  `<div class="le"><span class="lt">${esc(l.t)}</span><span class="lm">${esc(l.msg)}</span></div>`).join("");
}
tick();setInterval(tick,3000);
</script>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body, ctype):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/status.json"):
            try:
                data = STATUS.read_text(encoding="utf-8")
            except Exception:
                data = '{"current_step":"(waiting for first status write)","pipeline":[],"tasks":[],"results":[],"log":[]}'
            self._send(data, "application/json")
        else:
            self._send(PAGE, "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"MNKAI-III live dashboard -> http://127.0.0.1:{a.port}  (Ctrl-C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
