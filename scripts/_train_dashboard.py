"""MNKAI-III · live training dashboard (futuristic).

Parses the ablation driver + per-fix training logs and serves a live,
auto-refreshing visualizer: current fix/fold/epoch with progress bars, the full
pipeline rail, GPU gauge, and live loss / macro-F1 / AUROC charts.

CPU-only (just reads text logs + nvidia-smi) — zero impact on training.

Run:  .venv/Scripts/python.exe scripts/_train_dashboard.py --port 8770
Open: http://localhost:8770   (also reachable on the LAN IP)
"""
from __future__ import annotations
import argparse, json, re, subprocess, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"

RE_EPOCH = re.compile(r"\[epoch (\d+)\] loss=([\d.]+) \| val_loss=([\d.]+) "
                      r"acc=([\d.]+) bal_acc=([\d.]+) macroF1=([\d.]+) AUROC=([\d.]+)")
RE_PROG = re.compile(r"epoch (\d+)/(\d+):\s+(\d+)%.*?(\d+)/(\d+) \[[^\]]*?([\d.]+)it/s, loss=([\d.]+)")
RE_BEST = re.compile(r"saved best \(macro_f1=([\d.]+)\)")
RE_FIXLOG = re.compile(r"ablation_phase3_fix([ABC])_.*_fold(\d)_train\.log")
RE_SCORE = re.compile(r"FIX ([ABC]): score=([\-\d.]+) (\{.*\})")
RE_WINNER = re.compile(r"WINNER = FIX ([ABC])")
RE_STAGE = re.compile(r"(STAGE \d[^\n]*)")

FIX_NAMES = {"A": "CE + gentle sampler (√)", "B": "Balanced-Softmax", "C": "CB-Focal (no resample)"}


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=8).strip().splitlines()[0]
        u, m, t, tp = [x.strip() for x in out.split(",")]
        return {"util": int(u), "mem": int(m), "mem_total": int(t), "temp": int(tp)}
    except Exception:
        return {"util": 0, "mem": 0, "mem_total": 0, "temp": 0}


def active_train_log():
    cands = list(LOGS.glob("ablation_*_fold*_train.log"))
    cands = [c for c in cands if c.is_file()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def state():
    drv = read(LOGS / "ablation_driver.log")
    stages = RE_STAGE.findall(drv)
    stage = stages[-1].strip() if stages else "initialising…"
    win = RE_WINNER.findall(drv)
    winner = win[-1] if win else None
    scores = {}
    for k, s, summ in RE_SCORE.findall(drv):
        try:
            scores[k] = {"score": float(s), **json.loads(summ.replace("'", '"').replace("True","true").replace("False","false"))}
        except Exception:
            scores[k] = {"score": float(s)}

    cur = {"fix": None, "fold": None, "epoch": 0, "total_epochs": 30, "pct": 0,
           "step": 0, "total_steps": 0, "it_s": 0.0, "loss": 0.0, "log": None}
    epochs = []
    lg = active_train_log()
    if lg is not None:
        txt = read(lg).replace("\r", "\n")
        m = RE_FIXLOG.search(lg.name)
        if m:
            cur["fix"], cur["fold"] = m.group(1), int(m.group(2))
        cur["log"] = lg.name
        for e in RE_EPOCH.finditer(txt):
            epochs.append({"epoch": int(e.group(1)), "loss": float(e.group(2)),
                           "val_loss": float(e.group(3)), "acc": float(e.group(4)),
                           "bal_acc": float(e.group(5)), "macroF1": float(e.group(6)),
                           "auroc": float(e.group(7))})
        progs = RE_PROG.findall(txt)
        if progs:
            ep, tot, pct, step, tsteps, its, loss = progs[-1]
            cur.update({"epoch": int(ep), "total_epochs": int(tot), "pct": int(pct),
                        "step": int(step), "total_steps": int(tsteps),
                        "it_s": float(its), "loss": float(loss)})
        best = RE_BEST.findall(txt)
        cur["best_f1"] = float(best[-1]) if best else (epochs[-1]["macroF1"] if epochs else 0.0)
        age = time.time() - lg.stat().st_mtime
        cur["live"] = age < 90

    # pipeline rail
    seq = ["A", "B", "C"]
    fix_status = {}
    for k in seq:
        if k in scores:
            fix_status[k] = "done"
        elif cur["fix"] == k:
            fix_status[k] = "running"
        else:
            fix_status[k] = "pending"
    rail = []
    rail.append({"name": "Fix A · " + FIX_NAMES["A"], "status": fix_status["A"],
                 "info": _fixinfo(scores.get("A"))})
    rail.append({"name": "Fix B · " + FIX_NAMES["B"], "status": fix_status["B"],
                 "info": _fixinfo(scores.get("B"))})
    rail.append({"name": "Fix C · " + FIX_NAMES["C"], "status": fix_status["C"],
                 "info": _fixinfo(scores.get("C"))})
    rail.append({"name": "Winner → folds 1-4", "status": ("running" if winner and cur["fold"] and cur["fold"]>0 else ("done" if "5-fold" in stage or "DONE" in drv[-400:] else "pending")),
                 "info": (f"winner Fix {winner}" if winner else "after bake-off")})
    rail.append({"name": "Publication eval (external)", "status": ("done" if "DONE" in drv[-400:] else "pending"),
                 "info": "OOF + external + calibration + conformal + fairness"})

    return {
        "ts": time.strftime("%H:%M:%S"),
        "stage": stage, "winner": winner, "scores": scores,
        "current": cur, "epochs": epochs, "gpu": gpu(), "rail": rail,
        "milestones": {"collapsed_baseline": 0.339, "fixA_oof": scores.get("A", {}).get("oof_macro_f1")},
        "fix_names": FIX_NAMES,
        "running": cur.get("live", False),
    }


def _fixinfo(s):
    if not s:
        return ""
    bits = []
    if "oof_macro_f1" in s: bits.append(f"OOF F1 {s['oof_macro_f1']:.3f}")
    if "collapsed_recall" in s: bits.append(f"recall {s['collapsed_recall']:.2f}")
    if "mel_recall" in s: bits.append(f"mel {s['mel_recall']:.2f}")
    return " · ".join(bits)


# ─────────────────────────────────────────────────── history & pipeline ──
RE_DONE = re.compile(r"done; best macro_f1 = ([\d.]+)")
FIX_FULL = {"A": "CE + gentle √-sampler", "B": "Balanced-Softmax", "C": "CB-Focal (no resample)"}

PIPELINE = [
    {"t": "Phase 1 · 9-class base", "d": "5-fold CV macro-F1 0.637 ± 0.009 · AUROC ≈0.95 · accuracy ≈85%", "status": "done"},
    {"t": "Phase 3 v1 · Indian 25-class", "d": "macro-F1 0.339 — MAJORITY-CLASS COLLAPSE (eczema/psoriasis recall ≈0)", "status": "warn"},
    {"t": "Diagnosis", "d": "double-balancing: inverse-freq sampler × cb_focal over-corrected the class prior", "status": "done"},
    {"t": "Phase 3c · rebalancing ablation", "d": "Fix A/B/C single-mechanism · winner chosen on OOF · external test pristine", "status": "running"},
    {"t": "Publication eval", "d": "OOF + external ensemble + calibration + conformal + per-Fitzpatrick fairness", "status": "pending"},
]


def label_for(name: str):
    m = RE_FIXLOG.search(name)
    if m:
        return ("Phase 3c · Ablation", "p3c", f"Fix {m.group(1)} · fold {m.group(2)}", int(m.group(2)), m.group(1))
    m = re.match(r"phase3_fold(\d)_train\.log", name)
    if m:
        return ("Phase 3 v1 · collapsed", "p3v1", f"Fold {m.group(1)}", int(m.group(1)), None)
    m = re.match(r"fold(\d)_train\.log", name)
    if m:
        return ("Phase 1 · 9-class", "p1", f"Fold {m.group(1)}", int(m.group(1)), None)
    return ("Other", "other", name.replace("_train.log", ""), None, None)


def parse_epochs(path: Path):
    txt = read(path).replace("\r", "\n")
    eps = [{"epoch": int(e.group(1)), "loss": float(e.group(2)), "val_loss": float(e.group(3)),
            "acc": float(e.group(4)), "bal_acc": float(e.group(5)), "macroF1": float(e.group(6)),
            "auroc": float(e.group(7))} for e in RE_EPOCH.finditer(txt)]
    best = RE_BEST.findall(txt)
    bestf = float(best[-1]) if best else (max((x["macroF1"] for x in eps), default=0.0))
    donem = RE_DONE.findall(txt)
    return eps, bestf, bool(donem), (float(donem[-1]) if donem else bestf)


def history_runs():
    runs = []
    for p in LOGS.glob("*fold*_train.log"):
        if not p.is_file():
            continue
        phase, group, label, fold, fix = label_for(p.name)
        eps, bestf, done, final = parse_epochs(p)
        if not eps:
            continue
        runs.append({"id": p.name, "phase": phase, "group": group, "label": label,
                     "fix": fix, "fold": fold, "n_epochs": len(eps),
                     "best_f1": round(bestf, 4), "final_f1": round(final, 4),
                     "done": done, "mtime": p.stat().st_mtime})
    runs.sort(key=lambda r: (r["group"], r.get("fix") or "", r["fold"] if r["fold"] is not None else 0))
    return runs


def run_detail(name: str):
    if "/" in name or "\\" in name or ".." in name:
        return {"error": "bad name"}
    p = LOGS / name
    if not p.is_file():
        return {"error": "not found"}
    phase, group, label, fold, fix = label_for(name)
    eps, bestf, done, final = parse_epochs(p)
    return {"id": name, "phase": phase, "label": label, "fix": fix,
            "fix_full": FIX_FULL.get(fix, ""), "epochs": eps, "n_epochs": len(eps),
            "best_f1": round(bestf, 4), "final_f1": round(final, 4), "done": done}


# ─────────────────────────────────────────────────────────── HTML (static) ──
PAGE = r"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MNKAI-III · Live Training</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap" rel=stylesheet>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#04060d;--cyan:#38e0ff;--violet:#8b6dff;--mint:#43f3c7;--good:#34e5a3;--warn:#ffc24b;--bad:#ff647c;
 --ink:#eaf1ff;--mut:#9fb2d6;--dim:#5c6f92;--line:rgba(120,160,230,.16);--glass:rgba(18,28,50,.5)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif;overflow-x:hidden}
h1,h2,.dg{font-family:"Space Grotesk",sans-serif}.mono{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
#bg{position:fixed;inset:0;z-index:-1;background:
 radial-gradient(900px 500px at 85% -5%,rgba(139,109,255,.18),transparent 60%),
 radial-gradient(800px 480px at 5% 0%,rgba(56,224,255,.15),transparent 60%),
 linear-gradient(transparent 97%,rgba(56,224,255,.05) 97%) 0 0/46px 46px,
 linear-gradient(90deg,transparent 97%,rgba(56,224,255,.05) 97%) 0 0/46px 46px,var(--bg);
 animation:drift 40s linear infinite}@keyframes drift{to{background-position:0 0,0 0,0 920px,920px 0,0 0}}
.wrap{max-width:1240px;margin:0 auto;padding:22px}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:12px}.logo{font-size:26px;filter:drop-shadow(0 0 10px var(--cyan))}
.brand h1{margin:0;font-size:22px;letter-spacing:.4px}.brand h1 b{color:var(--cyan)}
.brand p{margin:0;color:var(--mut);font-size:12px}
.pill{display:flex;align-items:center;gap:8px;padding:8px 14px;border:1px solid var(--line);border-radius:999px;font-size:12.5px;color:var(--mut);background:var(--glass);backdrop-filter:blur(12px)}
.pill .dot{width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 10px var(--good);animation:beat 1.6s infinite}
.pill.idle .dot{background:var(--warn);box-shadow:0 0 10px var(--warn)}@keyframes beat{50%{opacity:.4}}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}
.card{background:var(--glass);backdrop-filter:blur(20px) saturate(150%);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 20px 50px -24px rgba(0,0,0,.7);position:relative;overflow:hidden}
.card::before{content:"";position:absolute;inset:0;border-radius:inherit;padding:1px;background:linear-gradient(150deg,rgba(160,200,255,.25),transparent 42%);-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none}
.s12{grid-column:span 12}.s8{grid-column:span 8}.s7{grid-column:span 7}.s6{grid-column:span 6}.s5{grid-column:span 5}.s4{grid-column:span 4}
@media(max-width:900px){.s8,.s7,.s6,.s5,.s4{grid-column:span 12}}
.ct{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--cyan);margin:0 0 14px}
/* now training */
.now{display:flex;gap:24px;align-items:center;flex-wrap:wrap}
.now .ring{--p:0;width:128px;height:128px;border-radius:50%;flex:0 0 auto;display:grid;place-items:center;position:relative;
 background:conic-gradient(var(--cyan) calc(var(--p)*1%),rgba(120,160,230,.12) 0)}
.now .ring::after{content:"";position:absolute;inset:9px;border-radius:50%;background:#070b16}
.now .ring .pv{position:relative;z-index:2;text-align:center}.now .ring .pv b{font-size:30px;font-family:"Space Grotesk"}.now .ring .pv span{display:block;font-size:10px;color:var(--mut)}
.now .meta{flex:1;min-width:260px}
.now .fix{font-size:26px;font-family:"Space Grotesk";font-weight:700}
.now .sub{color:var(--mut);font-size:13px;margin:4px 0 14px}
.bar{height:12px;border-radius:8px;background:rgba(120,160,230,.12);overflow:hidden;margin:6px 0}
.bar i{display:block;height:100%;border-radius:8px;background:linear-gradient(90deg,var(--cyan),var(--violet));transition:width .6s cubic-bezier(.2,.8,.2,1);box-shadow:0 0 16px rgba(56,224,255,.5)}
.kv{display:flex;gap:22px;flex-wrap:wrap;margin-top:14px}
.kv div{font-size:12px;color:var(--dim)}.kv b{display:block;font-size:20px;color:var(--ink);font-family:"Space Grotesk"}
.kv b small{font-size:12px;color:var(--mut)}
/* gpu */
.gauge{text-align:center}.gauge .big{font-size:40px;font-family:"Space Grotesk";font-weight:700;background:linear-gradient(120deg,var(--cyan),var(--violet));-webkit-background-clip:text;background-clip:text;color:transparent}
.gbar{height:8px;border-radius:6px;background:rgba(120,160,230,.12);overflow:hidden;margin-top:10px}.gbar i{display:block;height:100%;background:linear-gradient(90deg,var(--mint),var(--cyan));transition:width .5s}
.gstat{display:flex;justify-content:space-around;margin-top:14px;font-size:12px;color:var(--mut)}.gstat b{display:block;color:var(--ink);font-size:16px;font-family:"Space Grotesk"}
/* rail */
.rail{display:flex;flex-direction:column;gap:10px}
.step{display:flex;align-items:center;gap:14px;padding:12px 14px;border-radius:12px;border:1px solid var(--line);background:rgba(10,16,30,.4);transition:.3s}
.step .ic{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;font-size:13px;flex:0 0 auto;border:1px solid var(--line)}
.step.done .ic{background:rgba(52,229,163,.16);color:var(--good);border-color:rgba(52,229,163,.4)}
.step.running{border-color:var(--cyan);box-shadow:0 0 22px -6px rgba(56,224,255,.5)}
.step.running .ic{background:rgba(56,224,255,.16);color:var(--cyan);border-color:var(--cyan);animation:beat 1.4s infinite}
.step.pending{opacity:.55}.step .nm{font-size:13.5px;font-weight:600}.step .nf{font-size:11.5px;color:var(--mut)}
/* fix compare */
.fixc{display:flex;flex-direction:column;gap:12px}
.fixrow{display:grid;grid-template-columns:64px 1fr 54px;gap:12px;align-items:center;font-size:13px}
.fixrow .fb{height:10px;border-radius:6px;background:rgba(120,160,230,.12);overflow:hidden}.fixrow .fb i{display:block;height:100%;background:linear-gradient(90deg,var(--violet),var(--cyan));transition:width .8s}
.win{color:var(--good);font-weight:700}
canvas{max-height:240px}
.foot{text-align:center;color:var(--dim);font-size:11.5px;margin-top:22px}
.hidden{display:none!important}
.tabs{display:flex;gap:10px;margin-bottom:18px}
.tb{padding:10px 18px;border-radius:12px;border:1px solid var(--line);background:var(--glass);color:var(--mut);
 cursor:pointer;font-size:13.5px;font-weight:600;font-family:Inter;transition:.25s}
.tb:hover{color:var(--ink);border-color:rgba(120,160,230,.3)}
.tb.active{color:#04121f;background:linear-gradient(90deg,var(--cyan),var(--violet));border-color:transparent;box-shadow:0 8px 22px -8px rgba(56,224,255,.5)}
/* pipeline timeline */
.tl{position:relative;padding-left:26px;margin:4px 0 8px}
.tl::before{content:"";position:absolute;left:8px;top:6px;bottom:6px;width:2px;background:linear-gradient(var(--cyan),var(--violet))}
.tlrow{position:relative;padding:10px 0}
.tlrow::before{content:"";position:absolute;left:-22px;top:14px;width:12px;height:12px;border-radius:50%;background:#070b16;border:2px solid var(--dim)}
.tlrow.done::before{border-color:var(--good);background:var(--good);box-shadow:0 0 10px var(--good)}
.tlrow.warn::before{border-color:var(--warn);background:var(--warn);box-shadow:0 0 10px var(--warn)}
.tlrow.running::before{border-color:var(--cyan);background:var(--cyan);box-shadow:0 0 12px var(--cyan);animation:beat 1.4s infinite}
.tlrow .tt{font-size:14px;font-weight:600;font-family:"Space Grotesk"}.tlrow .td{font-size:12.5px;color:var(--mut)}
/* history list */
.hgroup{margin-bottom:18px}.hgroup h3{font-size:13px;color:var(--cyan);margin:0 0 10px;letter-spacing:.08em;text-transform:uppercase}
.hrow{display:grid;grid-template-columns:1fr auto auto;gap:14px;align-items:center;padding:12px 16px;border:1px solid var(--line);
 border-radius:12px;background:rgba(10,16,30,.4);cursor:pointer;transition:.22s;margin-bottom:8px}
.hrow:hover{border-color:var(--cyan);transform:translateX(3px);box-shadow:0 0 20px -8px rgba(56,224,255,.5)}
.hrow.sel{border-color:var(--cyan);background:rgba(56,224,255,.08)}
.hrow .hl{font-size:14px;font-weight:600}.hrow .hm{font-size:11.5px;color:var(--mut)}
.hrow .hf{font-family:"Space Grotesk";font-size:18px;font-weight:700;background:linear-gradient(120deg,var(--cyan),var(--violet));-webkit-background-clip:text;background-clip:text;color:transparent}
.hrow .hs{font-size:11px;padding:3px 9px;border-radius:7px;border:1px solid var(--line);color:var(--mut)}
.hrow .hs.done{color:var(--good);border-color:rgba(52,229,163,.4)}.hrow .hs.warn{color:var(--warn);border-color:rgba(255,194,75,.4)}
.detail-empty{color:var(--dim);font-size:13px;text-align:center;padding:40px 0}
.brainwrap{position:relative;height:clamp(560px,82vh,1200px);background:radial-gradient(1200px 600px at 50% 42%,rgba(56,224,255,.07),transparent 70%)}
.brainwrap:fullscreen{height:100vh;width:100vw;background:#04060d}
#brainCanvas{width:100%;height:100%;display:block}
.brain-fs{position:absolute;top:18px;right:18px;width:42px;height:42px;border-radius:12px;border:1px solid var(--line);
 background:rgba(7,11,22,.6);backdrop-filter:blur(10px);color:var(--ink);font-size:18px;cursor:pointer;transition:.2s;z-index:3}
.brain-fs:hover{border-color:var(--cyan);background:rgba(56,224,255,.14);transform:scale(1.06)}
.brainhud{position:absolute;top:18px;left:18px;background:rgba(7,11,22,.55);backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:12px;padding:13px 15px;font-size:11.5px;min-width:240px}
.bh-row{display:flex;gap:16px;justify-content:space-between;padding:3px 0}.bh-k{color:var(--dim);letter-spacing:.08em}.bh-v{color:var(--ink);font-family:"JetBrains Mono";text-align:right}
.brainlegend{position:absolute;bottom:50px;left:18px;display:flex;gap:18px;font-size:11.5px;color:var(--mut)}
.brainlegend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:middle}
.braincap{position:absolute;bottom:14px;left:18px;right:18px;font-size:11px;color:var(--dim);max-width:780px;line-height:1.4}
@media(max-width:900px){.brainwrap{height:440px}.braincap{display:none}.brainlegend{bottom:14px}}
</style></head><body><div id=bg></div>
<div class=wrap>
 <div class=top>
  <div class=brand><span class=logo>🧬</span><div><h1>MNKAI<b>-III</b> · Live Training</h1><p id=stage>connecting…</p></div></div>
  <div class="pill" id=hb><span class=dot></span><span id=hbtxt>live</span></div>
 </div>
 <div class=tabs>
   <button class="tb active" data-v=live>● Live training</button>
   <button class="tb" data-v=history>◷ History &amp; Pipeline</button>
   <button class="tb" data-v=brain>🧠 Neural Brain</button>
 </div>
 <div id=view-live>
 <div class=grid>
  <div class="card s8"><p class=ct>● Now training</p><div class=now>
    <div class=ring id=ring><div class=pv><b id=rpct>0%</b><span>epoch</span></div></div>
    <div class=meta>
      <div class=fix id=fix>—</div><div class=sub id=fixsub></div>
      <div class=bar><i id=epbar style=width:0></i></div>
      <div class="mono" id=stepinfo style="font-size:12px;color:var(--mut)"></div>
      <div class=kv>
        <div>Best macro-F1<b id=bestf1>—</b></div>
        <div>Loss<b id=loss>—</b></div>
        <div>Val macro-F1<b id=vf1>—</b></div>
        <div>Speed<b id=its>—<small> it/s</small></b></div>
      </div>
    </div></div></div>
  <div class="card s4"><p class=ct>● GPU</p><div class=gauge>
    <div class=big id=gutil>0%</div><div class=gbar><i id=gbar style=width:0></i></div>
    <div class=gstat><div>VRAM<b id=gmem>—</b></div><div>Temp<b id=gtemp>—</b></div></div></div></div>

  <div class="card s8"><p class=ct>● Macro-F1 / AUROC · per epoch (live)</p><canvas id=cf1></canvas></div>
  <div class="card s4"><p class=ct>● Pipeline</p><div class=rail id=rail></div></div>

  <div class="card s8"><p class=ct>● Loss · train vs val (live)</p><canvas id=closs></canvas></div>
  <div class="card s4"><p class=ct>● Fix bake-off</p><div class=fixc id=fixc></div>
    <p style="color:var(--dim);font-size:11px;margin-top:12px">Baseline (collapsed): <b class=mono>0.339</b> → fixes recover common-class recall.</p></div>
 </div>
 </div><!-- /view-live -->

 <div id=view-history class=hidden>
  <div class=grid>
   <div class="card s5"><p class=ct>● Whole pipeline · start → now</p><div class=tl id=tl></div></div>
   <div class="card s7"><p class=ct>● Completed runs · click to inspect</p><div id=hlist></div></div>
   <div class="card s12" id=detail-card><p class=ct>● Run detail</p>
     <div id=detail-head class=detail-empty>Select a run on the left to see its full epoch-by-epoch history.</div>
     <div id=detail-charts class=hidden>
       <div class=grid style="gap:18px">
         <div class=s6><p class=ct style="color:var(--mut)">macro-F1 / AUROC</p><canvas id=dcf1></canvas></div>
         <div class=s6><p class=ct style="color:var(--mut)">loss · train vs val</p><canvas id=dcloss></canvas></div>
       </div>
     </div>
   </div>
  </div>
 </div><!-- /view-history -->

 <div id=view-brain class=hidden>
  <div class=card style="padding:0;overflow:hidden">
   <div class=brainwrap>
    <canvas id=brainCanvas></canvas>
    <button class=brain-fs id=brain-fs title="Fullscreen (Esc to exit)">⛶</button>
    <div class=brainhud>
     <div class=bh-row><span class=bh-k>MODEL</span><span class=bh-v>DINOv2 ViT-B/14 · 86.5M params · 12 blocks</span></div>
     <div class=bh-row><span class=bh-k>STATUS</span><span class="bh-v" id=bh-mode>—</span></div>
     <div class=bh-row><span class=bh-k>EPOCH</span><span class="bh-v" id=bh-ep>—</span></div>
     <div class=bh-row><span class=bh-k>LOSS</span><span class="bh-v" id=bh-loss>—</span></div>
     <div class=bh-row><span class=bh-k>MACRO-F1</span><span class="bh-v" id=bh-f1>—</span></div>
    </div>
    <div class=brainlegend><span><i style="background:#38e0ff"></i>forward signal</span><span><i style="background:#ff647c"></i>gradient (backprop)</span><span><i style="background:#43f3c7"></i>active neuron</span></div>
    <div class=braincap>Architecture is real (patches → 12 transformer blocks → 25-class head). Pulse rate, neuron activity & the gradient wave are driven by the live loss / epoch / GPU — a data-driven representation, not a literal render of all 86.5M weights.</div>
   </div>
  </div>
 </div><!-- /view-brain -->

 <div class=foot>auto-refresh 2.5s · winner chosen on held-out OOF validation · external test kept pristine</div>
</div>
<script>
let f1c,lossc;
function mkF1(){const c=document.getElementById('cf1');f1c=new Chart(c,{type:'line',data:{labels:[],datasets:[
 {label:'macro-F1',data:[],borderColor:'#38e0ff',backgroundColor:'rgba(56,224,255,.12)',fill:true,tension:.35,pointRadius:0,borderWidth:2.5},
 {label:'AUROC',data:[],borderColor:'#43f3c7',tension:.35,pointRadius:0,borderWidth:1.5,borderDash:[5,4]}]},
 options:{animation:{duration:500},plugins:{legend:{labels:{color:'#9fb2d6',boxWidth:10,font:{size:11}}}},
 scales:{y:{min:0,max:1,grid:{color:'rgba(120,160,230,.1)'},ticks:{color:'#5c6f92'}},x:{grid:{display:false},ticks:{color:'#5c6f92'}}}}});}
function mkLoss(){const c=document.getElementById('closs');lossc=new Chart(c,{type:'line',data:{labels:[],datasets:[
 {label:'train loss',data:[],borderColor:'#8b6dff',backgroundColor:'rgba(139,109,255,.12)',fill:true,tension:.35,pointRadius:0,borderWidth:2.5},
 {label:'val loss',data:[],borderColor:'#ffc24b',tension:.35,pointRadius:0,borderWidth:1.5,borderDash:[5,4]}]},
 options:{animation:{duration:500},plugins:{legend:{labels:{color:'#9fb2d6',boxWidth:10,font:{size:11}}}},
 scales:{y:{grid:{color:'rgba(120,160,230,.1)'},ticks:{color:'#5c6f92'}},x:{grid:{display:false},ticks:{color:'#5c6f92'}}}}});}
mkF1();mkLoss();
const ICON={done:'✓',running:'▶',pending:'○'};
async function tick(){
 let s;try{s=await (await fetch('/api/state')).json();}catch(e){return;}
 window.LASTSTATE=s;
 document.getElementById('stage').textContent=s.stage;
 const hb=document.getElementById('hb');hb.classList.toggle('idle',!s.running);document.getElementById('hbtxt').textContent=s.running?'training live':'idle';
 const c=s.current;
 document.getElementById('fix').textContent=c.fix?('Fix '+c.fix+(c.fold!=null?' · fold '+c.fold:'')):'—';
 document.getElementById('fixsub').textContent=c.fix?(s.fix_names[c.fix]||'')+'  ·  epoch '+c.epoch+'/'+c.total_epochs:'waiting…';
 document.getElementById('rpct').textContent=(c.pct||0)+'%';
 document.getElementById('ring').style.setProperty('--p',c.pct||0);
 document.getElementById('epbar').style.width=(c.pct||0)+'%';
 document.getElementById('stepinfo').textContent='step '+(c.step||0)+' / '+(c.total_steps||0);
 document.getElementById('bestf1').textContent=(c.best_f1!=null?c.best_f1.toFixed(4):'—');
 document.getElementById('loss').textContent=(c.loss!=null?c.loss.toFixed(3):'—');
 document.getElementById('its').innerHTML=(c.it_s!=null?c.it_s.toFixed(2):'—')+'<small> it/s</small>';
 const ep=s.epochs||[];document.getElementById('vf1').textContent=ep.length?ep[ep.length-1].macroF1.toFixed(3):'—';
 // gpu
 const g=s.gpu;document.getElementById('gutil').textContent=g.util+'%';document.getElementById('gbar').style.width=g.util+'%';
 document.getElementById('gmem').innerHTML=(g.mem/1024).toFixed(1)+' / '+(g.mem_total/1024).toFixed(0)+' GB';
 document.getElementById('gtemp').textContent=g.temp+'°C';
 // charts
 const labels=ep.map(e=>e.epoch);
 f1c.data.labels=labels;f1c.data.datasets[0].data=ep.map(e=>e.macroF1);f1c.data.datasets[1].data=ep.map(e=>e.auroc);f1c.update();
 lossc.data.labels=labels;lossc.data.datasets[0].data=ep.map(e=>e.loss);lossc.data.datasets[1].data=ep.map(e=>e.val_loss);lossc.update();
 // rail
 document.getElementById('rail').innerHTML=s.rail.map(r=>`<div class="step ${r.status}"><div class=ic>${ICON[r.status]}</div><div><div class=nm>${r.name}</div><div class=nf>${r.info||r.status}</div></div></div>`).join('');
 // fix compare
 const sc=s.scores||{};const keys=['A','B','C'];const vals=keys.map(k=>sc[k]&&sc[k].oof_macro_f1||0);const mx=Math.max(...vals,0.6);
 document.getElementById('fixc').innerHTML=keys.map(k=>{const v=sc[k]&&sc[k].oof_macro_f1;const wn=s.winner===k;
   return `<div class=fixrow><span class="${wn?'win':''}">Fix ${k}${wn?' ★':''}</span><div class=fb><i style="width:${v?Math.min(100,v/mx*100):0}%"></i></div><span class="mono ${wn?'win':''}">${v?v.toFixed(3):'—'}</span></div>`;}).join('');
}
tick();setInterval(tick,2500);

// ---- tabs ----
let histLoaded=false;
document.querySelectorAll('.tb').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tb').forEach(x=>x.classList.remove('active'));b.classList.add('active');
 const v=b.dataset.v;
 document.getElementById('view-live').classList.toggle('hidden',v!=='live');
 document.getElementById('view-history').classList.toggle('hidden',v!=='history');
 document.getElementById('view-brain').classList.toggle('hidden',v!=='brain');
 if(v==='history'){loadHistory();}
 if(v==='brain'){setTimeout(brainStart,30);}else{brainStop();}
});

const GICON={done:'done',warn:'collapse',running:'running',pending:'pending'};
async function loadHistory(){
 let h;try{h=await (await fetch('/api/history')).json();}catch(e){return;}
 // pipeline timeline
 document.getElementById('tl').innerHTML=h.pipeline.map(p=>
   `<div class="tlrow ${p.status}"><div class=tt>${p.t}</div><div class=td>${p.d}</div></div>`).join('');
 // group runs
 const order=['p1','p3v1','p3c','other'];const names={p1:'Phase 1 · 9-class base',p3v1:'Phase 3 v1 · collapsed',p3c:'Phase 3c · ablation',other:'Other'};
 const groups={};h.runs.forEach(r=>{(groups[r.group]=groups[r.group]||[]).push(r);});
 let html='';order.forEach(g=>{ if(!groups[g])return;
   html+=`<div class=hgroup><h3>${names[g]}</h3>`+groups[g].map(r=>{
     const st=g==='p3v1'?'warn':(r.done?'done':'running');
     return `<div class=hrow data-id="${r.id}"><div><div class=hl>${r.label}</div><div class=hm>${r.n_epochs} epochs${r.done?' · finished':' · partial'}</div></div>`+
       `<div class=hf>${r.final_f1.toFixed(3)}</div><div class="hs ${st}">${st==='warn'?'collapsed':(r.done?'done':'live')}</div></div>`;
   }).join('')+`</div>`;});
 document.getElementById('hlist').innerHTML=html||'<div class=detail-empty>No completed runs yet.</div>';
 document.querySelectorAll('.hrow').forEach(row=>row.onclick=()=>{
   document.querySelectorAll('.hrow').forEach(x=>x.classList.remove('sel'));row.classList.add('sel');loadRun(row.dataset.id);});
 histLoaded=true;
}
let dcf1,dcloss;
async function loadRun(id){
 let r;try{r=await (await fetch('/api/run?log='+encodeURIComponent(id))).json();}catch(e){return;}
 if(r.error){document.getElementById('detail-head').textContent='could not load run';return;}
 const ep=r.epochs||[];
 document.getElementById('detail-head').classList.add('hidden');
 document.getElementById('detail-charts').classList.remove('hidden');
 document.querySelector('#detail-card .ct').innerHTML='● Run detail — <b style="color:var(--ink)">'+r.label+'</b> '+
   (r.fix_full?'· '+r.fix_full+' ':'')+'· final macro-F1 <b style="color:var(--cyan)">'+r.final_f1.toFixed(4)+'</b> · '+ep.length+' epochs';
 const labels=ep.map(e=>e.epoch);
 const o={animation:{duration:400},plugins:{legend:{labels:{color:'#9fb2d6',boxWidth:10,font:{size:11}}}},
   scales:{y:{grid:{color:'rgba(120,160,230,.1)'},ticks:{color:'#5c6f92'}},x:{grid:{display:false},ticks:{color:'#5c6f92'}}}};
 if(dcf1)dcf1.destroy();if(dcloss)dcloss.destroy();
 dcf1=new Chart(document.getElementById('dcf1'),{type:'line',data:{labels,datasets:[
   {label:'macro-F1',data:ep.map(e=>e.macroF1),borderColor:'#38e0ff',backgroundColor:'rgba(56,224,255,.12)',fill:true,tension:.35,pointRadius:0,borderWidth:2.5},
   {label:'AUROC',data:ep.map(e=>e.auroc),borderColor:'#43f3c7',tension:.35,pointRadius:0,borderWidth:1.5,borderDash:[5,4]}]},
   options:{...o,scales:{...o.scales,y:{...o.scales.y,min:0,max:1}}}});
 dcloss=new Chart(document.getElementById('dcloss'),{type:'line',data:{labels,datasets:[
   {label:'train loss',data:ep.map(e=>e.loss),borderColor:'#8b6dff',backgroundColor:'rgba(139,109,255,.12)',fill:true,tension:.35,pointRadius:0,borderWidth:2.5},
   {label:'val loss',data:ep.map(e=>e.val_loss),borderColor:'#ffc24b',tension:.35,pointRadius:0,borderWidth:1.5,borderDash:[5,4]}]},options:o});
}

// ---------- Neural Brain (data-driven, alive) ----------
const B={cv:null,ctx:null,W:0,H:0,DPR:1,layers:[],nodes:[],conns:[],parts:[],raf:0,t:0,running:false,lastEp:0,ripple:-999};
function rnd(s){let x=Math.sin(s*99.71)*43758.545;return x-Math.floor(x);}
function brainResize(){if(!B.cv)return;B.DPR=Math.min(devicePixelRatio||1,2);
 const r=B.cv.getBoundingClientRect();B.W=B.cv.width=Math.max(300,r.width*B.DPR);B.H=B.cv.height=Math.max(300,r.height*B.DPR);brainBuild();}
function brainBuild(){
 const counts=[18];for(let i=0;i<12;i++)counts.push(9+Math.round(rnd(i+1)*5));counts.push(16);
 const NL=counts.length,padX=84*B.DPR,padY=60*B.DPR;B.nodes=[];B.layers=[];
 for(let l=0;l<NL;l++){const x=padX+(B.W-2*padX)*(l/(NL-1));const n=counts[l];const layer=[];
  for(let i=0;i<n;i++){const y=padY+(B.H-2*padY)*((i+0.5)/n)+(rnd(l*50+i)*2-1)*11*B.DPR;
   const node={hx:x,hy:y,x,y,l,phase:rnd(l*13+i)*6.28,fa:rnd(l*9+i)*6.28,fire:0};B.nodes.push(node);layer.push(node);}B.layers.push(layer);}
 B.conns=[];
 for(let l=0;l<NL-1;l++){const a=B.layers[l],b=B.layers[l+1];
  a.forEach((na,ai)=>{const k=2+Math.round(rnd(l*100+ai)*1.7);
   for(let c=0;c<k;c++){const bi=Math.floor(rnd(l*200+ai*7+c)*b.length);
    B.conns.push({a:na,b:b[bi],w:rnd(l*300+ai*11+c*3)*0.8+0.12,drift:rnd(ai+c+l)*6.28,cx:0,cy:0});}});}
 B.parts=[];
}
function bez(c,t){const u=1-t;return [u*u*c.a.x+2*u*t*c.cx+t*t*c.b.x, u*u*c.a.y+2*u*t*c.cy+t*t*c.b.y];}
function spawnP(back){const c=B.conns[Math.floor(Math.random()*B.conns.length)];if(c)B.parts.push({c,p:0,back,sp:0.6+Math.random()*0.9});}
function brainLoop(){
 if(!B.running)return;const ctx=B.ctx;const s=window.LASTSTATE||{};const cur=s.current||{};
 const loss=cur.loss!=null?cur.loss:1.0,its=cur.it_s||2;
 const act=Math.min(1,Math.max(.18,loss/3.0));
 B.t+=0.016;const D=B.DPR;
 const epc=(s.epochs||[]).length;
 if(epc>B.lastEp){B.lastEp=epc;B.ripple=B.W+100*D;B.flash=1;}
 // float nodes + decay fire
 for(const n of B.nodes){n.x=n.hx+Math.sin(B.t*0.6+n.fa)*5*D;n.y=n.hy+Math.cos(B.t*0.5+n.fa)*5*D;n.fire*=0.90;}
 // trail fade (instead of hard clear → motion blur)
 ctx.globalCompositeOperation='source-over';
 ctx.fillStyle='rgba(4,6,13,0.30)';ctx.fillRect(0,0,B.W,B.H);
 ctx.globalCompositeOperation='lighter';
 // curved synapses
 for(const c of B.conns){const dx=c.b.x-c.a.x,dy=c.b.y-c.a.y,len=Math.hypot(dx,dy)||1;
  const off=Math.sin(c.drift+B.t*0.25)*26*D;
  c.cx=(c.a.x+c.b.x)/2+(-dy/len)*off;c.cy=(c.a.y+c.b.y)/2+(dx/len)*off;
  const w=Math.max(0.03,c.w+Math.sin(B.t*0.5+c.drift)*0.13);
  const g=ctx.createLinearGradient(c.a.x,c.a.y,c.b.x,c.b.y);
  g.addColorStop(0,`rgba(70,150,255,${w*0.20})`);g.addColorStop(1,`rgba(150,110,255,${w*0.17})`);
  ctx.strokeStyle=g;ctx.lineWidth=w*1.5*D;ctx.beginPath();ctx.moveTo(c.a.x,c.a.y);ctx.quadraticCurveTo(c.cx,c.cy,c.b.x,c.b.y);ctx.stroke();}
 // backprop wave
 if(B.ripple>-100*D){const x=B.ripple;const g=ctx.createLinearGradient(x-80*D,0,x+80*D,0);
  g.addColorStop(0,'rgba(255,100,124,0)');g.addColorStop(.5,`rgba(255,100,124,${0.16+0.12*(B.flash||0)})`);g.addColorStop(1,'rgba(255,100,124,0)');
  ctx.fillStyle=g;ctx.fillRect(x-80*D,0,160*D,B.H);B.ripple-=12*D;}
 if(B.flash)B.flash*=0.92;
 // nodes (with fire + mouse)
 for(const n of B.nodes){const pulse=0.5+0.5*Math.sin(B.t*(2+its*0.3)+n.phase);
  const near=Math.abs(n.x-B.ripple)<76*D?1:0;
  let mb=0;if(B.mouse){const md=Math.hypot(n.x-B.mouse.x,n.y-B.mouse.y);if(md<130*D)mb=1-md/(130*D);}
  const fire=n.fire;if(near)n.fire=Math.max(n.fire,0.7);
  const a=Math.min(1,0.32+0.5*pulse*act+near*0.5+fire*0.6+mb*0.55);
  const r=(2.2+pulse*2.3*act+near*2.6+fire*3.4+mb*2.4)*D;
  const col=near?'255,100,124':(fire>0.3?'130,235,255':(n.l===0?'67,243,199':(n.l===B.layers.length-1?'160,120,255':'56,210,255')));
  ctx.beginPath();ctx.arc(n.x,n.y,r,0,6.28);ctx.fillStyle=`rgba(${col},${a})`;
  ctx.shadowColor=`rgba(${col},.9)`;ctx.shadowBlur=(8+fire*16)*D*pulse;ctx.fill();}
 ctx.shadowBlur=0;
 // particles + trails + firing on arrival
 if(Math.random()<act*0.7+0.25)spawnP(false);
 if(B.ripple>0&&Math.random()<0.6)spawnP(true);
 for(let i=B.parts.length-1;i>=0;i--){const pt=B.parts[i];const pp=pt.p;pt.p+=0.014*pt.sp;
  if(pt.p>=1){const tgt=pt.back?pt.c.a:pt.c.b;tgt.fire=Math.min(1,tgt.fire+0.85);B.parts.splice(i,1);continue;}
  const t=pt.back?1-pt.p:pt.p,tp=pt.back?1-pp:pp;const P=bez(pt.c,t),Q=bez(pt.c,tp);
  const cs=pt.back?'255,100,124':'130,225,255';
  ctx.strokeStyle=`rgba(${cs},.45)`;ctx.lineWidth=2*D;ctx.beginPath();ctx.moveTo(Q[0],Q[1]);ctx.lineTo(P[0],P[1]);ctx.stroke();
  ctx.beginPath();ctx.arc(P[0],P[1],2.4*D,0,6.28);ctx.fillStyle=`rgba(${cs},.98)`;
  ctx.shadowColor=`rgba(${cs},.9)`;ctx.shadowBlur=13*D;ctx.fill();}
 ctx.shadowBlur=0;ctx.globalCompositeOperation='source-over';
 document.getElementById('bh-mode').textContent=B.ripple>0?'◀ backprop · weight update':'▶ forward pass';
 document.getElementById('bh-ep').textContent=cur.epoch?cur.epoch+' / '+cur.total_epochs:'idle';
 document.getElementById('bh-loss').textContent=cur.loss!=null?cur.loss.toFixed(3):'—';
 const ep=s.epochs||[];document.getElementById('bh-f1').textContent=ep.length?ep[ep.length-1].macroF1.toFixed(3):'—';
 B.raf=requestAnimationFrame(brainLoop);
}
function brainStart(){if(B.running)return;B.cv=document.getElementById('brainCanvas');if(!B.cv)return;B.ctx=B.cv.getContext('2d');
 if(!B._wired){
  B.cv.addEventListener('mousemove',e=>{const r=B.cv.getBoundingClientRect();B.mouse={x:(e.clientX-r.left)*B.DPR,y:(e.clientY-r.top)*B.DPR};});
  B.cv.addEventListener('mouseleave',()=>{B.mouse=null;});
  document.getElementById('brain-fs').onclick=()=>{const w=document.querySelector('.brainwrap');
   if(!document.fullscreenElement){w.requestFullscreen&&w.requestFullscreen();}else{document.exitFullscreen&&document.exitFullscreen();}};
  document.addEventListener('fullscreenchange',()=>{if(B.running)setTimeout(brainResize,80);});
  B._wired=true;}
 brainResize();B.running=true;B.lastEp=((window.LASTSTATE||{}).epochs||[]).length;brainLoop();}
function brainStop(){B.running=false;cancelAnimationFrame(B.raf);}
addEventListener('resize',()=>{if(B.running)brainResize();});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(state()).encode(); ct = "application/json"
        elif self.path.startswith("/api/history"):
            body = json.dumps({"runs": history_runs(), "pipeline": PIPELINE}).encode(); ct = "application/json"
        elif self.path.startswith("/api/run"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            body = json.dumps(run_detail(q.get("log", [""])[0])).encode(); ct = "application/json"
        else:
            body = PAGE.encode("utf-8"); ct = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    print(f"[dash] live training dashboard -> http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
