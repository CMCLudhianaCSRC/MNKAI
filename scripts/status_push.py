"""Update the live overnight status file (logs/live_status.json) that the dashboard reads.

Examples:
  python scripts/status_push.py --log "Started rigor batch" --step "Bootstrap CIs on 25-class" --mood working
  python scripts/status_push.py --stage rigor=running
  python scripts/status_push.py --stage rigor=done --stage-note rigor="25-class 0.573 [0.55-0.59]"
  python scripts/status_push.py --result "25-class macro-F1=0.573|deduped external"
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path

STATUS = Path(__file__).resolve().parent.parent / "logs" / "live_status.json"


def load():
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"pipeline": [], "tasks": [], "results": [], "log": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step")
    ap.add_argument("--detail")
    ap.add_argument("--next")
    ap.add_argument("--mood")
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--stage", action="append", default=[], help="id=status (done|running|pending|error)")
    ap.add_argument("--stage-note", action="append", default=[], help="id=note text")
    ap.add_argument("--task", action="append", default=[], help="name=status|detail")
    ap.add_argument("--result", action="append", default=[], help="metric=value|note")
    a = ap.parse_args()

    s = load()
    now = datetime.now()
    s["updated_at"] = now.astimezone().isoformat()
    if a.step is not None:
        s["current_step"] = a.step
    if a.detail is not None:
        s["current_detail"] = a.detail
    if a.next is not None:
        s["next"] = a.next
    if a.mood is not None:
        s["mood"] = a.mood

    for entry in a.log:
        s.setdefault("log", []).insert(0, {"t": now.strftime("%H:%M"), "msg": entry})
    s["log"] = s.get("log", [])[:40]

    for kv in a.stage:
        if "=" in kv:
            sid, st = kv.split("=", 1)
            for p in s.get("pipeline", []):
                if p.get("id") == sid:
                    p["status"] = st
    for kv in getattr(a, "stage_note"):
        if "=" in kv:
            sid, note = kv.split("=", 1)
            for p in s.get("pipeline", []):
                if p.get("id") == sid:
                    p["note"] = note

    for kv in a.task:
        if "=" in kv:
            name, rest = kv.split("=", 1)
            st, _, detail = rest.partition("|")
            found = False
            for t in s.setdefault("tasks", []):
                if t.get("name") == name:
                    t["status"] = st
                    if detail:
                        t["detail"] = detail
                    found = True
            if not found:
                s["tasks"].append({"name": name, "status": st, "detail": detail})

    for kv in a.result:
        if "=" in kv:
            metric, rest = kv.split("=", 1)
            value, _, note = rest.partition("|")
            found = False
            for r in s.setdefault("results", []):
                if r.get("metric") == metric:
                    r["value"] = value
                    if note:
                        r["note"] = note
                    found = True
            if not found:
                s["results"].append({"metric": metric, "value": value, "note": note})

    STATUS.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"status updated @ {s['updated_at']}")


if __name__ == "__main__":
    main()
