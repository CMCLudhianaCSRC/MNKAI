"""LLM history layer (Milestone-1) — adaptive clinical interview + Bayesian fusion.

Provider-agnostic (default: Gemini 2.5-flash). The image model stays the anchor:
the LLM only (1) proposes the most *discriminating* questions for the current
differential, and (2) turns the answers into per-class log-likelihood-ratios that
we combine with the image prior by Bayes. The LLM never diagnoses on its own and
never overrides a cancer flag.

    key: secrets/gemini_key.txt  or  env GEMINI_API_KEY
    self-test:  PYTHONPATH=src .venv/Scripts/python.exe demo_app/llm.py
"""
from __future__ import annotations
import os, json, math, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def _key() -> str | None:
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k.strip()
    f = ROOT / "secrets" / "gemini_key.txt"
    return f.read_text().strip() if f.is_file() else None


def available() -> bool:
    return _key() is not None


def _gemini(prompt: str, system: str | None = None, schema: dict | None = None,
            temperature: float = 0.2, timeout: int = 40, image_b64: str | None = None):
    key = _key()
    if not key:
        raise RuntimeError("no Gemini key (secrets/gemini_key.txt or GEMINI_API_KEY)")
    gen = {"temperature": temperature}
    if schema is not None:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = schema
    parts = []
    if image_b64:                                          # photo-grounded: the model SEES the lesion
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    parts.append({"text": prompt})
    body = {"contents": [{"parts": parts}], "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    req = urllib.request.Request(_ENDPOINT.format(m=MODEL), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    txt = r["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(txt) if schema is not None else txt


SYSTEM = ("You are a careful dermatology decision-support assistant. You do NOT diagnose and you "
          "NEVER override a possible cancer. You help by asking the few most discriminating history "
          "questions and by weighing the answers as evidence. Be concise, safe, and defer to a "
          "clinician. If red-flag features appear, always recommend in-person review.")

_Q_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "object",
    "properties": {"q": {"type": "string"}, "why": {"type": "string"}}, "required": ["q", "why"]}}},
    "required": ["questions"]}


def should_interview(conformal_set_size: float, top_margin: float, k: int = 4, tau: float = 0.15) -> bool:
    """Confidence gate: only interview when the image classifier is uncertain
    (large conformal prediction set OR small top-1/top-2 margin)."""
    return (conformal_set_size is not None and conformal_set_size > k) or \
           (top_margin is not None and top_margin < tau)


def propose_questions(differential: list[dict], meta: dict | None = None, k: int = 3,
                      image_b64: str | None = None) -> list[dict]:
    """differential: [{'name':..,'prob':..}] (image model top-k). If image_b64 is given the
    model SEES the lesion and asks PHOTO-GROUNDED questions. Returns k targeted questions."""
    dif = ", ".join(f"{d['name']} ({d['prob']*100:.0f}%)" for d in differential[:6])
    m = ", ".join(f"{kk}={vv}" for kk, vv in (meta or {}).items() if vv) or "none"
    seen = ("You are shown the actual lesion photo. Look at it (morphology, colour, scale, distribution) "
            "AND the image model's differential, then ask questions whose answers — combined with what you "
            "see — best separate the likely diagnoses. Reference visible features when useful "
            "(e.g. 'the border looks irregular — has it changed shape?').") if image_b64 else \
           "You are given the image model's differential (you cannot see the photo)."
    prompt = (f"{seen}\nDifferential: {dif}. Patient metadata: {m}.\n"
              f"Ask the {k} MOST discriminating history questions (duration, itch, triggers/exposures, new "
              f"medications, spread, systemic symptoms, change over time, prior similar). One short question "
              f"each + a brief reason. Prioritise questions that change management or flag danger.")
    out = _gemini(prompt, SYSTEM, _Q_SCHEMA, temperature=0.3, image_b64=image_b64)
    return out.get("questions", [])[:k]


_F_SCHEMA = {"type": "object", "properties": {
    "log_lr": {"type": "array", "description": "per-condition log-likelihood-ratio in [-2,2] from history",
               "items": {"type": "object", "properties": {"condition": {"type": "string"},
                         "value": {"type": "number"}}, "required": ["condition", "value"]}},
    "red_flags": {"type": "array", "items": {"type": "string"}},
    "explanation": {"type": "string"}},
    "required": ["log_lr", "explanation"]}


def fuse_history(differential: list[dict], qa: list[dict], image_probs: dict) -> dict:
    """qa: [{'q':..,'a':..}]. Returns {posterior:{class:prob}, red_flags, explanation, questions_used}."""
    names = [d["name"] for d in differential[:8]]
    qat = "; ".join(f"Q:{x['q']} A:{x['a']}" for x in qa if x.get("a"))
    dif = ", ".join(f"{d['name']} ({d['prob']*100:.0f}%)" for d in differential[:8])
    prompt = (f"Image-model differential: {dif}.\nPatient answered: {qat}.\n"
              f"For EACH of these conditions {names}, give a log-likelihood-ratio in [-2, 2] for how "
              f"much the HISTORY (not the image) supports it: positive = supported, negative = argued "
              f"against, 0 = neutral. Also list any red-flag features and a one-line explanation. "
              f"Do not restate the image probabilities — judge only the history evidence.")
    out = _gemini(prompt, SYSTEM, _F_SCHEMA, temperature=0.1)
    llr = {str(x["condition"]): float(x["value"]) for x in out.get("log_lr", []) if "condition" in x}
    # Bayesian update: posterior_logit = log(prior) + history logLR
    post = {}
    for c, p in image_probs.items():
        post[c] = math.log(max(p, 1e-6)) + float(llr.get(c, 0.0))
    mx = max(post.values())
    exp = {c: math.exp(v - mx) for c, v in post.items()}
    z = sum(exp.values()) or 1.0
    posterior = {c: round(v / z, 4) for c, v in sorted(exp.items(), key=lambda kv: -kv[1])}
    return {"posterior": posterior, "log_lr": {k: round(float(v), 2) for k, v in llr.items()},
            "red_flags": out.get("red_flags", []), "explanation": out.get("explanation", "")}


if __name__ == "__main__":
    if not available():
        print("no key"); raise SystemExit
    diff = [{"name": "eczema", "prob": 0.34}, {"name": "contact_dermatitis", "prob": 0.28},
            {"name": "psoriasis", "prob": 0.20}, {"name": "tinea", "prob": 0.10}]
    img = {"eczema": 0.34, "contact_dermatitis": 0.28, "psoriasis": 0.20, "tinea": 0.10}
    print("== questions =="); qs = propose_questions(diff, {"age": 30, "site": "hands"})
    for q in qs: print("  -", q["q"], " |", q["why"])
    ans = [{"q": qs[0]["q"], "a": "Started after using a new cleaning product; itchy; only on hands"}]
    print("== fusion =="); import json as _j; print(_j.dumps(fuse_history(diff, ans, img), indent=2))
