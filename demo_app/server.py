"""MNKAI-III demo — backend server (Python stdlib, no extra deps).

Run:
  PYTHONPATH=src .venv/Scripts/python.exe demo_app/server.py --port 8800
Then open http://localhost:8800

API (JSON in / JSON out, images as base64 data-URLs):
  POST /api/analyze     {image, meta, options}     -> full analysis
  POST /api/scorecam    {image, meta, target_idx}  -> Score-CAM overlay (slow)
  GET  /api/embedding_space                          -> prototype scatter (if built)
  GET  /api/health
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import inference as I
import explain as X
import analytics as AN
from inference import engine, list_models, to_b64_png, decode_dataurl, CLASS_INFO

STATIC = HERE / "static"


def analyze(payload: dict) -> dict:
    eng = engine(payload.get("model"))
    opts = payload.get("options", {}) or {}
    meta = payload.get("meta", {}) or {}
    pil = decode_dataurl(payload["image"])
    orig_size = list(pil.size)

    img_t, tab_t, disp_u8 = eng.preprocess(pil, meta)
    logits, probs, dt = eng.predict(img_t, tab_t)
    order = probs.argsort()[::-1]
    classes = eng.classes
    topk = [{"class": classes[i], "info": CLASS_INFO.get(classes[i], {}),
             "prob": round(float(probs[i]), 4)} for i in order[:5]]
    target = int(opts.get("target_idx", order[0]))

    # uncertainty (MC-dropout optional, default on with small n)
    mc_mean = mc_std = None
    if opts.get("mc_dropout", True):
        n = int(opts.get("mc_samples", 15))
        mc_mean, mc_std = eng.mc_dropout(img_t, tab_t, n=n)
    unc = AN.uncertainty_summary(probs, mc_mean, mc_std)
    decision = AN.decision_support(classes, probs, unc, meta, CLASS_INFO, eng.key)

    # explainability
    heat = {}
    try: heat["grad_cam"] = to_b64_png(X.overlay(disp_u8, X.grad_cam(eng, img_t, tab_t, target)))
    except Exception: heat["grad_cam"] = None
    try: heat["grad_cam_pp"] = to_b64_png(X.overlay(disp_u8, X.grad_cam_pp(eng, img_t, tab_t, target)))
    except Exception: heat["grad_cam_pp"] = None
    try: heat["saliency"] = to_b64_png((X.saliency(eng, img_t, tab_t, target) * 255).astype("uint8"))
    except Exception: heat["saliency"] = None
    att = None
    try:
        amap = X.attention_rollout(eng, img_t)
        att = to_b64_png(X.overlay(disp_u8, amap)) if amap is not None else None
    except Exception:
        att = None
    heat["attention_rollout"] = att

    fmaps = []
    try:
        for fm in X.feature_maps(eng, img_t):
            fmaps.append({"layer": fm["layer"], "depth_frac": fm["depth_frac"],
                          "img": to_b64_png(X.overlay(disp_u8, fm["map"]))})
    except Exception:
        pass

    stages = []
    try:
        for s in X.how_it_saw(eng, disp_u8, img_t, tab_t, target):
            stages.append({"title": s["title"], "caption": s["caption"], "kind": s["kind"],
                           "img": to_b64_png(s["img"])})
    except Exception:
        traceback.print_exc()

    # embedding / latent
    emb = eng.embedding(img_t)
    latent = {"nearest_class_centroid": AN.nearest_class_in_latent(emb),
              "uploaded_xy": AN.project_uploaded(emb),
              "embedding_dim": int(emb.shape[0]),
              "embedding_l2": round(float((emb ** 2).sum() ** 0.5), 3)}

    return {
        "ok": True,
        "model": eng.model_info(),
        "image_meta": {"original_size": orig_size, "input_size": [eng.img_size, eng.img_size],
                       "mode": pil.mode, "supplied_meta": meta},
        "prediction": {"class": classes[target], "info": CLASS_INFO.get(classes[target], {}),
                       "prob": round(float(probs[target]), 4),
                       "inference_ms": round(dt, 1)},
        "topk": topk,
        "softmax": [{"class": c, "prob": round(float(probs[i]), 4)} for i, c in enumerate(classes)],
        "uncertainty": unc,
        "decision_support": decision,
        "heatmaps": heat,
        "feature_maps": fmaps,
        "pipeline": stages,
        "latent": latent,
        "processed_image": to_b64_png(disp_u8),
    }


def scorecam(payload: dict) -> dict:
    eng = engine(payload.get("model"))
    pil = decode_dataurl(payload["image"])
    img_t, tab_t, disp_u8 = eng.preprocess(pil, payload.get("meta", {}) or {})
    _, probs, _ = eng.predict(img_t, tab_t)
    target = int(payload.get("target_idx", int(probs.argmax())))
    cam = X.score_cam(eng, img_t, tab_t, target, top_k=int(payload.get("top_k", 24)))
    return {"ok": True, "score_cam": to_b64_png(X.overlay(disp_u8, cam))}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send((STATIC / "index.html").read_bytes(), ctype="text/html; charset=utf-8")
        if self.path.startswith("/static/"):
            f = STATIC / self.path[len("/static/"):]
            if f.is_file():
                ct = ("text/css" if f.suffix == ".css" else
                      "application/javascript" if f.suffix == ".js" else "application/octet-stream")
                return self._send(f.read_bytes(), ctype=ct)
            return self._send({"error": "not found"}, 404)
        if self.path == "/api/health":
            return self._send({"ok": True, "model": engine().model_info(), "models": list_models()})
        if self.path == "/api/models":
            return self._send({"ok": True, "models": list_models()})
        if self.path == "/api/embedding_space":
            z = AN.load_prototypes()
            if z is None:
                return self._send({"ok": False, "reason": "prototypes not built; run demo_app/_compute_prototypes.py"})
            return self._send({"ok": True, "coords": z["coords"].tolist(),
                               "labels": [str(x) for x in z["labels"]],
                               "classes": z["classes"], "method": z["method"]})
        return self._send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode())
            if self.path == "/api/analyze":
                return self._send(analyze(payload))
            if self.path == "/api/scorecam":
                return self._send(scorecam(payload))
            return self._send({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._send({"ok": False, "error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"[demo] loading model ...", flush=True)
    info = engine().model_info()
    print(f"[demo] model ready: {info['architecture']} ({info['params_millions']}M) on {info['device']}", flush=True)
    print(f"[demo] serving http://{args.host}:{args.port}  (open http://localhost:{args.port})", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
