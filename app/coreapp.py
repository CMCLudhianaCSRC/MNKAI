# app/app.py
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from PIL import Image
from gradio.events import SelectData
import random

# Short names to keep the bar-plot tidy (edit if you use other labels)
PRETTY = {"mel": "MEL", "nv": "NV", "bcc": "BCC", "ak": "AK",
          "bkl": "BKL", "df": "DF", "vasc": "VASC", "scc": "SCC"}

# make local imports work whether run as "python -m app.app" or "python app/app.py"
APP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
for p in (APP_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- metrics helpers (AUC/F1 table + pretty markdown) ----
# ----------------------------------------------------------

# Patch pytorch-grad-cam destructor to avoid cleanup warnings on some versions
try:
    import pytorch_grad_cam.base_cam as base_cam
    def _safe_del(self):
        try:
            ag = getattr(self, "activations_and_grads", None)
            if ag is not None:
                ag.release()
        except Exception:
            pass
    base_cam.BaseCAM.__del__ = _safe_del
except Exception:
    pass

# -------- fallbacks if app/ui_utils.py is missing ----------
def _fallback_pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)

def _fallback_overlay_heatmap(np_rgb: np.ndarray, grayscale_cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    try:
        import cv2
        heat = (np.clip(grayscale_cam, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
        heat = cv2.resize(heat, (np_rgb.shape[1], np_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        over = (alpha * heat + (1 - alpha) * np_rgb).astype(np.uint8)
        return Image.fromarray(over)
    except Exception:
        heat = (np.clip(grayscale_cam, 0, 1) * 255).astype(np.uint8)
        heat3 = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=-1)
        heat3 = (heat3 * alpha + np_rgb * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(heat3)

try:
    from app.ui_utils import pil_to_rgb_np, overlay_heatmap
except Exception:
    try:
        from .ui_utils import pil_to_rgb_np, overlay_heatmap
    except Exception:
        pil_to_rgb_np = _fallback_pil_to_rgb_np
        overlay_heatmap = _fallback_overlay_heatmap
# ----------------------------------------------------------

from src.transforms import AugConfig, build_valid
from src.model import build_model
from src.utils import get_device

_TEMP_MISSING_ONCE = False  # one-time warning gate for temperature file
_SAMPLE_POOL = None  # cached manifest-sourced image paths


def _detect_path_column(df: pd.DataFrame) -> str:
    candidates = ["filepath", "image_path", "image", "path", "filename"]
    cols = {str(c).lower(): c for c in df.columns}
    for key in candidates:
        if key in cols:
            return cols[key]
    return ""


def _resolve_image_path(path_str: str, data_cfg: dict) -> Optional[Path]:
    if not path_str:
        return None
    raw = Path(path_str)
    if raw.is_file():
        return raw

    images_dir = data_cfg.get("images_dir") or ""
    if images_dir:
        img_root = Path(images_dir)
        if not img_root.is_absolute():
            img_root = Path(ROOT_DIR) / img_root
        candidate = img_root / path_str
        if candidate.is_file():
            return candidate

    if not raw.is_absolute():
        candidate = Path(ROOT_DIR) / raw
        if candidate.is_file():
            return candidate
    return None


def _load_sample_pool(cfg: dict) -> list[str]:
    global _SAMPLE_POOL
    if _SAMPLE_POOL is not None:
        return _SAMPLE_POOL

    data_cfg = cfg.get("data", {})
    manifest = data_cfg.get("manifest_csv") or data_cfg.get("metadata_csv")
    if not manifest:
        logger.warning("[samples] No manifest_csv configured; sample gallery disabled.")
        _SAMPLE_POOL = []
        return _SAMPLE_POOL

    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path(ROOT_DIR) / manifest_path
    if not manifest_path.is_file():
        logger.warning("[samples] Manifest file missing at %s", manifest_path)
        _SAMPLE_POOL = []
        return _SAMPLE_POOL

    try:
        df = pd.read_csv(manifest_path)
    except Exception as exc:
        logger.warning("[samples] Failed to read manifest: %s", exc)
        _SAMPLE_POOL = []
        return _SAMPLE_POOL

    path_col = _detect_path_column(df)
    if not path_col:
        logger.warning("[samples] Could not infer image path column in manifest.")
        _SAMPLE_POOL = []
        return _SAMPLE_POOL

    values = df[path_col].dropna().astype(str).tolist()
    if not values:
        _SAMPLE_POOL = []
        return _SAMPLE_POOL

    rng = random.Random(42)
    rng.shuffle(values)
    pool: list[str] = []
    max_candidates = min(len(values), 2000)
    for value in values[:max_candidates]:
        resolved = _resolve_image_path(value, data_cfg)
        if resolved is not None and resolved.is_file():
            pool.append(resolved.as_posix())
    _SAMPLE_POOL = pool
    if not _SAMPLE_POOL:
        logger.warning("[samples] No valid image paths resolved from manifest.")
    else:
        logger.info("[samples] Cached %d sample image paths.", len(_SAMPLE_POOL))
    return _SAMPLE_POOL


def get_random_samples(cfg: dict, limit: int = 8) -> list[str]:
    pool = _load_sample_pool(cfg)
    if not pool:
        return []
    if len(pool) <= limit:
        return pool
    return random.sample(pool, limit)


def build_chat_reply(message: str, cfg: dict) -> str:
    prompt = (message or "").strip()
    if not prompt:
        return "Hi! Ask me about training, inference or how to interpret the outputs."

    lower = prompt.lower()
    replies: list[str] = []
    train_cfg = cfg.get("train", {})
    model_name = train_cfg.get("model_name", "the current backbone")
    class_count = len(cfg.get("classes", {}).get("names", []))
    logging_cfg = cfg.get("logging", {})
    log_dir = Path(logging_cfg.get("log_dir", "logs"))
    log_file = logging_cfg.get("log_file", "train_isic.log")
    log_path = (log_dir / log_file).as_posix()

    if any(word in lower for word in ("log", "metric", "f1")):
        replies.append(
            f"Training metrics stream into `{log_path}` and TensorBoard (`{cfg.get('tensorboard', {}).get('log_dir', 'runs')}`)."
        )
    if "slow" in lower or "speed" in lower:
        replies.append(
            "For faster live inference keep hair removal off, disable flip TTA, and use the sample gallery to try cached images."
        )
    if "accuracy" in lower or "improve" in lower:
        replies.append(
            "Accuracy jumps after warm-up; the ReduceLROnPlateau scheduler lowers the LR automatically once macro-F1 stalls."
        )
    if "sample" in lower or "demo" in lower:
        replies.append(
            "Hit *Refresh sample gallery* to pull eight random manifest entries and click any tile to load it instantly."
        )
    if "model" in lower or "backbone" in lower:
        replies.append(
            f"We're running a `{model_name}` backbone tuned for {class_count} skin lesion classes."
        )
    if "help" in lower or not replies:
        replies.append(
            "Need more? Ask about deployment tips, calibration, or how to export ONNX via `tools/export_onnx.py`."
        )
    return " ".join(replies)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_temperature_scale(logits: torch.Tensor, cfg) -> torch.Tensor:
    """
    Optionally apply temperature scaling using checkpoints/temperature.json,
    controlled by:
      deploy:
        apply_temperature: true
        temperature_file: "checkpoints/temperature.json"
    """
    global _TEMP_MISSING_ONCE
    deploy = cfg.get("deploy", {})
    if not deploy or not deploy.get("apply_temperature", False):
        return logits
    temp_path = Path(deploy.get("temperature_file", "checkpoints/temperature.json"))
    if not temp_path.is_file():
        if not _TEMP_MISSING_ONCE:
            logger.warning(f"[temp] Temperature file not found: {temp_path}. Skipping.")
            _TEMP_MISSING_ONCE = True
        return logits
    try:
        data = json.loads(temp_path.read_text(encoding="utf-8"))
        T = float(max(1e-3, data.get("temperature", 1.0)))
        return logits / T
    except Exception as e:
        if not _TEMP_MISSING_ONCE:
            logger.warning(f"[temp] Failed to read {temp_path}: {e}")
            _TEMP_MISSING_ONCE = True
        return logits


def _pick_last_conv(model: torch.nn.Module):
    """Heuristically find the last Conv2d-like layer for CAM."""
    convs = []
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is not None and getattr(w, "ndim", 0) == 4:
            convs.append(m)
    return convs[-1] if convs else None


def try_gradcam(model: torch.nn.Module, device: torch.device, np_rgb: np.ndarray) -> Image.Image | None:
    """Return PIL heatmap overlay or None; robust across pytorch-grad-cam versions."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception:
        logger.warning("[gradcam] Not installed. To enable: pip install git+https://github.com/jacobgil/pytorch-grad-cam")
        return None

    target_layer = _pick_last_conv(model)
    if target_layer is None:
        logger.warning("[gradcam] No convolutional layer found; skipping heatmap.")
        return None

    # Downscale for speed if large
    try:
        import cv2
        h, w = np_rgb.shape[:2]
        scale = 256 / max(h, w)
        np_proc = cv2.resize(np_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else np_rgb
    except Exception:
        np_proc = np_rgb

    x = torch.from_numpy((np_proc.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)

    try:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                pred = int(logits.argmax(dim=-1).item())
            grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)], eigen_smooth=False)[0]
    except TypeError:
        # older versions may expect use_cuda kwarg
        try:
            with GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == "cuda")) as cam:
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                    pred = int(logits.argmax(dim=-1).item())
                grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)], eigen_smooth=False)[0]
        except Exception as e2:
            logger.warning(f"[gradcam] Failed to compute heatmap: {e2}")
            return None
    except Exception as e:
        logger.warning(f"[gradcam] Failed to compute heatmap: {e}")
        return None

    heat_small = overlay_heatmap(np_proc, grayscale_cam)  # PIL
    if np_proc is not np_rgb:
        heat = heat_small.resize((np_rgb.shape[1], np_rgb.shape[0]), Image.BILINEAR)
    else:
        heat = heat_small
    return heat


def load_model(ckpt_path: str, cfg):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "classes" in ckpt:
        classes = [str(c) for c in ckpt["classes"]]
    else:
        cfg_classes = cfg.get("classes", {}).get("names")
        if not cfg_classes:
            raise KeyError("`classes` missing in checkpoint and config; cannot determine class order.")
        classes = [str(c) for c in cfg_classes]
        logger.warning("[load_model] `classes` not stored in %s. Falling back to config order.", ckpt_path)
    device = get_device()
    model = build_model(
        cfg["train"]["model_name"],
        num_classes=len(classes),
        pretrained=False,
        drop_rate=0.0,
        compile_model=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, classes, device


def make_pipeline(cfg):
    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"]["normalize_mean"]),
        std=tuple(cfg["data"]["normalize_std"]),
    )
    return build_valid(aug, center_crop=True)


def infer_image(pil_img, hair_rm, do_tta, abstain_thr, show_cam, model, device, classes, valid_tf, cfg):
    if pil_img is None:
        empty_df = pd.DataFrame(columns=["class", "prob"])
        return None, "No image provided", "[]", empty_df, None

    np_rgb_orig = pil_to_rgb_np(pil_img)

    # optional hair removal (OpenCV blackhat + Telea inpaint)
    np_rgb_proc = np_rgb_orig
    if hair_rm:
        try:
            import cv2
            from src.hair_removal import remove_hair_bgr
            bgr = cv2.cvtColor(np_rgb_orig, cv2.COLOR_RGB2BGR)
            bgr = remove_hair_bgr(bgr)
            np_rgb_proc = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"[hair] hair removal failed: {e}")

    tf_out = valid_tf(image=np_rgb_proc)
    x = tf_out["image"].unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(x)
        logits = maybe_temperature_scale(logits, cfg)
        if do_tta:
            flips = ([3], [2], [2, 3])  # H, W, HW flips
            bag = [logits]
            for dims in flips:
                bag.append(maybe_temperature_scale(model(torch.flip(x, dims=dims)), cfg))
            logits = torch.stack(bag, dim=0).mean(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top = int(probs.argmax())
    top_p = float(probs[top])
    shown = f"Predicted: {PRETTY.get(classes[top], classes[top])} ({top_p:.3f})"
    if abstain_thr is not None and top_p < float(abstain_thr):
        shown = f"Predicted: ABSTAIN (<{float(abstain_thr):.2f})"

    order = list(np.argsort(-probs))
    rows = [{"class": PRETTY.get(classes[i], classes[i]), "prob": float(probs[i])} for i in order]
    df_plot = pd.DataFrame(rows, columns=["class", "prob"])

    heat = try_gradcam(model, device, np_rgb_proc) if show_cam else None
    return Image.fromarray(np_rgb_orig), shown, json.dumps(rows, indent=2), df_plot, heat


def build_ui(model, device, classes, valid_tf, cfg):
    css = """
    #hero-card {
        background: linear-gradient(135deg, #f5f7fb 0%, #eef2ff 100%);
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 12px;
    }
    #hero-card ul { margin: 0 0 0 1.2rem; }
    #assistant-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
    }
    """
    with gr.Blocks(title="DermDesk - Local Dermatology AI", css=css) as demo:
        gr.Markdown("# DermDesk - Local Dermatology AI")
        gr.Markdown(
            "### Local-first lesion triage\n"
            "- Upload or pick a sample image from the manifest\n"
            "- Toggle hair removal, TTA, and Grad-CAM to compare outputs\n"
            "- Chat with the assistant for quick training or deployment tips",
            elem_id="hero-card",
        )

        sample_paths_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("#### Upload or pick an image")
                img = gr.Image(type="pil", label="Upload Image")
                hair = gr.Checkbox(label="Apply hair removal (experimental)", value=False)
                do_tta = gr.Checkbox(label="Flip-TTA (slower, more stable)", value=False)
                show_cam = gr.Checkbox(label="Show Grad-CAM heatmap", value=True)
                abstain = gr.Slider(0.0, 0.9, value=0.55, step=0.05, label="Abstain below prob.")
                btn = gr.Button("Predict", variant="primary")

                with gr.Accordion("Need an example image?", open=False):
                    gr.Markdown(
                        "Refresh to grab eight random samples from the manifest. Click any tile to load it instantly."
                    )
                    sample_btn = gr.Button("Refresh sample gallery", variant="secondary")
                    samples_gallery = gr.Gallery(
                        label="Sample images",
                        columns=4,
                        height=220,
                        show_label=False,
                    )

            with gr.Column(scale=2):
                gr.Markdown("#### Results")
                out_img = gr.Image(label="Input Preview", height=320)
                out_label = gr.Markdown()
                out_json = gr.Code(label="Raw probabilities (JSON)")
                out_plot = gr.BarPlot(
                    x="class",
                    y="prob",
                    label="Class Probabilities",
                    x_title="Class",
                    y_title="Prob.",
                    vertical=True,
                )
                out_heat = gr.Image(label="Grad-CAM Heatmap", height=320)

        def _ui_predict(im, hr, tta, cam_on, abstain_thr):
            return infer_image(im, hr, tta, abstain_thr, cam_on, model, device, classes, valid_tf, cfg)

        def _refresh_samples():
            choices = get_random_samples(cfg, limit=8)
            gallery_items = [(path, Path(path).name) for path in choices]
            return gallery_items, choices

        def _apply_sample(evt: SelectData, paths):
            if not paths:
                raise gr.Error("Refresh the gallery first to load sample images.")
            if evt is None:
                raise gr.Error("Tap on a sample thumbnail to load it.")
            index = getattr(evt, "index", None)
            if isinstance(index, (list, tuple)):
                index = index[0] if index else 0
            if index is None:
                index = 0
            try:
                idx = int(index)
            except Exception:
                idx = 0
            idx = max(0, min(idx, len(paths) - 1))
            chosen = paths[idx]
            try:
                with Image.open(chosen) as im:
                    return im.convert("RGB").copy()
            except Exception as exc:
                logger.warning("[samples] Failed to open sample %s: %s", chosen, exc)
                raise gr.Error("Unable to open the selected sample image.")

        sample_btn.click(
            _refresh_samples,
            inputs=[],
            outputs=[samples_gallery, sample_paths_state],
            queue=False,
        )
        samples_gallery.select(
            _apply_sample,
            inputs=[sample_paths_state],
            outputs=[img],
            queue=False,
        )

        btn.click(
            fn=_ui_predict,
            inputs=[img, hair, do_tta, show_cam, abstain],
            outputs=[out_img, out_label, out_json, out_plot, out_heat],
        )

        gr.Markdown("### DermDesk Assistant")
        with gr.Row(elem_id="assistant-card"):
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Assistant responses", height=380)
                chat_input = gr.Textbox(
                    label="Ask DermDesk",
                    placeholder="Example: How do I speed up inference?",
                    lines=3,
                )
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat", variant="secondary")

                def _chat(message, history):
                    history = history or []
                    reply = build_chat_reply(message, cfg)
                    history = history + [(message, reply)]
                    return history, ""

                def _reset_chat():
                    return [], ""

                chat_input.submit(_chat, inputs=[chat_input, chat], outputs=[chat, chat_input], queue=False)
                send_btn.click(_chat, inputs=[chat_input, chat], outputs=[chat, chat_input], queue=False)
                clear_btn.click(_reset_chat, inputs=None, outputs=[chat, chat_input], queue=False)

            with gr.Column(scale=1, min_width=280):
                gr.Markdown(
                    "#### Pro tips\n"
                    "- Logs stream to `logs/train_isic.log`\n"
                    "- TensorBoard runs are under `runs/`\n"
                    "- Use flip TTA sparingly for latency-sensitive demos\n"
                    "- Export ONNX with `python tools/export_onnx.py`\n"
                    "- Temperature scaling lives in `deploy.temperature_file`\n"
                )

        demo.load(
            lambda: _refresh_samples(),
            inputs=None,
            outputs=[samples_gallery, sample_paths_state],
            queue=False,
        )

    return demo


def main(args):
    cfg = load_config(args.config)
    model, classes, device = load_model(args.checkpoint, cfg)
    valid_tf = make_pipeline(cfg)
    demo = build_ui(model, device, classes, valid_tf, cfg)

    try:
        demo.launch(server_name=args.host, server_port=args.port, inbrowser=True, share=args.share)
    except Exception as e:
        logger.warning(f"[ui] Local launch failed ({e}); retrying with share=True")
        demo.launch(server_name="0.0.0.0", server_port=args.port, inbrowser=True, share=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    main(args)
