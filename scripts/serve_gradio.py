"""Gradio demo app (PPT specifies Gradio 4.x).

Loads any classifier checkpoint, runs flip + multi-scale TTA, optionally
applies temperature scaling and the abstention threshold.

    python -m scripts.serve_gradio \
        --ckpt checkpoints/convnext_tiny_fold0/best.pt \
        --classes mel,nv,bcc,scc,df,bkl,ak,vasc \
        --img-size 384
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from mnkai.data.transforms import AugConfig, build_valid
from mnkai.evaluation.tta import multiscale_tta
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.logging import setup_logging


def _load_checkpoint(path: str, num_classes: int, img_size: int, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    cfg = obj.get("config", {})
    name = cfg.get("train", {}).get("model_name") or "convnext_tiny.fb_in22k_ft_in1k"
    mcfg = ModelConfig(name=name, pretrained=False, num_classes=num_classes,
                       drop_rate=0.0, drop_path_rate=0.0, head_dropout=0.0)
    model = build_model(mcfg).to(device).eval()
    model.load_state_dict(obj.get("model_state") or obj["state_dict"], strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", default="mel,nv,bcc,scc,df,bkl,ak,vasc")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--abstain-threshold", type=float, default=0.0,
                    help="if max-confidence < this, return 'abstain'")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--strip-exif", action="store_true",
                    help="reflect PPT ethics: drop EXIF before showing the image back")
    args = ap.parse_args()

    setup_logging("logs", "gradio.log", "INFO")
    classes = [c.strip().lower() for c in args.classes.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint(args.ckpt, len(classes), args.img_size, device)
    aug = AugConfig(img_size=args.img_size)
    tfm = build_valid(aug)
    logger.info(f"[gradio] loaded {args.ckpt} on {device}; serving on :{args.port}")

    @torch.no_grad()
    def infer(img: Image.Image):
        if img is None:
            return {}, "Please upload an image."
        if args.strip_exif:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=95)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        x = tfm(img).unsqueeze(0).to(device)
        prob = multiscale_tta(model, x).cpu().numpy()[0]
        if args.temperature != 1.0:
            logits = np.log(np.clip(prob, 1e-12, 1.0))
            prob = np.exp(logits / args.temperature)
            prob = prob / prob.sum()
        scores = {c: float(prob[i]) for i, c in enumerate(classes)}
        scores = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
        if prob.max() < args.abstain_threshold:
            note = f"⚠️ low confidence ({prob.max():.2f}) — model abstained"
        else:
            top = max(scores, key=scores.get)
            note = f"Top: **{top}** ({scores[top]*100:.1f}%)"
        return scores, note

    with gr.Blocks(title="MNKAI Dermatology Classifier — Demo") as demo:
        gr.Markdown("# MNKAI — 8-class dermatology classifier (research demo)")
        gr.Markdown("Upload a clinical or dermoscopic image. **Not for clinical use.**")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="image")
                btn = gr.Button("predict", variant="primary")
            with gr.Column():
                out_label = gr.Label(num_top_classes=len(classes), label="probabilities")
                out_md = gr.Markdown()
        btn.click(infer, inputs=[img_in], outputs=[out_label, out_md])

    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
