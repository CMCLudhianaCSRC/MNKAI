"""Export a trained mobile model to ONNX (and optionally TFLite via onnx-tf).

The TFLite path requires `onnx`, `onnxruntime`, and `onnx-tf` (optional). If
TFLite tools are not installed, only ONNX is produced and the user can convert
later via `onnx2tf` or `tflite-runtime`.

    python -m scripts.export_mobile \
        --ckpt checkpoints/mobilenetv3_large_fold0/best.pt \
        --img-size 256 --opset 17 \
        --out checkpoints/mobilenetv3_large_fold0/student.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from mnkai.models.backbones import ModelConfig, build_model
from mnkai.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", action="store_true",
                    help="run dynamic INT8 quantization on the ONNX output")
    args = ap.parse_args()

    setup_logging("logs", "export_mobile.log", "INFO")
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {})
    name = cfg.get("train", {}).get("model_name") or "mobilenetv3_large_100.ra_in1k"
    mcfg = ModelConfig(name=name, pretrained=False, num_classes=args.num_classes,
                       drop_rate=0.0, drop_path_rate=0.0, head_dropout=0.0)
    model = build_model(mcfg).cpu().eval()
    model.load_state_dict(obj.get("model_state") or obj["state_dict"], strict=False)

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    logger.info(f"[mobile] wrote {out} ({out.stat().st_size/1e6:.2f} MB)")

    if args.int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            int8_path = out.with_suffix(".int8.onnx")
            quantize_dynamic(str(out), str(int8_path), weight_type=QuantType.QInt8)
            logger.info(f"[mobile] INT8 quantized -> {int8_path} ({int8_path.stat().st_size/1e6:.2f} MB)")
        except Exception as e:
            logger.warning(f"[mobile] INT8 quant failed: {e}")


if __name__ == "__main__":
    main()
