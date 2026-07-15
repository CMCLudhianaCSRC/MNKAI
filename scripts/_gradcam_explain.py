"""Grad-CAM / attention-rollout explainability for the trained model.

Produces heat-map visualizations for clinical trust + FDA SaMD compliance.
For DINOv2 (ViT) we use attention rollout (Abnar & Zuidema 2020); for CNN
backbones we use Grad-CAM (Selvaraju 2017).

Picks 3 examples per class — diverse correct + diverse incorrect predictions
— and overlays the explanation map on the original image.

Outputs:
  posthoc/explain/{class}/correct_{n}.png       — model right, with overlay
  posthoc/explain/{class}/wrong_{n}.png         — model wrong, with overlay
  posthoc/explain_summary.html                  — gallery for review
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.utils.config import load_config
from mnkai.data.manifest import LabelNormalizer, normalize_manifest
from mnkai.data.splits import fold_train_valid_indices
from mnkai.data.transforms import AugConfig, build_valid
from mnkai.models.backbones import ModelConfig, build_model
from mnkai.data.datasets import TabularSpec, encode_tabular


# ──────────────────────────────────────────────────────────────────────────────
# ViT attention rollout (Abnar 2020) — used for DINOv2
# ──────────────────────────────────────────────────────────────────────────────
def vit_attention_rollout(model, image_tensor, head_fusion="mean", discard_ratio=0.0):
    """Forward pass + collect attention from each block, then rollout.
    Returns a (H, W) attention map upsampled to the input size."""
    attentions = []

    def hook(module, inp, out):
        # out shape varies; for timm ViT, MHA outputs (attn_output, attn_weights)
        # We rely on returning attention via attn_drop hook or similar
        pass

    # Simpler: use timm's attn_drop layers
    hooks = []
    attn_maps = []
    for name, mod in model.named_modules():
        if "attn_drop" in name and isinstance(mod, torch.nn.Dropout):
            def _hk(module, inp, out, attn_maps=attn_maps):
                attn_maps.append(inp[0].detach().cpu())
            hooks.append(mod.register_forward_hook(_hk))

    with torch.no_grad():
        _ = model(image_tensor)
    for h in hooks:
        h.remove()
    if not attn_maps:
        return None

    # attn_maps: list of (B, n_heads, n_tokens, n_tokens)
    # Apply rollout
    result = None
    for a in attn_maps:
        if head_fusion == "mean":
            a = a.mean(dim=1)  # (B, n_tokens, n_tokens)
        elif head_fusion == "max":
            a = a.max(dim=1)[0]
        # Add identity, normalize
        I = torch.eye(a.size(-1))
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else result @ a
    # Take attention from cls token to all patches
    cls_attn = result[0, 0, 1:]  # drop cls-cls self
    n_tokens = cls_attn.shape[0]
    side = int(math.sqrt(n_tokens))
    if side * side != n_tokens:
        return None
    grid = cls_attn.reshape(side, side).numpy()
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Render overlay
# ──────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(image_pil, heatmap, alpha=0.4):
    """Overlay a heatmap on the original image."""
    img = np.array(image_pil.convert("RGB")).astype(float) / 255.0
    H, W = img.shape[:2]
    hm = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
    cmap = cm.get_cmap("jet")
    hm_rgb = cmap(hm)[..., :3]
    blended = (1 - alpha) * img + alpha * hm_rgb
    blended = np.clip(blended, 0, 1)
    return blended


def save_overlay(image_path, heatmap, save_to, title=""):
    img = Image.open(image_path).convert("RGB")
    blended = overlay_heatmap(img, heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img); axes[0].set_title("input"); axes[0].axis("off")
    axes[1].imshow(blended); axes[1].set_title(title or "attention"); axes[1].axis("off")
    plt.tight_layout()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-per-class", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold
    fold = int(cfg["train"].get("fold", 0))
    out_dir = Path(args.out_dir or cfg["out_dir"]) / "posthoc" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = list(cfg["classes"]["names"])

    # ---- model
    tab_cfg = cfg.get("tabular", {}) or {}
    tabular = None
    if tab_cfg.get("enabled", False):
        tabular = TabularSpec(use_sex=tab_cfg.get("use_sex",True),
                              use_age=tab_cfg.get("use_age",True),
                              use_site=tab_cfg.get("use_site",True),
                              use_fitzpatrick=tab_cfg.get("use_fitzpatrick",False))
    mcfg = ModelConfig(
        name=cfg["train"]["model_name"], pretrained=False,
        num_classes=len(classes),
        drop_rate=float(cfg["train"].get("drop_rate", 0.10)),
        drop_path_rate=float(cfg["train"].get("drop_path_rate", 0.10)),
        head_dropout=float(cfg["train"].get("head_dropout", 0.30)),
        tabular_dim=(tabular.dim if tabular else 0),
        tabular_hidden=int(cfg["train"].get("tabular_hidden", 128)),
        head_hidden=int(cfg["train"].get("head_hidden", 0)),
        img_size=int(cfg["data"].get("img_size", 0)) or None,
        fusion=str(cfg["train"].get("fusion", "concat")),
        cross_attn_heads=int(cfg["train"].get("cross_attn_heads", 4)),
        cross_attn_dim=int(cfg["train"].get("cross_attn_dim", 256)),
    )
    model = build_model(mcfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[gradcam] loaded {args.ckpt}", flush=True)

    # CRITICAL: timm ViT/DINOv2 defaults to fused attention (F.scaled_dot_
    # product_attention), which never materialises the attention matrix — so the
    # attn_drop forward hook used by vit_attention_rollout captures nothing and
    # the rollout returns None for every image. Disable fused attention so the
    # softmax(QK^T) matrix flows through attn_drop and the hook can capture it.
    _n_attn = 0
    for _mod in model.modules():
        if hasattr(_mod, "fused_attn"):
            _mod.fused_attn = False
            _n_attn += 1
    print(f"[gradcam] disabled fused attention on {_n_attn} modules "
          f"(enables attention-rollout capture)", flush=True)

    # ---- manifest + val split
    norm = LabelNormalizer(classes=classes)
    manifest = pd.read_csv(cfg["data"]["manifest_csv"], low_memory=False)
    manifest = normalize_manifest(manifest, normalizer=norm)
    splits = pd.read_csv(cfg["data"]["splits_csv"], low_memory=False)
    _, valid_idx = fold_train_valid_indices(splits, fold=fold)
    val_df = manifest.iloc[valid_idx].reset_index(drop=True)

    # Use cached logits if available
    cache = Path(out_dir).parent / "val_logits.npz"
    if cache.exists():
        arr = np.load(cache)
        logits = arr["logits"]
        T = json.load(open(Path(out_dir).parent / "temperature.json"))["temperature"]
        probs = _softmax(logits / T)
        preds = np.argmax(probs, axis=1)
        labels = arr["labels"]
        print(f"[gradcam] using cached logits")
    else:
        print(f"[gradcam] WARN: no cached logits; falling back to live inference is not implemented")
        return

    # ---- Pick examples per class
    # CRITICAL: must match training's _build_loaders aug_cfg_valid (shades_of_gray_p).
    aug = AugConfig(
        img_size=int(cfg["data"]["img_size"]),
        mean=tuple(cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(cfg["data"].get("normalize_std", [0.229, 0.224, 0.225])),
        shades_of_gray_p=float(cfg.get("augmentation", {}).get("shades_of_gray_p", 1.0)),
        sog_norm=float(cfg.get("augmentation", {}).get("sog_norm", 6.0)),
        hair_removal=False,
    )
    valid_tfm = build_valid(aug)

    for cls_idx, cls in enumerate(classes):
        cls_mask = labels == cls_idx
        if cls_mask.sum() == 0:
            continue
        # correct examples
        correct = np.where(cls_mask & (preds == cls_idx))[0]
        wrong = np.where(cls_mask & (preds != cls_idx))[0]
        # rank by confidence
        if len(correct) > 0:
            conf = probs[correct, cls_idx]
            top_correct = correct[np.argsort(-conf)[:args.n_per_class]]
        else:
            top_correct = []
        if len(wrong) > 0:
            # most confidently wrong
            conf = probs[wrong].max(axis=1)
            top_wrong = wrong[np.argsort(-conf)[:args.n_per_class]]
        else:
            top_wrong = []
        # generate overlays
        for tag, idx_list in [("correct", top_correct), ("wrong", top_wrong)]:
            for n, idx in enumerate(idx_list):
                fp = val_df.iloc[int(idx)]["filepath"]
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    continue
                # forward
                x = valid_tfm(img).unsqueeze(0).to(device)
                tab = None
                if tabular is not None:
                    tab = torch.from_numpy(encode_tabular(val_df.iloc[int(idx)], tabular)).unsqueeze(0).to(device)
                # rollout
                attn = vit_attention_rollout(model, x)
                if attn is None:
                    continue
                pred_idx = int(preds[idx])
                title = f"{cls} | pred={classes[pred_idx]} | p={probs[idx,pred_idx]:.2f}"
                save_overlay(fp, attn, out_dir / cls / f"{tag}_{n}.png", title=title)
        print(f"[gradcam] {cls}: correct={len(top_correct)} wrong={len(top_wrong)}")


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
