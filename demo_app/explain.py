"""MNKAI-III demo — explainability suite for the DINOv2 ViT.

Robust methods (always available): Grad-CAM, Grad-CAM++, Saliency, Feature maps,
"how the model saw it" pipeline. Best-effort (graceful fallback if the timm
version differs): Attention rollout, Score-CAM.

ViT specifics: patch tokens are the last `num_prefix_tokens`-offset tokens of
forward_features; we reshape them to the patch grid (16x16 for 224/14) and
upsample to the input resolution.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ----------------------------------------------------------------- colour --
def _jet(x: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 (jet-like)."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(base_u8: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """Overlay a [0,1] cam (HxW) on an RGB uint8 image."""
    H, W = base_u8.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_r = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = _jet(cam_r).astype(np.float32)
    out = (1 - alpha) * base_u8.astype(np.float32) + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def _norm(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    m = a.max()
    return a / m if m > 1e-8 else a


def _grid(bb):
    prefix = int(getattr(bb, "num_prefix_tokens", 1))
    g = getattr(getattr(bb, "patch_embed", None), "grid_size", None)
    if g is None:
        g = (16, 16)
    return prefix, int(g[0]), int(g[1])


def _tokens_to_map(tok_patch: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """(num_patches,) -> (gh, gw) normalised."""
    n = gh * gw
    if tok_patch.shape[0] != n:
        s = int(round(tok_patch.shape[0] ** 0.5)); gh = gw = s; n = s * s
    return _norm(tok_patch[:n].reshape(gh, gw))


# ------------------------------------------------------------- Grad-CAM(++) --
def _gradcam_core(eng, img_t, tab_t, target_idx, plus=False):
    model = eng.model
    bb = model.backbone
    prefix, gh, gw = _grid(bb)
    acts, grads = {}, {}

    h1 = bb.blocks[-1].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))

    def bwd(m, gi, go):
        grads["g"] = go[0]
    h2 = bb.blocks[-1].register_full_backward_hook(bwd)

    model.zero_grad(set_to_none=True)
    logits = model(img_t, tabular=tab_t) if tab_t is not None else model(img_t)
    score = logits[0, target_idx]
    score.backward()
    h1.remove(); h2.remove()

    A = acts["a"][0, prefix:, :]            # (N, C)
    G = grads["g"][0, prefix:, :]           # (N, C)
    if not plus:
        w = G.mean(0)                       # (C,)
    else:
        G2 = G ** 2; G3 = G2 * G
        denom = 2 * G2 + (A * G3).sum(0, keepdim=True)
        alpha = G2 / torch.clamp(denom, min=1e-8)
        w = (alpha * F.relu(G)).sum(0)
    cam = F.relu((w.unsqueeze(0) * A).sum(-1))   # (N,)
    cam = cam.detach().cpu().numpy()
    return _tokens_to_map(cam, gh, gw)


def grad_cam(eng, img_t, tab_t, target_idx):
    return _gradcam_core(eng, img_t, tab_t, target_idx, plus=False)


def grad_cam_pp(eng, img_t, tab_t, target_idx):
    return _gradcam_core(eng, img_t, tab_t, target_idx, plus=True)


# -------------------------------------------------------------- Saliency --
def saliency(eng, img_t, tab_t, target_idx):
    model = eng.model
    x = img_t.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x, tabular=tab_t) if tab_t is not None else model(x)
    logits[0, target_idx].backward()
    s = x.grad[0].abs().max(0)[0].detach().cpu().numpy()
    return _norm(s)


# ----------------------------------------------------------- Feature maps --
@torch.no_grad()
def feature_maps(eng, img_t, layers=None):
    """Mean patch-token activation map at several transformer blocks."""
    bb = eng.model.backbone
    prefix, gh, gw = _grid(bb)
    nblocks = len(bb.blocks)
    if layers is None:
        layers = sorted(set([max(0, nblocks // 4 - 1), nblocks // 2 - 1,
                             3 * nblocks // 4 - 1, nblocks - 1]))
    caps = {}
    hooks = [bb.blocks[i].register_forward_hook(
        (lambda idx: (lambda m, inp, out: caps.__setitem__(idx, out)))(i)) for i in layers]
    _ = bb.forward_features(img_t)
    for h in hooks:
        h.remove()
    out = []
    for i in layers:
        A = caps[i][0, prefix:, :]                     # (N, C)
        amap = A.norm(dim=-1).detach().cpu().numpy()   # token activation strength
        out.append({"layer": int(i), "depth_frac": round((i + 1) / nblocks, 2),
                    "map": _tokens_to_map(amap, gh, gw)})
    return out


# ---------------------------------------------------- Attention rollout --
@torch.no_grad()
def attention_rollout(eng, img_t):
    """Best-effort: monkeypatch each block's attention to capture softmax maps,
    then roll out (A+I product). Returns a (gh,gw) map or None on incompatibility."""
    bb = eng.model.backbone
    prefix, gh, gw = _grid(bb)
    attns = []
    patched = []
    try:
        for blk in bb.blocks:
            attn_mod = blk.attn
            orig = attn_mod.forward

            def make(mod, orig_fwd):
                head_dim = getattr(mod, "head_dim", None)
                attn_dim = getattr(mod, "attn_dim", None)
                post_norm = getattr(mod, "norm", torch.nn.Identity())

                def fwd(x, attn_mask=None, is_causal=False):
                    B, N, C = x.shape
                    hd = head_dim or (C // mod.num_heads)
                    qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, hd).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    if hasattr(mod, "q_norm"):
                        q, k = mod.q_norm(q), mod.k_norm(k)
                    a = (q * mod.scale) @ k.transpose(-2, -1)
                    a = a.softmax(dim=-1)
                    attns.append(a.mean(1).detach())   # avg heads -> (B,N,N)
                    x = a @ v
                    x = x.transpose(1, 2).reshape(B, N, attn_dim or C)
                    x = post_norm(x)
                    x = mod.proj(x)
                    if hasattr(mod, "proj_drop"):
                        x = mod.proj_drop(x)
                    return x
                return fwd
            attn_mod.forward = make(attn_mod, orig)
            patched.append((attn_mod, orig))
        _ = bb.forward_features(img_t)
    except Exception:
        for mod, orig in patched:
            mod.forward = orig
        return None
    finally:
        for mod, orig in patched:
            mod.forward = orig
    if not attns:
        return None
    R = torch.eye(attns[0].size(-1), device=attns[0].device)
    for a in attns:
        a = a[0] + torch.eye(a.size(-1), device=a.device)
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
    cls_to_patch = R[0, prefix:].cpu().numpy()
    return _tokens_to_map(cls_to_patch, gh, gw)


# ------------------------------------------------------------- Score-CAM --
@torch.no_grad()
def score_cam(eng, img_t, tab_t, target_idx, top_k=24):
    """Expensive: mask input by top-K activation channels, weight by score."""
    bb = eng.model.backbone
    prefix, gh, gw = _grid(bb)
    caps = {}
    h = bb.blocks[-1].register_forward_hook(lambda m, i, o: caps.__setitem__("a", o))
    _ = bb.forward_features(img_t)
    h.remove()
    A = caps["a"][0, prefix:, :]                       # (N, C)
    strength = A.abs().mean(0)
    idxs = torch.topk(strength, min(top_k, A.shape[1])).indices
    base = img_t
    H, W = img_t.shape[-2:]
    cams = torch.zeros(gh * gw, device=img_t.device)
    weights = []
    masks = []
    for c in idxs:
        m = _tokens_to_map(A[:, c].detach().cpu().numpy(), gh, gw)
        mt = torch.from_numpy(m).to(img_t.device).view(1, 1, gh, gw)
        mt = F.interpolate(mt, size=(H, W), mode="bilinear", align_corners=False)
        masks.append(mt)
    for c, mt in zip(idxs, masks):
        masked = base * mt
        lg = eng.model(masked, tabular=tab_t) if tab_t is not None else eng.model(masked)
        weights.append(F.softmax(lg, -1)[0, target_idx].item())
    weights = torch.tensor(weights, device=img_t.device)
    weights = F.relu(weights)
    for w, c in zip(weights, idxs):
        cams += w * torch.from_numpy(_tokens_to_map(A[:, c].detach().cpu().numpy(), gh, gw)).to(img_t.device).view(-1)
    return _norm(cams.cpu().numpy().reshape(gh, gw))


# --------------------------------------------- "How the model saw it" --------
def how_it_saw(eng, disp_u8, img_t, tab_t, target_idx):
    """A curated, intuitive pipeline of stages from raw pixels to decision.
    Mixes faithful model internals (early/mid/late block activations, grad-cam)
    with classic-CV illustrations (edges/texture) clearly labelled as such."""
    from PIL import ImageFilter
    stages = []
    pil = Image.fromarray(disp_u8)

    stages.append({"title": "1 · Original", "kind": "image",
                   "caption": "The lesion as uploaded.", "img": disp_u8})
    stages.append({"title": "2 · Preprocessed (224²)", "kind": "image",
                   "caption": "Resized + normalised exactly as in training.", "img": disp_u8})

    edges = pil.convert("L").filter(ImageFilter.FIND_EDGES)
    stages.append({"title": "3 · Edge structure", "kind": "image",
                   "caption": "Illustrative: borders & shape cues (classic CV).",
                   "img": np.stack([np.asarray(edges)] * 3, -1)})

    tex = pil.filter(ImageFilter.EMBOSS)
    stages.append({"title": "4 · Texture relief", "kind": "image",
                   "caption": "Illustrative: surface texture / scale (classic CV).",
                   "img": np.asarray(tex.convert("RGB"))})

    fmaps = feature_maps(eng, img_t)
    labels = ["5 · Early features", "6 · Mid-level features", "7 · High-level lesion features"]
    pick = fmaps[:3] if len(fmaps) >= 3 else fmaps
    for lab, fm in zip(labels, pick):
        stages.append({"title": lab, "kind": "heat",
                       "caption": f"Real activation strength, block {fm['layer']} "
                                  f"({int(fm['depth_frac']*100)}% depth).",
                       "img": overlay(disp_u8, fm["map"])})

    cam = grad_cam(eng, img_t, tab_t, target_idx)
    stages.append({"title": "8 · Decision evidence (Grad-CAM)", "kind": "heat",
                   "caption": "Where the model looked to decide the class.",
                   "img": overlay(disp_u8, cam)})
    return stages
