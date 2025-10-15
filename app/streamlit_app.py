"""Streamlit demo for MNKAI Teledermatology."""

import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import timm
import yaml
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# -------------------------
# Configuration helpers
# -------------------------
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def load_model_ckpt(
    model_name: str,
    num_classes: int,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[List[str]]]:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} from {ckpt_path}")
    model.eval().to(device)
    classes = ckpt.get("classes")
    if classes is not None:
        classes = [str(c) for c in classes]
    return model, classes


# -------------------------
# Image + inference utils
# -------------------------
def preprocess_pil(img: Image.Image, img_size: int = 352) -> Tuple[torch.Tensor, Image.Image]:
    aligned = ImageOps.exif_transpose(img.convert("RGB"))
    resized = aligned.resize((img_size, img_size), Image.BICUBIC)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr)
    return tensor, resized


@torch.no_grad()
def infer_probs(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    batch = x.to(device, non_blocking=True)
    out = model(batch)
    if tta:
        out_h = model(torch.flip(batch, dims=[3]))
        out_v = model(torch.flip(batch, dims=[2]))
        out_hv = model(torch.flip(batch, dims=[2, 3]))
        out = (out + out_h + out_v + out_hv) / 4.0
    probs = F.softmax(out, dim=-1).squeeze(0).cpu().numpy()
    return probs


def pick_last_conv(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


def pick_first_conv(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


def find_cam_layer_name(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
) -> Optional[str]:
    conv_names: Dict[str, torch.Size] = {}
    hooks = []

    def make_hook(name: str):
        def _hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                conv_names[name] = out.shape
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=device)
            model(dummy)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    best_name: Optional[str] = None
    best_score = -1
    for name, shape in conv_names.items():
        if len(shape) >= 4:
            h, w = shape[-2], shape[-1]
            score = h * w
            if score > best_score:
                best_score = score
                best_name = name
    return best_name


def get_module_by_name(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    return None


def find_first_conv_layer_name(model: torch.nn.Module) -> Optional[str]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            return name
    return None


def compute_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    target_index: int,
    target_layer: Optional[torch.nn.Module] = None,
) -> Optional[np.ndarray]:
    target_layer = target_layer or pick_last_conv(model)
    if target_layer is None:
        return None

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(_module, _inp, out):
        activations.append(out.detach())

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)  # type: ignore[attr-defined]
    try:
        model.zero_grad(set_to_none=True)
        tensor = x.clone().to(device).requires_grad_(True)
        out = model(tensor)
        score = out[:, target_index].squeeze()
        score.backward()
        if not activations or not gradients:
            return None
        act = activations[-1]
        grad = gradients[-1]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        heat = cam.squeeze().cpu().numpy()
        if heat.max() <= heat.min():
            return None
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat
    except Exception:
        return None
    finally:
        handle_f.remove()
        handle_b.remove()
        model.zero_grad(set_to_none=True)


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    if heatmap is None:
        return base
    cmap = cm.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]
    colored_uint8 = (colored * 255).astype(np.uint8)
    overlay = Image.fromarray(colored_uint8).resize(base.size, Image.BILINEAR)
    base_np = np.asarray(base).astype(np.float32)
    overlay_np = np.asarray(overlay).astype(np.float32)
    blended = (alpha * overlay_np + (1 - alpha) * base_np).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def top_k(probs: np.ndarray, classes: List[str], k: int = 3) -> List[Tuple[str, float]]:
    idx = np.argsort(-probs)[:k]
    return [(classes[i], float(probs[i])) for i in idx]


def render_activation_story(
    activation_map: torch.Tensor,
    probs: np.ndarray,
    classes: List[str],
    top_items: List[Tuple[str, float]],
) -> Optional[plt.Figure]:
    if activation_map is None:
        return None
    try:
        act = activation_map.detach().cpu().numpy()
    except Exception:
        return None
    if act.ndim != 3:
        return None
    channels = act.shape[0]
    if channels == 0:
        return None

    flattened = act.reshape(channels, -1)
    energy = flattened.mean(axis=1)
    order = np.argsort(-energy)
    top_indices = order[: min(3, len(order))]
    selected = act[top_indices]
    aggregated = act.mean(axis=0)

    fig = plt.figure(figsize=(6.8, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
    slot_map = [gs[0, 0], gs[0, 1], gs[1, 0]]
    for i, idx in enumerate(top_indices):
        ax = fig.add_subplot(slot_map[i])
        ax.imshow(selected[i], cmap="magma")
        ax.set_title(f"Channel {idx}", fontsize=9)
        ax.axis("off")
    agg_ax = fig.add_subplot(gs[1, 1])
    agg_ax.imshow(aggregated, cmap="inferno")
    agg_ax.set_title("Mean activation", fontsize=9)
    agg_ax.axis("off")

    ax4 = fig.add_subplot(gs[:, 2])
    ax4.set_title("Neural pulse", fontsize=10)
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
    radius_in = 0.15
    radius_out = 0.9
    center = np.array([0.0, 0.0])
    ax4.scatter([0], [0], s=400, c="#1f2933", marker="o", edgecolor="none")
    for angle, prob, label in zip(angles, probs, classes):
        endpoint = np.array([np.cos(angle), np.sin(angle)]) * radius_out
        color = cm.plasma(prob)
        ax4.plot(
            [center[0], endpoint[0]],
            [center[1], endpoint[1]],
            color=color,
            linewidth=1.5 + prob * 3,
            alpha=0.8,
        )
        ax4.scatter(
            endpoint[0],
            endpoint[1],
            s=250,
            c=[color],
            edgecolors="white",
            linewidths=0.6,
        )
        ax4.text(
            endpoint[0] * 1.1,
            endpoint[1] * 1.1,
            f"{label.upper()}\n{prob * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
        )
    ax4.axis("off")
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)

    fig.suptitle("Inner-layer activation & output pulse", fontsize=12)
    fig.tight_layout()
    return fig


# -------------------------
# Clinical descriptor map
# -------------------------
CONDITION_INFO: Dict[str, Dict[str, str]] = {
    "mel": {
        "title": "Melanoma",
        "summary": "Pigmented malignancy that requires rapid dermatology review. Depth on biopsy determines staging.",
        "care": "Urgent referral for excision biopsy; further surgery and oncology follow-up depend on Breslow depth.",
    },
    "nv": {
        "title": "Melanocytic nevus",
        "summary": "Common mole; most are benign but change in size, shape, or color warrants evaluation.",
        "care": "Routine monitoring; dermoscopy follow-up if atypical features or patient risk factors.",
    },
    "bcc": {
        "title": "Basal cell carcinoma",
        "summary": "Slow-growing keratinocyte cancer; rarely metastasizes but causes local tissue destruction.",
        "care": "Arrange dermatology visit for biopsy and definitive removal (excision, Mohs, or topical therapy).",
    },
    "scc": {
        "title": "Squamous cell carcinoma",
        "summary": "Potentially aggressive keratinocyte cancer; risk of nodal spread especially on head and neck.",
        "care": "Priority referral for biopsy and surgical excision; consider imaging for high-risk sites.",
    },
    "df": {
        "title": "Dermatofibroma",
        "summary": "Benign fibrohistiocytic nodule often on extremities; usually stable in size.",
        "care": "No treatment needed unless symptomatic or atypical; surgical excision is curative if desired.",
    },
    "bkl": {
        "title": "Benign keratosis",
        "summary": "Includes seborrheic keratosis and related lesions; appears stuck-on with waxy surface.",
        "care": "Reassure if diagnosis is secure; cryotherapy or curettage for cosmetic removal.",
    },
    "ak": {
        "title": "Actinic keratosis",
        "summary": "Premalignant scaly lesion on sun-damaged skin; small risk of progressing to SCC.",
        "care": "Field therapy (topical 5-FU, imiquimod) or cryotherapy; emphasize sun protection.",
    },
    "vasc": {
        "title": "Vascular lesion",
        "summary": "Typically hemangioma or angioma; often blanching and benign.",
        "care": "Observation unless symptomatic; laser or surgical removal for bleeding or cosmetic reasons.",
    },
}


DISCLAIMER = (
    "This tool is for demo purposes only. It is not a medical diagnosis. "
    "Always refer to a qualified clinician for definitive assessment."
)


def build_chat_reply(
    prompt: str,
    top_prediction: Optional[str],
    top_prob: Optional[float],
    top_list: Optional[List[Tuple[str, float]]],
) -> str:
    clean_prompt = (prompt or "").strip().lower()
    if not clean_prompt:
        return "Feel free to ask about the lesion categories, risk, or next clinical steps."
    if top_prediction is None:
        return "Upload an image first so I can tailor the guidance."

    info = CONDITION_INFO.get(top_prediction.lower())
    if info is None:
        return "I do not have clinical notes for this label yet, but please seek a clinician's advice."

    lines = [
        f"Our model leans toward **{info['title']}** with approximately {top_prob * 100:.1f}% confidence."
    ]
    if "what" in clean_prompt or "summary" in clean_prompt or "diagnosis" in clean_prompt:
        lines.append(info["summary"])
    if any(term in clean_prompt for term in ("treatment", "care", "next", "do", "best course", "therapy")):
        lines.append(info["care"])
    if any(term in clean_prompt for term in ("other", "else", "differential", "compare")) and top_list:
        formatted = ", ".join(f"{name} ({prob*100:.1f}%)" for name, prob in top_list[:3])
        lines.append(f"Other classes considered: {formatted}.")
    lines.append(DISCLAIMER)
    return "\n\n".join(lines)


def summarise_prediction(top_prediction: str, top_prob: float) -> str:
    info = CONDITION_INFO.get(top_prediction.lower())
    title = info["title"] if info else top_prediction
    summary = info["summary"] if info else "Clinical guidance pending."
    care = info["care"] if info else "Discuss findings with a dermatologist."
    return (
        f"**Model insight**\n\n"
        f"Primary suggestion: **{title}** ({top_prob * 100:.1f}% confidence).\n\n"
        f"{summary}\n\n"
        f"Typical next steps: {care}\n\n"
        f"{DISCLAIMER}"
    )


# -------------------------
# Streamlit layout
# -------------------------
st.set_page_config(page_title="MNKAI Telederm Demo", page_icon="🩺", layout="wide")
st.title("MNKAI Teledermatology Demo")
st.caption("Upload a dermatoscopic or clinical image to explore class probabilities and attention maps.")

with st.sidebar:
    st.subheader("Model inputs")
    cfg_path = st.text_input("Config (.yaml)", "configs/mnkai_isic.yaml")
    ckpt_path = st.text_input("Checkpoint (.pt)", "checkpoints/run_final/best.pt")
    model_name = st.text_input("Model name", "convnextv2_nano.fcmae")
    img_size = st.number_input("Image size", value=320, min_value=160, max_value=768, step=16)
    tta = st.checkbox("Enable flip TTA", value=True)
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    top_k_display = st.slider("Top classes to display", min_value=3, max_value=8, value=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Device: **{device.type.upper()}**")

# Load config / classes
classes: Optional[List[str]] = None
try:
    cfg = load_config(cfg_path)
    class_cfg = cfg.get("classes", {}).get("names")
    if class_cfg:
        classes = [str(c).strip().lower() for c in class_cfg]
except Exception as exc:
    st.warning(f"Could not read config: {exc}")

# Load model
try:
    model, classes_from_ckpt = load_model_ckpt(
        model_name,
        num_classes=len(classes) if classes else 8,
        ckpt_path=ckpt_path,
        device=device,
    )
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

if classes_from_ckpt:
    classes = [str(c).strip().lower() for c in classes_from_ckpt]
if not classes:
    classes = ["mel", "nv", "bcc", "scc", "df", "bkl", "ak", "vasc"]

st.success(f"Loaded **{model_name}** with {len(classes)} output classes.")

cam_session_key = "cam_layer_name"
cam_model_key = "cam_layer_model"
cam_size_key = "cam_layer_size"

refresh_cam = (
    st.session_state.get(cam_session_key) is None
    or st.session_state.get(cam_model_key) != model_name
    or st.session_state.get(cam_size_key) != int(img_size)
)

if refresh_cam:
    layer_name = find_cam_layer_name(model, device, int(img_size))
    st.session_state[cam_session_key] = layer_name
    st.session_state[cam_model_key] = model_name
    st.session_state[cam_size_key] = int(img_size)
else:
    layer_name = st.session_state.get(cam_session_key)

cam_target_layer = get_module_by_name(model, layer_name) if layer_name else None

# determine activation layer (first conv)
act_session_key = "activation_layer_name"
act_model_key = "activation_layer_model"
activation_layer_name = st.session_state.get(act_session_key)
if (
    activation_layer_name is None
    or st.session_state.get(act_model_key) != model_name
):
    activation_layer_name = find_first_conv_layer_name(model)
    st.session_state[act_session_key] = activation_layer_name
    st.session_state[act_model_key] = model_name
activation_layer = get_module_by_name(model, activation_layer_name) if activation_layer_name else None

# inform user if fallback
if cam_target_layer is None:
    st.info(
        "Using fallback Grad-CAM target. Heatmap quality may vary for transformer-style backbones.",
        icon="ℹ️",
    )


# -------------------------
# Prediction workflow
# -------------------------
upload_col, result_col = st.columns([1.1, 1.5], gap="large")
uploaded_image: Optional[Image.Image] = None

with upload_col:
    st.subheader("Image upload")
    uploaded = st.file_uploader("Select a dermatoscopic or clinical image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        uploaded_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

with result_col:
    st.subheader("Model output")
    if uploaded_image is None:
        st.info("Upload an image to see predictions, Grad-CAM, and assistant insights.")

last_top: Optional[List[Tuple[str, float]]] = None
heatmap_img: Optional[Image.Image] = None

if uploaded_image is not None:
    tensor, resized = preprocess_pil(uploaded_image, img_size=int(img_size))
    batch = tensor.unsqueeze(0)
    activation_holder: Dict[str, torch.Tensor] = {}
    act_handle = None
    if activation_layer is not None:
        def _act_hook(_module, _inp, out):
            try:
                activation_holder["map"] = out.detach().cpu()
            except Exception:
                pass
        act_handle = activation_layer.register_forward_hook(_act_hook)
    try:
        start = time.time()
        probabilities = infer_probs(model, batch, device=device, tta=tta)
        latency = (time.time() - start) * 1000.0
    finally:
        if act_handle is not None:
            act_handle.remove()

    activation_tensor = activation_holder.get("map")
    if activation_tensor is not None and activation_tensor.dim() == 4:
        activation_tensor = activation_tensor.squeeze(0)

    provisional_top = top_k(probabilities, classes, k=top_k_display)
    activation_fig = render_activation_story(activation_tensor, probabilities, classes, provisional_top)

    last_top = provisional_top
    primary_class, primary_prob = last_top[0]

    if show_cam:
        heat = compute_gradcam(
            model,
            batch,
            device,
            target_index=classes.index(primary_class),
            target_layer=cam_target_layer,
        )
        if heat is not None:
            heatmap_img = overlay_heatmap(resized, heat)

    with result_col:
        st.markdown(f"**Primary class:** {primary_class.upper()} ({primary_prob * 100:.2f}% confidence)")
        st.caption(f"Inference latency: {latency:.1f} ms | TTA {'on' if tta else 'off'}")

        metrics_cols = st.columns(min(3, len(last_top)))
        for col, (name, prob) in zip(metrics_cols, last_top[:3]):
            with col:
                st.metric(label=name.upper(), value=f"{prob*100:.2f} %")

        table_df = pd.DataFrame(last_top, columns=["class", "probability"])
        st.dataframe(
            table_df.assign(probability=lambda d: (d["probability"] * 100).map("{:.2f}%".format)),
            hide_index=True,
            use_container_width=True,
        )

        chart_df = pd.DataFrame({"class": classes, "prob": probabilities})
        st.bar_chart(chart_df.set_index("class"), use_container_width=True)

        if activation_fig is not None:
            st.markdown("#### Neural activation vignette")
            st.pyplot(activation_fig, use_container_width=True)
            plt.close(activation_fig)
        elif activation_layer is None:
            st.info("Activation visualisation unavailable for this model.", icon="ℹ️")

    with upload_col:
        if heatmap_img is not None:
            st.image(heatmap_img, caption="Grad-CAM overlay", use_column_width=True)
        elif show_cam:
            st.warning("Grad-CAM could not be generated for this architecture.")

    # Update chat state with summary
    summary_text = summarise_prediction(primary_class, primary_prob)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.session_state.get("last_summary_class") != primary_class:
        st.session_state.chat_history.append({"role": "assistant", "content": summary_text})
        st.session_state.last_summary_class = primary_class
    st.session_state.last_prediction = primary_class
    st.session_state.last_probability = primary_prob
    st.session_state.last_top_list = last_top


# -------------------------
# Assistant chat
# -------------------------
st.markdown("### Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hello! Ask about what the predicted diagnosis means, how confident the model is, "
                "or what next clinical steps usually look like."
            ),
        }
    ]
    st.session_state.last_prediction = None
    st.session_state.last_probability = None
    st.session_state.last_top_list = None

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the findings or next steps...")
if prompt is not None:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    reply = build_chat_reply(
        prompt,
        st.session_state.get("last_prediction"),
        st.session_state.get("last_probability"),
        st.session_state.get("last_top_list"),
    )
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

st.markdown("---")
st.caption("Run locally with: `streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 7860`")
