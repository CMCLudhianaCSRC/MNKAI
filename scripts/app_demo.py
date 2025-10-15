import os, json
from typing import List, Tuple
import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import timm

# ---------------------------
# CONFIG — EDIT THESE
# ---------------------------
CHECKPOINT = r"checkpoints\best.pt"   # <- point to your best checkpoint
CLASSES = ["mel","nv","bcc","scc","df","bkl","ak","vasc"]  # keep order identical to training
IMG_SIZE = 320
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
MODEL_NAME = "tf_efficientnet_b3_ns"
DEFAULT_TTA = True

DISCLAIMER = (
    "MNKAI is a research demo, **not** a medical device. "
    "Do not use for diagnosis or treatment. Always consult a clinician."
)

# ---------------------------
# Torch helpers
# ---------------------------
def to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img).astype("float32") / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2,0,1))  # HWC->CHW
    return torch.from_numpy(x)

def softmax_topk(logits: torch.Tensor, k:int=3):
    p = F.softmax(logits, dim=-1).detach().cpu().numpy().flatten()
    idx = p.argsort()[::-1][:k]
    return [(CLASSES[i], float(p[i])) for i in idx]

def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(checkpoint_path, map_location="cpu")  # keep weights_only=False for older PyTorch
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.",""): v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    try:
        model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model, device

MODEL, DEVICE = load_model(CHECKPOINT)

@torch.no_grad()
def predict_image(img: Image.Image, tta: bool=True):
    if img is None:
        return "<span>No image.</span>", {}
    MODEL.eval()
    x = to_tensor(img).unsqueeze(0).to(DEVICE, non_blocking=True)
    try:
        x = x.to(memory_format=torch.channels_last)
    except Exception:
        pass

    def fwd(inp):
        with torch.autocast(device_type="cuda", enabled=(DEVICE.type=="cuda")):
            return MODEL(inp)

    logits = fwd(x)
    if tta:
        logits_h = fwd(torch.flip(x, dims=[3]))
        logits_v = fwd(torch.flip(x, dims=[2]))
        logits   = (logits + logits_h + logits_v) / 3.0

    topk = softmax_topk(logits, k=min(5, len(CLASSES)))
    labels = [n for n,_ in topk]
    probs  = [p for _,p in topk]
    risk_txt = "High risk" if (labels and labels[0] in ("mel","bcc","scc")) else "Low–Moderate risk"
    risk_html = (
        f'<span style="padding:6px 10px;border-radius:9999px;'
        f'background:linear-gradient(90deg,#f43f5e,#ef4444);color:white;font-weight:700">{risk_txt}</span>'
        if "High" in risk_txt else
        f'<span style="padding:6px 10px;border-radius:9999px;'
        f'background:linear-gradient(90deg,#06b6d4,#22c55e);color:white;font-weight:700">{risk_txt}</span>'
    )
    # Use JSON to avoid the gr.Label schema bug
    probs_map = {labels[i]: round(probs[i], 4) for i in range(len(labels))}
    return risk_html, probs_map

# ---------------------------
# Tiny offline “LLM” helper
# ---------------------------
SYN_DB = [
    {"label":"tinea (fungal infection)","keys":["ring","itch","scaly","annular","groin","foot","armpit"],
     "adv":"Keep area dry; topical antifungal may help. See clinician if widespread or not better in 2 weeks."},
    {"label":"eczema (dermatitis)","keys":["itch","dry","patch","flexural","elbow","knee","long-standing"],
     "adv":"Use bland moisturizer; avoid harsh soaps. Steroid creams require clinician advice."},
    {"label":"psoriasis","keys":["thick","silvery scale","elbow","knee","scalp","plaque","chronic"],
     "adv":"Chronic—needs clinician plan. Moisturize; avoid skin trauma."},
    {"label":"melanoma risk","keys":["changing mole","irregular border","very dark","bleeding","asymmetry","rapid growth"],
     "adv":"Urgent dermatologist evaluation recommended."},
    {"label":"bacterial infection","keys":["pus","painful","warm","red streak","fever","boil","abscess"],
     "adv":"Seek clinician; may require antibiotics."}
]

def dermassist_chat(user_text: str) -> str:
    from rapidfuzz import fuzz
    q = (user_text or "").lower().strip()
    if not q:
        return "Tell me symptom + location + duration. Example: 'itchy round patch on arm for 3 weeks'."
    best, best_sc = None, -1
    for item in SYN_DB:
        score = 0
        for k in item["keys"]:
            if k in q: score += 12
        score += fuzz.partial_ratio(q, " ".join(item["keys"])) * 0.2
        if score > best_sc:
            best, best_sc = item, score
    return (
        f"**Most likely bucket:** {best['label'].title()}\n\n"
        f"**Typical clues:** {', '.join(best['keys'][:6])}…\n\n"
        f"**Next steps:** {best['adv']}\n\n> {DISCLAIMER}"
    )

# ---------------------------
# Gradio UI
# ---------------------------
CSS = """
.gradio-container {font-family: Inter, ui-sans-serif, system-ui;}
footer {visibility: hidden}
#hero {background: radial-gradient(1200px 600px at 20% 20%, #1f2937, #0b0f17);}
"""

def build_interface():
    with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="indigo", font=[gr.themes.GoogleFont("Inter")])) as demo:
        gr.Markdown(f"""
<div id="hero" style="padding:18px;border-radius:16px;color:#e5e7eb">
  <h1 style="margin:6px 0;font-weight:800;font-size:28px;">MNKAI — Tele-Derm Demo</h1>
  <div style="opacity:.85">{DISCLAIMER}</div>
</div>
""")

        with gr.Tab("🖼️ Lesion Classifier"):
            with gr.Row():
                img_in  = gr.Image(type="pil", label="Upload lesion photo", height=380)
                with gr.Column():
                    tta = gr.Checkbox(value=DEFAULT_TTA, label="TTA (flips)")
                    btn = gr.Button("Predict", variant="primary", size="lg")
                    risk = gr.HTML()
                    topk_tbl = gr.JSON(label="Top-k probabilities (JSON)")

            btn.click(lambda im, t: predict_image(im, bool(t)),
                      inputs=[img_in, tta],
                      outputs=[risk, topk_tbl])

        with gr.Tab("💬 DermAssist Chat"):
            chat_in = gr.Textbox(lines=3, placeholder="e.g., 'itchy round patch on arm for 3 weeks'")
            chat_btn = gr.Button("Ask", variant="primary")
            chat_out = gr.Markdown()
            chat_btn.click(dermassist_chat, inputs=chat_in, outputs=chat_out)

        gr.Markdown("<div style='opacity:.7;margin-top:16px'>Runs locally. No data leaves your device.</div>")
    return demo

if __name__ == "__main__":
    demo = build_interface()
    # Prefer localhost; auto-fallback to share if blocked.
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except Exception as e:
        print("[WARN] Localhost blocked or unavailable. Falling back to share=True. Error:", e)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
