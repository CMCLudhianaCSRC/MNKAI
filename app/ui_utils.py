# app/ui_utils.py
import numpy as np
from PIL import Image

def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)

def overlay_heatmap(np_rgb: np.ndarray, grayscale_cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    try:
        import cv2
        heat = (np.clip(grayscale_cam, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
        heat = cv2.resize(heat, (np_rgb.shape[1], np_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        over = (alpha * heat + (1 - alpha) * np_rgb).astype(np.uint8)
        return Image.fromarray(over)
    except Exception:
        # colorize without cv2
        h = (np.clip(grayscale_cam, 0, 1) * 255).astype(np.uint8)
        heat3 = np.stack([h, np.zeros_like(h), 255 - h], axis=-1)
        heat3 = (alpha * heat3 + (1 - alpha) * np_rgb).astype(np.uint8)
        return Image.fromarray(heat3)
