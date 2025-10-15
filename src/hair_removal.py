# src/hair_removal.py
import cv2
import numpy as np
from loguru import logger

def _resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img, 1.0
    scale = max_side / float(s)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale

def remove_hair_bgr(img_bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    """
    Simple, fast hair removal:
      - downscale (max_side)
      - black-hat morph to detect hair
      - TELEA inpaint
      - upsample back if needed
    """
    try:
        small, scale = _resize_max_side(img_bgr, max_side)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, th = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(small, th, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        if scale != 1.0:
            inpainted = cv2.resize(inpainted, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return inpainted
    except cv2.error as e:
        logger.error(f"[hair] OpenCV error during inpaint — {e}. Returning original (no hair removal).")
        return img_bgr
    except Exception as e:
        logger.error(f"[hair] Unexpected error: {e}. Returning original.")
        return img_bgr
