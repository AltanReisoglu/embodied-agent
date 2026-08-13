"""Turning rendered frames into something a chat-completions API accepts."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw


def to_data_uri(rgb: np.ndarray, *, quality: int = 88) -> str:
    """Encode an RGB array as a base64 JPEG data URI (the `image_url.url` field)."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def draw_pixel_grid(rgb: np.ndarray, *, spacing: int = 80) -> np.ndarray:
    """Overlay a labelled pixel grid.

    VLMs name image positions far more accurately when the coordinate frame is drawn on
    the image instead of left implicit, and every pixel the model reports feeds straight
    into `measure()` for back-projection.
    """
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for x in range(spacing, w, spacing):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 60), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 0, 220))
    for y in range(spacing, h, spacing):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 60), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 0, 220))

    return np.asarray(img)


def crop_and_upscale(
    rgb: np.ndarray, bbox: tuple[int, int, int, int], *, min_side: int = 384
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop `bbox` = (x0, y0, x1, y1), clamped to the frame, and upscale small crops.

    Also returns the clamped bbox so the caller can tell the model what it actually got.
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((max(0, min(int(x0), w - 1)), max(0, min(int(x1), w))))
    y0, y1 = sorted((max(0, min(int(y0), h - 1)), max(0, min(int(y1), h))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError(f"bbox {bbox} is degenerate after clamping to the {w}x{h} frame")

    crop = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[y0:y1, x0:x1])
    scale = max(1.0, min_side / min(crop.size))
    if scale > 1.0:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    return np.asarray(crop), (x0, y0, x1, y1)
