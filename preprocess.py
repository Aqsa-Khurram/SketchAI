import numpy as np
from PIL import Image, ImageOps, ImageFilter


def preprocess_image(image):
    """
    Preprocess a live-canvas drawing to match QuickDraw format.

    Pipeline:
      1. Grayscale + invert  (drawing = white on black, like QuickDraw)
      2. Tight crop + square pad with margin
      3. Lanczos resize to 28×28
      4. Normalize to [0,1]  (same as load_data: raw/255)
      5. Unit-norm flatten   (same as load_data)

    The key insight: QuickDraw drawings are centered with white strokes
    on a black background.  Matching this convention is the biggest
    single factor in closing the train/inference distribution gap.
    """
    img = Image.fromarray(image).convert("L")
    img = ImageOps.invert(img)          # black bg → white strokes

    arr = np.array(img, dtype=np.float32)

    # ── Tight crop around strokes ──────────────────────────────────
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

        h, w     = cropped.shape
        side     = max(h, w)
        margin   = max(4, side // 6)          # ~16% padding on each side
        pad_side = side + 2 * margin

        canvas            = np.zeros((pad_side, pad_side), dtype=np.float32)
        y_off             = margin + (side - h) // 2
        x_off             = margin + (side - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = cropped
    else:
        canvas = arr                          # blank → return zeros

    # ── Resize 28×28 ───────────────────────────────────────────────
    pil_c  = Image.fromarray(canvas)
    pil_28 = pil_c.resize((28, 28), Image.LANCZOS)
    arr28  = np.array(pil_28, dtype=np.float32)

    # ── Normalize: [0,1] then unit-norm ────────────────────────────
    arr28 /= 255.0
    flat   = arr28.flatten()
    norm   = np.linalg.norm(flat) + 1e-8
    return flat / norm