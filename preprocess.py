import numpy as np
from PIL import Image, ImageOps


class ImagePreprocessor:
    """
    Converts a live-canvas drawing (RGB numpy array) into the
    unit-normalised 784-d vector expected by the classifier.

    Pipeline
    --------
    1. Grayscale + invert  (white strokes on black bg, like QuickDraw)
    2. Tight crop + square pad with margin
    3. Lanczos resize to 28 × 28
    4. Normalise to [0, 1]
    5. Unit-norm flatten
    """

    def __init__(self, threshold: int = 10, margin_ratio: float = 1 / 6):
        """
        Parameters
        ----------
        threshold    : pixel brightness threshold for detecting strokes
        margin_ratio : padding added around the tight crop (fraction of side)
        """
        self.threshold    = threshold
        self.margin_ratio = margin_ratio

    # ── Public ─────────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray  – H × W × 3 (or H × W) uint8 canvas image

        Returns
        -------
        np.ndarray  – flat float32 vector of length 784, unit-normalised
        """
        canvas = self._to_canvas(image)
        arr28  = self._resize_28(canvas)
        return self._normalise(arr28)

    # ── Private ────────────────────────────────────────────────────
    def _to_canvas(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image).convert("L")
        img = ImageOps.invert(img)                 # black bg → white strokes
        arr = np.array(img, dtype=np.float32)
        return self._crop_and_pad(arr)

    def _crop_and_pad(self, arr: np.ndarray) -> np.ndarray:
        rows = np.any(arr > self.threshold, axis=1)
        cols = np.any(arr > self.threshold, axis=0)

        if not (rows.any() and cols.any()):
            return arr                             # blank canvas → return as-is

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

        h, w     = cropped.shape
        side     = max(h, w)
        margin   = max(4, int(side * self.margin_ratio))
        pad_side = side + 2 * margin

        canvas            = np.zeros((pad_side, pad_side), dtype=np.float32)
        y_off             = margin + (side - h) // 2
        x_off             = margin + (side - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = cropped
        return canvas

    @staticmethod
    def _resize_28(arr: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(arr)
        pil_28  = pil_img.resize((28, 28), Image.LANCZOS)
        return np.array(pil_28, dtype=np.float32)

    @staticmethod
    def _normalise(arr28: np.ndarray) -> np.ndarray:
        arr28 /= 255.0
        flat  = arr28.flatten()
        norm  = np.linalg.norm(flat) + 1e-8
        return flat / norm


# ── Module-level convenience function (keeps old call-sites working) ──
_default_preprocessor = ImagePreprocessor()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    return _default_preprocessor.process(image)