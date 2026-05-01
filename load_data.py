import numpy as np
import os

CATEGORIES = [
    "circle", "square", "triangle", "tree", "house",
    "star", "cloud", "moon", "sun", "book",
    "beach", "clock", "fish", "airplane", "leaf",
    "calendar", "pencil", "guitar", "cup", "flower",
    "cat", "car", "umbrella", "bicycle", "ladder",
    "key", "dog", "lightning", "chair", "hat"
]


class DataLoader:
    """Loads and normalises QuickDraw .npy datasets from disk."""

    DEFAULT_DATA_PATH = "data"

    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path

    # ── Public ─────────────────────────────────────────────────────
    def load(self, samples_per_class: int = 2000) -> dict:
        """
        Load all categories from .npy files.

        Returns
        -------
        dict[str, np.ndarray]  – label → float32 array of shape (N, 784),
                                  unit-normalised rows in [0, 1].
        """
        data = {}
        for category in CATEGORIES:
            data[category] = self._load_category(category, samples_per_class)
        return data

    def load_raw_pixels(
        self,
        picked_indices: dict,
        samples_per_class: int = 50,
    ) -> dict:
        """
        Load raw pixel arrays (28×28 float32) for a set of hand-picked
        indices per category.  Used to build AI-draw prototypes.

        Parameters
        ----------
        picked_indices : dict[str, list[int]]
            Category → list of row indices to use.
            Categories not in the dict fall back to the first
            *samples_per_class* rows.
        """
        pixel_prototypes = {}
        for category in CATEGORIES:
            path = os.path.join(self.data_path, category + ".npy")
            try:
                raw = np.load(path)                        # uint8, (N, 784)
                indices = picked_indices.get(
                    category, list(range(min(samples_per_class, len(raw))))
                )
                chosen = raw[indices].astype(np.float32) / 255.0
                pixel_prototypes[category] = [
                    chosen[i].reshape(28, 28) for i in range(len(chosen))
                ]
            except Exception:
                pixel_prototypes[category] = [np.zeros((28, 28), dtype=np.float32)]
        return pixel_prototypes

    # ── Private ────────────────────────────────────────────────────
    def _load_category(self, category: str, samples_per_class: int) -> np.ndarray:
        path = os.path.join(self.data_path, category + ".npy")
        print(f"Loading {category}...")

        try:
            raw = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except Exception as exc:
            raise RuntimeError(f"Error loading {path}: {exc}")

        if raw.shape[0] < samples_per_class:
            print(f"Warning: only {raw.shape[0]} samples available for {category}")

        raw = raw[:samples_per_class].astype(np.float32) / 255.0

        norm = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        if np.any(norm == 0):
            raise ValueError(f"Zero vector detected in {category} data")

        raw = raw / norm
        print(category, "shape:", raw.shape)
        return raw


# ── Module-level convenience function (keeps old call-sites working) ──
def load_data(samples_per_class: int = 2000, data_path: str = DataLoader.DEFAULT_DATA_PATH) -> dict:
    return DataLoader(data_path).load(samples_per_class)