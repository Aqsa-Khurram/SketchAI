import numpy as np
import os

DATA_PATH = "data"

CATEGORIES = [
    "circle", "square", "triangle", "tree", "house",
    "star", "cloud", "moon", "sun", "book",
    "beach", "clock", "fish", "airplane", "leaf",
    "calendar", "pencil", "guitar", "cup", "flower",
    "cat", "car", "umbrella", "bicycle", "ladder",
    "key", "dog", "lightning", "chair", "hat"
]

def load_data(samples_per_class=2000):
    data = {}

    for category in CATEGORIES:
        path = os.path.join(DATA_PATH, category + ".npy")
        print(f"Loading {category}...")

        try:
            raw = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {path}: {e}")

        if raw.shape[0] < samples_per_class:
            print(f"Warning: only {raw.shape[0]} samples available for {category}")

        raw = raw[:samples_per_class]
        raw = raw / 255.0

        # Normalize to unit vectors
        norm = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        if np.any(norm == 0):
            raise ValueError(f"Zero vector detected in {category} data")
        raw = raw / norm

        data[category] = raw
        print(category, "shape:", raw.shape)

    return data