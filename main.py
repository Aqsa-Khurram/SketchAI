import numpy as np
import random
from sklearn.decomposition import PCA
import tkinter as tk

from load_data import load_data, CATEGORIES
from search import (
    create_class_prototypes,
    create_centroids,
    a_star_classification,
    a_star_top_k,
)
from gui import SketchApp, AIDrawApp, ModeSelectApp, TimerModeSelectApp
from preprocess import preprocess_image


# ─────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────
def augment_samples(samples, n_augmented=300):
    """
    Light augmentation by adding small Gaussian noise to existing samples.
    Helps confusable classes (sun/circle, flower/star) generalise better.
    Returns original + augmented samples concatenated.
    """
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.015, size=(n_augmented, samples.shape[1]))
    idx = rng.integers(0, len(samples), size=n_augmented)
    augmented = samples[idx] + noise

    # Re-normalize augmented samples
    norms = np.linalg.norm(augmented, axis=1, keepdims=True) + 1e-8
    augmented = augmented / norms

    return np.vstack([samples, augmented])


# ─────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────
def train_model():
    # ── Hand-picked sample indices per category ───────────────────────
    # Run browse_dataset.py to visually browse the .npy files and find
    # good indices, then paste them here.  Any category not listed falls
    # back to the first 50 samples automatically.
    PICKED_SAMPLES = {
        "circle": [2],
        "square": [12],
        "triangle": [3],
        "tree": [61, 133, 472],
        "house": [14, 48, 394],
        "star": [11, 39],
        "cloud": [70],
        "moon": [48],
        "sun": [412],
        "book": [43, 89, 111, 124],
        "beach": [7],
        "clock": [0, 40, 97],
        "fish": [2, 105],
        "airplane": [42, 149, 201, 202],
        "leaf": [0, 19, 90],
        "calendar": [8, 108],
        "pencil": [45, 47, 103],
        "guitar": [14, 39, 248],
        "cup": [7, 32, 126],
        "flower": [29, 36, 54],
        "cat": [3, 150],
        "car": [48, 86, 194],
        "umbrella": [41, 70, 128],
        "bicycle": [40],
        "ladder": [10, 145, 174],
        "key": [4, 49, 79, 96],
        "dog": [2, 342],
        "lightning": [3, 40, 140],
        "chair": [13, 151, 190],
        "hat": [23, 42, 104],
    }

    # --- Load raw pixel data for AI Draw display (before ANY normalisation) ---
    import os
    pixel_prototypes = {}
    for category in CATEGORIES:
        path = os.path.join("data", category + ".npy")
        try:
            raw = np.load(path)                           # uint8, shape (N, 784)
            indices = PICKED_SAMPLES.get(category, list(range(min(50, len(raw)))))
            chosen = raw[indices].astype(np.float32) / 255.0
            pixel_prototypes[category] = [chosen[i].reshape(28, 28)
                                          for i in range(len(chosen))]
        except Exception:
            pixel_prototypes[category] = [np.zeros((28, 28), dtype=np.float32)]

    dataset = load_data(samples_per_class=2000)

    if not dataset:
        raise RuntimeError("Dataset is empty! Check your data path and .npy files.")

    train_data = {}
    test_data = {}

    for label, samples in dataset.items():
        split_index = int(0.75 * len(samples))
        train_data[label] = samples[:split_index]
        test_data[label] = samples[split_index:]

    # --- PCA: 150 components captures more variance than 100 ---
    # Especially helpful for fine-grained classes like flower vs star
    all_train_samples = np.vstack(list(train_data.values()))
    pca = PCA(n_components=150, random_state=42)
    pca.fit(all_train_samples)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # --- Transform train ---
    for label in train_data:
        transformed = pca.transform(train_data[label])
        norms = np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8
        train_data[label] = transformed / norms

    # --- Augment visually similar / hard classes ---
    # These pairs are most commonly confused:
    HARD_CLASSES = {"sun", "circle", "flower", "star", "clock", "moon"}
    for label in HARD_CLASSES:
        if label in train_data:
            train_data[label] = augment_samples(train_data[label], n_augmented=300)
            print(f"Augmented {label}: {len(train_data[label])} samples")

    # --- Transform test ---
    for label in test_data:
        transformed = pca.transform(test_data[label])
        norms = np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8
        test_data[label] = transformed / norms

    # --- Build search structures ---
    prototypes = create_class_prototypes(train_data)
    # 35 centroids per class — better coverage for diverse classes
    centroids = create_centroids(train_data, centroids_per_class=35)

    return pca, prototypes, centroids, test_data, pixel_prototypes


# ─────────────────────────────────────────
# Prediction Wrapper
# ─────────────────────────────────────────
def predict(input_vector, pca, prototypes, centroids):
    x = pca.transform([input_vector])[0]
    x = x / (np.linalg.norm(x) + 1e-8)
    label, _ = a_star_classification(x, prototypes, centroids)
    return label


def predict_top3(input_vector, pca, prototypes, centroids):
    """Returns list of (label, distance) for top 3 predictions."""
    x = pca.transform([input_vector])[0]
    x = x / (np.linalg.norm(x) + 1e-8)
    return a_star_top_k(x, prototypes, centroids, k=3)


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
def run_evaluation(pca, prototypes, centroids, test_data, total_tests=500):
    correct = 0
    labels = list(test_data.keys())
    confusion = {}  # track which pairs are confused most

    for _ in range(total_tests):
        true_label = random.choice(labels)
        sample = random.choice(test_data[true_label])

        predicted_label, _ = a_star_classification(sample, prototypes, centroids)

        if predicted_label == true_label:
            correct += 1
        else:
            key = (true_label, predicted_label)
            confusion[key] = confusion.get(key, 0) + 1

    accuracy = (correct / total_tests) * 100
    print(f"\nTotal Tests : {total_tests}")
    print(f"Correct     : {correct}")
    print(f"Accuracy    : {accuracy:.2f}%")

    # Show top confusions
    top_confusions = sorted(confusion.items(), key=lambda x: -x[1])[:10]
    print("\nTop confusions (true → predicted : count):")
    for (true, pred), count in top_confusions:
        print(f"  {true:12s} → {pred:12s} : {count}")


# ─────────────────────────────────────────
# GUI  (mode selection → human draws OR ai draws)
# ─────────────────────────────────────────
def run_gui(pca, prototypes, centroids, pixel_prototypes):
    """
    Show a mode-selection screen.
    • Human Draws  → original SketchApp (AI guesses)
    • AI Draws     → AIDrawApp (human guesses)
    A "Switch Mode" button in each app returns to this screen.
    """
    root = tk.Tk()

    # ── helpers to clear and rebuild ──────────────────────────────
    def clear_root():
        for w in root.winfo_children():
            w.destroy()

    def launch_mode_select():
        clear_root()
        ModeSelectApp(root,
                      on_human_draws=launch_human_draws,
                      on_ai_draws=launch_ai_draws)

    def launch_human_draws():
        clear_root()

        def gui_predict(vector):
            top3 = predict_top3(vector, pca, prototypes, centroids)
            lines = []
            for rank, (label, dist) in enumerate(top3, 1):
                confidence = max(0, (1 - dist) * 100)
                lines.append(f"#{rank} {label}  ({confidence:.0f}%)")
            return "\n".join(lines)

        SketchApp(root, gui_predict, on_switch_mode=launch_mode_select)

    def launch_ai_draws():
        clear_root()
        TimerModeSelectApp(
            root,
            on_timed   = lambda: _start_ai_draws(timed=True),
            on_relaxed = lambda: _start_ai_draws(timed=False),
            on_back    = launch_mode_select,
        )

    def _start_ai_draws(timed):
        clear_root()
        AIDrawApp(root,
                  categories=CATEGORIES,
                  pixel_prototypes=pixel_prototypes,
                  on_switch_mode=launch_mode_select,
                  timed_mode=timed)

    launch_mode_select()
    root.mainloop()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    print("Training model...")
    pca, prototypes, centroids, test_data, pixel_prototypes = train_model()
    print("Model ready!\n")

    mode = "gui"   # "eval" or "gui"

    if mode == "eval":
        run_evaluation(pca, prototypes, centroids, test_data)
    elif mode == "gui":
        run_gui(pca, prototypes, centroids, pixel_prototypes)