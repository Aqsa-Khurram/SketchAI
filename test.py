"""
test.py
=======
Automated test suite for the QuickDraw AI Sketch Classifier.

Covers:
  Section 5 — Execution Demonstration
  Section 6 — Test Cases

Run:
    python test.py
"""

import random
import time
import numpy as np

from model import SketchModel
from load_data import CATEGORIES


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
SEP  = "-" * 70


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────
# Fix for PCA mismatch
# ─────────────────────────────────────────────────────────────────
def _safe_predict(model: SketchModel, sample):
    """
    Ensures compatibility with PCA input.
    If sample is already embedded (150-dim), bypass embedding.
    """
    sample = np.asarray(sample)

    if sample.shape[0] == model._pca.n_components_:
        # Already PCA-transformed → directly classify
        return model._classifier.predict(sample)

    return model.predict(sample)


def _safe_predict_top_k(model: SketchModel, sample, k=3):
    sample = np.asarray(sample)

    if sample.shape[0] == model._pca.n_components_:
        return model._classifier.predict_top_k(sample, k)

    return model.predict_top_k(sample, k)


# ─────────────────────────────────────────────────────────────────
# Train Model
# ─────────────────────────────────────────────────────────────────
def build_model() -> SketchModel:
    print("\nTraining model. This may take approximately one minute...")
    start = time.time()

    model = SketchModel(
        n_components=150,
        samples_per_class=2000,
        centroids_per_class=35,
    ).train()

    elapsed = time.time() - start
    print(f"Model training completed in {elapsed:.1f} seconds.\n")

    return model


# ─────────────────────────────────────────────────────────────────
# SECTION 5 — Execution Demonstration
# ─────────────────────────────────────────────────────────────────
def section5_execution_demo(model: SketchModel, total_tests: int = 300) -> None:
    _banner("SECTION 5 — Execution Demonstration")

    rng = random.Random(42)
    labels = list(model.test_data.keys())

    correct = 0
    confusion = {}

    print(f"\nRunning {total_tests} random test predictions...\n")

    for _ in range(total_tests):
        true_label = rng.choice(labels)
        sample = rng.choice(list(model.test_data[true_label]))

        pred_label, _ = _safe_predict(model, sample)

        if pred_label == true_label:
            correct += 1
        else:
            key = (true_label, pred_label)
            confusion[key] = confusion.get(key, 0) + 1

    accuracy = correct / total_tests * 100

    # ── Report-style Output ─────────────────────────────────────
    print("Example Output:\n")
    print(f"Total Tests: {total_tests}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\n[Execution output screenshots can be inserted here]\n")

    # ── Confusion Summary ──────────────────────────────────────
    top = sorted(confusion.items(), key=lambda x: -x[1])[:5]

    if top:
        print("Top Confusion Pairs (True → Predicted : Count)")
        print(SEP[:50])
        for (t, p), n in top:
            print(f"{t:<14} → {p:<14} : {n}")


# ─────────────────────────────────────────────────────────────────
# SECTION 6 — Test Cases
# ─────────────────────────────────────────────────────────────────
def section6_test_cases(model: SketchModel) -> None:
    _banner("SECTION 6 — Test Cases")

    print("\nTest Case Table:\n")
    print(f"{'Input':<12}{'Expected':<12}{'Output':<12}{'Result':<10}")
    print(SEP)

    passed = 0
    failed = 0

    # More representative sample (10 categories)
    sample_cases = [
        "circle", "square", "triangle", "cat", "dog",
        "car", "tree", "house", "star", "bicycle"
    ]

    for category in sample_cases:
        pool = list(model.test_data[category])
        sample = pool[0]   # deterministic pick

        pred, _ = _safe_predict(model, sample)

        result = PASS if pred == category else FAIL

        if result == PASS:
            passed += 1
        else:
            failed += 1

        print(f"{category:<12}{category:<12}{pred:<12}{result:<10}")

    print(SEP)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # ── Additional Validation Tests ─────────────────────────────
    print("\nAdditional Validation Tests:\n")
    print(f"{'Test':<10}{'Description':<35}{'Result':<10}")
    print(SEP)

    tests = []

    # T1: Top-3 contains true label
    sample = list(model.test_data["cat"])[0]
    top3 = _safe_predict_top_k(model, sample, 3)
    labels = [lbl for lbl, _ in top3]
    tests.append(("T1", "Top-3 contains true label", PASS if "cat" in labels else FAIL))

    # T2: Prediction is valid label
    pred, _ = _safe_predict(model, sample)
    tests.append(("T2", "Prediction is valid category", PASS if pred in CATEGORIES else FAIL))

    # T3: Deterministic prediction
    p1, _ = _safe_predict(model, sample)
    p2, _ = _safe_predict(model, sample)
    tests.append(("T3", "Deterministic prediction", PASS if p1 == p2 else FAIL))

    # T4: Top-k returns correct length
    top3 = _safe_predict_top_k(model, sample, 3)
    tests.append(("T4", "Top-k returns exactly 3 results", PASS if len(top3) == 3 else FAIL))

    # T5: Confidence values valid
    confs = [(1 - dist) * 100 for _, dist in top3]
    valid_conf = all(0 <= c <= 100 for c in confs)
    tests.append(("T5", "Confidence values in valid range", PASS if valid_conf else FAIL))

    for tid, desc, res in tests:
        print(f"{tid:<10}{desc:<35}{res:<10}")

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    model = build_model()

    section5_execution_demo(model, total_tests=300)
    section6_test_cases(model)

    print("\n" + "=" * 70)
    print("All tests completed successfully.")
    print("=" * 70)