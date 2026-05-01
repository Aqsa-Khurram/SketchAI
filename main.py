import numpy as np
import random
import tkinter as tk

from load_data import CATEGORIES
from model import SketchModel
from gui import SketchApp, AIDrawApp, ModeSelectApp, TimerModeSelectApp


# ─────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────
class ModelEvaluator:
    """Runs a random sample evaluation and prints accuracy + confusion."""

    def __init__(self, model: SketchModel):
        self.model = model

    def run(self, total_tests: int = 500) -> float:
        correct   = 0
        labels    = list(self.model.test_data.keys())
        confusion = {}

        for _ in range(total_tests):
            true_label = random.choice(labels)
            sample     = random.choice(self.model.test_data[true_label])

            predicted_label, _ = self.model.predict(sample)

            if predicted_label == true_label:
                correct += 1
            else:
                key = (true_label, predicted_label)
                confusion[key] = confusion.get(key, 0) + 1

        accuracy = (correct / total_tests) * 100
        print(f"\nTotal Tests : {total_tests}")
        print(f"Correct     : {correct}")
        print(f"Accuracy    : {accuracy:.2f}%")

        top_confusions = sorted(confusion.items(), key=lambda x: -x[1])[:10]
        print("\nTop confusions (true → predicted : count):")
        for (true, pred), count in top_confusions:
            print(f"  {true:12s} → {pred:12s} : {count}")

        return accuracy


# ─────────────────────────────────────────────────────────────────
# Application controller
# ─────────────────────────────────────────────────────────────────
class QuickDrawApp:
    """
    Top-level application controller.

    Owns the Tk root window and orchestrates screen transitions:
    mode selection → SketchApp (human draws) or AIDrawApp (AI draws).
    """

    def __init__(self, model: SketchModel):
        self.model = model
        self.root  = tk.Tk()

    # ── Public entry point ─────────────────────────────────────────
    def run(self) -> None:
        self._launch_mode_select()
        self.root.mainloop()

    # ── Screen launchers ───────────────────────────────────────────
    def _clear_root(self) -> None:
        for w in self.root.winfo_children():
            w.destroy()

    def _launch_mode_select(self) -> None:
        self._clear_root()
        ModeSelectApp(
            self.root,
            on_human_draws=self._launch_human_draws,
            on_ai_draws=self._launch_ai_draws,
        )

    def _launch_human_draws(self) -> None:
        self._clear_root()
        SketchApp(
            self.root,
            predict_fn=self._gui_predict,
            on_switch_mode=self._launch_mode_select,
        )

    def _launch_ai_draws(self) -> None:
        self._clear_root()
        TimerModeSelectApp(
            self.root,
            on_timed   = lambda: self._start_ai_draws(timed=True),
            on_relaxed = lambda: self._start_ai_draws(timed=False),
            on_back    = self._launch_mode_select,
        )

    def _start_ai_draws(self, timed: bool) -> None:
        self._clear_root()
        AIDrawApp(
            self.root,
            categories       = CATEGORIES,
            pixel_prototypes = self.model.pixel_prototypes,
            on_switch_mode   = self._launch_mode_select,
            timed_mode       = timed,
        )

    # ── Prediction callback used by SketchApp ──────────────────────
    def _gui_predict(self, vector: np.ndarray) -> str:
        top3  = self.model.predict_top_k(vector, k=3)
        lines = []
        for rank, (label, dist) in enumerate(top3, 1):
            confidence = max(0, (1 - dist) * 100)
            lines.append(f"#{rank} {label}  ({confidence:.0f}%)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    print("Training model...")
    model = SketchModel(
        n_components       = 150,
        samples_per_class  = 2000,
        centroids_per_class = 35,
    ).train()
    print("Model ready!\n")

    mode = "gui"   # "eval" or "gui"

    if mode == "eval":
        ModelEvaluator(model).run(total_tests=500)
    elif mode == "gui":
        QuickDrawApp(model).run()