# SketchAI 🎨

> A Quick Draw–inspired sketch recognition and AI drawing system built with classical machine learning — no deep learning frameworks required.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-10.0+-brightgreen?style=flat)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-blue?style=flat)

---

## What is SketchAI?

SketchAI is an interactive sketch recognition game with two play modes:

- **Human Draws** — You draw on a canvas and the AI guesses your sketch in real time, showing the top 3 predictions with confidence scores.
- **AI Draws** — The AI progressively reveals a hand-picked prototype sketch stroke by stroke while you try to guess the category. A timed mode adds a 20-second countdown for extra challenge.

The system classifies sketches across **30 categories** using a fully custom AI pipeline: PCA dimensionality reduction → K-Means centroid indexing → two-level A\* search → Borda-count ensemble ranking.

---

## Demo

```
Training model...
Loading circle... circle shape: (2000, 784)
Loading cat...    cat shape:    (2000, 784)
...
PCA explained variance: 84.7%
Augmented sun: 2300 samples
Model ready!
```

Once running, the GUI launches and you can start drawing immediately.

---

## Supported Categories

```
circle    square    triangle   tree      house
star      cloud     moon       sun       book
beach     clock     fish       airplane  leaf
calendar  pencil    guitar     cup       flower
cat       car       umbrella   bicycle   ladder
key       dog       lightning  chair     hat
```

---

## Project Structure

```
SketchAI/
│
├── main.py              # Entry point — QuickDrawApp controller & ModelEvaluator
├── model.py             # SketchModel — Facade over the full AI pipeline
├── search.py            # SketchIndex, SketchClassifier, DistanceMetrics, A* search
├── preprocess.py        # ImagePreprocessor — canvas → 28×28 unit-norm vector
├── load_data.py         # DataLoader — loads & normalises QuickDraw .npy files
├── gui.py               # Tkinter GUI — SketchApp, AIDrawApp, ModeSelectApp
├── browse__dataset.py   # Utility to browse raw dataset samples
├── requirements.txt     # Python dependencies
│
└── data/                # Dataset directory (not included — see Setup below)
    ├── circle.npy
    ├── cat.npy
    └── ...              # One .npy file per category
```

---

## How It Works

### AI Pipeline

```
Raw canvas drawing (RGB)
        │
        ▼
ImagePreprocessor
  • Grayscale + invert (white strokes on black background)
  • Tight bounding-box crop
  • Square padding with margin (1/6 of side)
  • Lanczos resize to 28×28
  • Unit-norm flatten → 784-d vector
        │
        ▼
PCA (150 components)
  • Trained on 75% of dataset (all 30 classes)
  • Reduces 784-d → 150-d (~85% explained variance)
  • Re-normalise to unit hypersphere
        │
        ▼
Two-Level A* Search
  • Level 1: class nodes, heuristic = min centroid distance (admissible)
  • Level 2: centroid nodes, cost = exact cosine distance
  • Terminates at first centroid pop → globally optimal
        │
        ▼
Borda-Count Ensemble (top-k)
  • Signal 1: best centroid cosine distance
  • Signal 2: class prototype (mean vector) distance
  • Signal 3: top-5 centroid mean distance
  • Rank-aggregate → final top-3 predictions
        │
        ▼
Prediction: [("cat", 82%), ("dog", 61%), ("horse", 44%)]
```

### Training Summary

| Step | Detail |
|------|--------|
| Dataset | Google Quick, Draw! — 2,000 samples per class |
| Train/Test split | 75% / 25% |
| PCA components | 150 |
| Centroids per class | 35 (K-Means, n_init=10) |
| Hard-class augmentation | +300 noise samples for: sun, circle, flower, star, clock, moon |
| Evaluation accuracy | ~72–78% (30-class problem) |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aqsa-Khurram/SketchAI.git
cd SketchAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Tkinter is bundled with standard Python installations. If missing on Linux: `sudo apt install python3-tk`

### 3. Download the dataset

SketchAI uses the [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) in `.npy` format (bitmap, 28×28 grayscale, flattened to 784 values).

Download the `.npy` files for all 30 categories listed above from:
**https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap**

Place all downloaded files inside a `data/` folder in the project root:

```
SketchAI/
└── data/
    ├── circle.npy
    ├── cat.npy
    ├── dog.npy
    └── ...
```

### 4. Run the application

```bash
python main.py
```

---

## Configuration

All key hyperparameters are set in `main.py` and can be tuned:

```python
model = SketchModel(
    n_components        = 150,   # PCA dimensions (higher = more detail, slower)
    samples_per_class   = 2000,  # Training samples per category
    centroids_per_class = 35,    # K-Means centroids per class (higher = more accurate, slower)
)
```

To switch between GUI mode and evaluation mode, change the `mode` variable in `main.py`:

```python
mode = "gui"   # Launch the interactive GUI
mode = "eval"  # Run accuracy evaluation on 500 test samples
```

---

## Evaluation

Running in `eval` mode prints a full accuracy report and top confusion pairs:

```
Total Tests : 500
Correct     : 371
Accuracy    : 74.20%

Top confusions (true → predicted : count):
  moon         → circle       : 12
  sun          → circle       : 10
  flower       → sun          : 8
  clock        → circle       : 7
  ...
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              PRESENTATION LAYER  (gui.py)            │
│   ModeSelectApp  SketchApp  AIDrawApp  TimerSelect   │
└──────────────────────┬──────────────────────────────┘
                       │ callbacks
┌──────────────────────▼──────────────────────────────┐
│         APPLICATION LAYER  (main.py + model.py)      │
│           QuickDrawApp        SketchModel            │
└────────────┬──────────────────────┬─────────────────┘
             │                      │
┌────────────▼──────────┐  ┌────────▼──────────────┐
│  AI / SEARCH LAYER    │  │  DATA LAYER            │
│  search.py            │  │  load_data.py          │
│  SketchIndex          │  │  preprocess.py         │
│  SketchClassifier     │  └────────────────────────┘
│  DistanceMetrics      │
└───────────────────────┘
```

---

## Requirements

```
numpy>=1.24
scikit-learn>=1.3
Pillow>=10.0
```

Tkinter is part of the Python standard library (no pip install needed).

---

## Known Limitations

- Visually similar categories (circle, sun, clock, moon) are frequently confused due to the flat pixel representation losing structural detail.
- No stroke-order information is used — drawings are treated as static bitmaps.
- Model training takes 30–60 seconds on first launch (no caching yet).
- Fixed 28×28 resolution loses fine-grained detail (e.g. guitar strings, clock hands).

---

## Future Work

- Replace PCA with a lightweight CNN encoder for higher accuracy
- Serialize the trained model to disk for instant startup on relaunch
- Add stroke-sequence modelling (RNN / Transformer) for richer features
- HNSW approximate nearest-neighbour index for faster large-scale search
- Expand to all 345 Quick, Draw! categories

---

## License

This project was developed as part of AI2002: Artificial Intelligence coursework.

---

*Built with Python, NumPy, scikit-learn, Pillow, and Tkinter.*
