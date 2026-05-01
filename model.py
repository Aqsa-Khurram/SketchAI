import numpy as np
from sklearn.decomposition import PCA

from load_data import DataLoader, CATEGORIES
from search import SketchIndex, SketchClassifier
from preprocess import ImagePreprocessor


class SketchModel:
    """
    End-to-end sketch classifier.

    Responsibilities
    ----------------
    - Load data via DataLoader
    - Optionally augment hard classes
    - Fit PCA for dimensionality reduction
    - Build SketchIndex (prototypes + centroids)
    - Expose predict / predict_top_k for inference

    Parameters
    ----------
    n_components         : PCA components (default 150)
    samples_per_class    : training samples per category (default 2000)
    centroids_per_class  : KMeans centroids per category (default 35)
    data_path            : directory that holds the .npy files
    random_state         : reproducibility seed
    """

    # Hand-picked representative sample indices per category.
    # Any category not listed falls back to the first 50 samples.
    DEFAULT_PICKED_SAMPLES: dict = {
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

    # Classes that are visually similar and benefit from augmentation.
    HARD_CLASSES: set = {"sun", "circle", "flower", "star", "clock", "moon"}

    def __init__(
        self,
        n_components: int = 150,
        samples_per_class: int = 2000,
        centroids_per_class: int = 35,
        data_path: str = "data",
        random_state: int = 42,
    ):
        self.n_components        = n_components
        self.samples_per_class   = samples_per_class
        self.centroids_per_class = centroids_per_class
        self.data_path           = data_path
        self.random_state        = random_state

        self._pca:           PCA | None            = None
        self._index:         SketchIndex | None     = None
        self._classifier:    SketchClassifier | None = None
        self._preprocessor:  ImagePreprocessor      = ImagePreprocessor()
        self._pixel_prototypes: dict                = {}
        self._test_data:     dict                   = {}

    # ── Public interface ───────────────────────────────────────────
    def train(self) -> "SketchModel":
        """Fit PCA and build the search index. Returns self for chaining."""
        loader = DataLoader(self.data_path)

        # Raw pixel prototypes for the AI-Draw display
        self._pixel_prototypes = loader.load_raw_pixels(self.DEFAULT_PICKED_SAMPLES)

        dataset = loader.load(self.samples_per_class)
        if not dataset:
            raise RuntimeError("Dataset is empty! Check your data path and .npy files.")

        train_data, self._test_data = self._split(dataset, ratio=0.75)

        # PCA
        self._pca = self._fit_pca(train_data)
        print(
            f"PCA explained variance: "
            f"{self._pca.explained_variance_ratio_.sum() * 100:.1f}%"
        )

        train_data = self._transform(train_data, self._pca)
        train_data = self._augment_hard_classes(train_data)
        self._test_data = self._transform(self._test_data, self._pca)

        # Build search index
        self._index = SketchIndex(
            centroids_per_class=self.centroids_per_class,
            random_state=self.random_state,
        ).fit(train_data)

        self._classifier = SketchClassifier(self._index)
        return self

    def predict(self, raw_vector: np.ndarray) -> tuple:
        """Return (label, distance) for the best matching class."""
        self._check_trained()
        x = self._embed(raw_vector)
        return self._classifier.predict(x)

    def predict_top_k(self, raw_vector: np.ndarray, k: int = 3) -> list:
        """Return [(label, distance), ...] for the top-k classes."""
        self._check_trained()
        x = self._embed(raw_vector)
        return self._classifier.predict_top_k(x, k)

    def predict_image(self, image: np.ndarray) -> tuple:
        """Preprocess a raw canvas image then predict."""
        vector = self._preprocessor.process(image)
        return self.predict(vector)

    def predict_image_top_k(self, image: np.ndarray, k: int = 3) -> list:
        """Preprocess a raw canvas image then return top-k predictions."""
        vector = self._preprocessor.process(image)
        return self.predict_top_k(vector, k)

    @property
    def pca(self) -> PCA:
        self._check_trained()
        return self._pca

    @property
    def prototypes(self) -> dict:
        self._check_trained()
        return self._index.prototypes

    @property
    def centroids(self) -> dict:
        self._check_trained()
        return self._index.centroids

    @property
    def test_data(self) -> dict:
        return self._test_data

    @property
    def pixel_prototypes(self) -> dict:
        return self._pixel_prototypes

    # ── Private helpers ────────────────────────────────────────────
    def _embed(self, raw_vector: np.ndarray) -> np.ndarray:
        """Apply PCA + unit-normalisation to a single raw vector."""
        x = self._pca.transform([raw_vector])[0]
        return x / (np.linalg.norm(x) + 1e-8)

    def _check_trained(self) -> None:
        if self._pca is None or self._index is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    @staticmethod
    def _split(dataset: dict, ratio: float) -> tuple:
        train, test = {}, {}
        for label, samples in dataset.items():
            split = int(ratio * len(samples))
            train[label] = samples[:split]
            test[label]  = samples[split:]
        return train, test

    def _fit_pca(self, train_data: dict) -> PCA:
        all_samples = np.vstack(list(train_data.values()))
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca.fit(all_samples)
        return pca

    @staticmethod
    def _transform(data: dict, pca: PCA) -> dict:
        transformed = {}
        for label, samples in data.items():
            x    = pca.transform(samples)
            norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            transformed[label] = x / norm
        return transformed

    def _augment_hard_classes(self, train_data: dict) -> dict:
        for label in self.HARD_CLASSES:
            if label in train_data:
                train_data[label] = self._augment(train_data[label])
                print(f"Augmented {label}: {len(train_data[label])} samples")
        return train_data

    def _augment(self, samples: np.ndarray, n_augmented: int = 300) -> np.ndarray:
        """Add small Gaussian noise to existing samples and re-normalise."""
        rng       = np.random.default_rng(self.random_state)
        noise     = rng.normal(0, 0.015, size=(n_augmented, samples.shape[1]))
        idx       = rng.integers(0, len(samples), size=n_augmented)
        augmented = samples[idx] + noise
        norms     = np.linalg.norm(augmented, axis=1, keepdims=True) + 1e-8
        augmented = augmented / norms
        return np.vstack([samples, augmented])