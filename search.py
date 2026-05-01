import numpy as np
import heapq
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────────────────────────
# Distance utilities
# ─────────────────────────────────────────────────────────────────
class DistanceMetrics:
    """Static collection of distance functions."""

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance in [0, 2]. Works on pre-normalised or raw vectors."""
        return 1.0 - float(np.dot(a, b)) / (
            np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        )

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """L2 distance — useful as a secondary tie-breaking signal."""
        return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────
# Internal A* node
# ─────────────────────────────────────────────────────────────────
class _AStarNode:
    """Priority-queue node for the two-level A* search."""

    __slots__ = ("label", "vector", "g", "h", "f", "node_type")

    def __init__(
        self,
        label: str,
        vector,
        g: float,
        h: float,
        node_type: str,
    ):
        self.label     = label
        self.vector    = vector
        self.g         = g
        self.h         = h
        self.f         = g + h
        self.node_type = node_type   # "class" | "centroid"

    def __lt__(self, other: "_AStarNode") -> bool:
        return self.f < other.f


# ─────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────
class SketchIndex:
    """
    Builds and stores the search structures (prototypes + centroids)
    used for fast nearest-class lookup.
    """

    def __init__(self, centroids_per_class: int = 40, random_state: int = 42):
        if centroids_per_class <= 0:
            raise ValueError("centroids_per_class must be positive")
        self.centroids_per_class = centroids_per_class
        self.random_state        = random_state
        self.prototypes: dict    = {}
        self.centroids: dict     = {}

    def fit(self, dataset: dict) -> "SketchIndex":
        """
        Compute prototypes and KMeans centroids from *dataset*.

        Parameters
        ----------
        dataset : dict[str, np.ndarray]  – label → (N, D) float array
        """
        self.prototypes = self._build_prototypes(dataset)
        self.centroids  = self._build_centroids(dataset)
        return self

    # ── Private builders ───────────────────────────────────────────
    @staticmethod
    def _build_prototypes(dataset: dict) -> dict:
        return {label: np.mean(samples, axis=0) for label, samples in dataset.items()}

    def _build_centroids(self, dataset: dict) -> dict:
        """KMeans centroids per class."""
        centroids = {}
        for label, samples in dataset.items():
            k  = min(self.centroids_per_class, len(samples))
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(samples)
            centroids[label] = km.cluster_centers_
        return centroids


# ─────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────
class SketchClassifier:
    """
    Two-level A* classifier + ensemble top-K ranker.

    Parameters
    ----------
    index : SketchIndex  – fitted search index (prototypes + centroids)
    """

    def __init__(self, index: SketchIndex):
        self.index = index

    # ── Public API ─────────────────────────────────────────────────
    def predict(self, input_vector: np.ndarray) -> tuple:
        """
        Return (label, distance) for the single best match.
        Uses a two-level A* over the centroid graph.
        """
        return self._a_star(input_vector)

    def predict_top_k(self, input_vector: np.ndarray, k: int = 3) -> list:
        """
        Return list of (label, distance) for the top-*k* matches.
        Uses the Borda-count ensemble for higher accuracy.
        """
        return self._ensemble_top_k(input_vector, k)

    # ── A* ─────────────────────────────────────────────────────────
    def _a_star(self, input_vector: np.ndarray) -> tuple:
        """
        Two-level A* over (class → centroid).
        Level-1 heuristic = min centroid distance (admissible).
        First centroid node popped is globally optimal.
        """
        iv        = self._normalise(input_vector)
        centroids = self.index.centroids
        open_list = []

        for label, cents in centroids.items():
            h = float(min(DistanceMetrics.cosine(iv, c) for c in cents))
            heapq.heappush(open_list, _AStarNode(label, None, 0.0, h, "class"))

        while open_list:
            cur = heapq.heappop(open_list)
            if cur.node_type == "centroid":
                return cur.label, cur.g
            for cent in centroids[cur.label]:
                g = float(DistanceMetrics.cosine(iv, cent))
                heapq.heappush(open_list, _AStarNode(cur.label, cent, g, 0.0, "centroid"))

        return None, float("inf")

    # ── Ensemble top-K ─────────────────────────────────────────────
    def _ensemble_top_k(self, input_vector: np.ndarray, k: int) -> list:
        """
        Combines three signals per class and rank-aggregates (Borda count):

        1. Centroid score  – best (min) cosine distance to any centroid
        2. Prototype score – cosine distance to the class mean vector
        3. Top-5 mean      – average distance to the 5 nearest centroids

        The class with the lowest total Borda rank is the best match.
        Returns [(label, centroid_score), ...] for the top-k labels.
        """
        iv         = self._normalise(input_vector)
        centroids  = self.index.centroids
        prototypes = self.index.prototypes
        labels     = list(centroids.keys())

        centroid_scores:  dict = {}
        prototype_scores: dict = {}
        top5_mean_scores: dict = {}

        for label in labels:
            cents = centroids[label]
            dists = sorted(DistanceMetrics.cosine(iv, c) for c in cents)

            centroid_scores[label]  = dists[0]
            prototype_scores[label] = DistanceMetrics.cosine(iv, prototypes[label])
            top5_mean_scores[label] = float(np.mean(dists[:5]))

        def ranks(score_dict: dict) -> dict:
            sorted_labels = sorted(score_dict, key=score_dict.get)
            return {lbl: i for i, lbl in enumerate(sorted_labels)}

        r1 = ranks(centroid_scores)
        r2 = ranks(prototype_scores)
        r3 = ranks(top5_mean_scores)

        borda         = {lbl: r1[lbl] + r2[lbl] + r3[lbl] for lbl in labels}
        sorted_labels = sorted(borda, key=borda.get)
        return [(lbl, centroid_scores[lbl]) for lbl in sorted_labels[:k]]

    # ── Utility ────────────────────────────────────────────────────
    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)


# ─────────────────────────────────────────────────────────────────
# Module-level convenience wrappers (keep old call-sites working)
# ─────────────────────────────────────────────────────────────────
def create_class_prototypes(dataset: dict) -> dict:
    return SketchIndex._build_prototypes(dataset)


def create_centroids(dataset: dict, centroids_per_class: int = 40) -> dict:
    idx = SketchIndex(centroids_per_class=centroids_per_class)
    return idx._build_centroids(dataset)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return DistanceMetrics.cosine(a, b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return DistanceMetrics.euclidean(a, b)


def a_star_classification(input_vector, prototypes, centroids_dict):
    index = SketchIndex.__new__(SketchIndex)
    index.prototypes = prototypes
    index.centroids  = centroids_dict
    clf = SketchClassifier(index)
    return clf.predict(input_vector)


def a_star_top_k(input_vector, prototypes, centroids_dict, k=3):
    index = SketchIndex.__new__(SketchIndex)
    index.prototypes = prototypes
    index.centroids  = centroids_dict
    clf = SketchClassifier(index)
    return clf.predict_top_k(input_vector, k)