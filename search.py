import numpy as np
import heapq
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────────────────────────
# Distance functions
# ─────────────────────────────────────────────────────────────────
def cosine_distance(a, b):
    """Cosine distance in [0,2]. Works on pre-normalized or raw vectors."""
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def euclidean_distance(a, b):
    """L2 distance — useful as a second signal for tie-breaking."""
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────
# Node for A* heap
# ─────────────────────────────────────────────────────────────────
class Node:
    def __init__(self, label, vector, g, h, node_type):
        self.label     = label
        self.vector    = vector
        self.g         = g
        self.h         = h
        self.f         = g + h
        self.node_type = node_type  # "class" | "centroid"

    def __lt__(self, other):
        return self.f < other.f


# ─────────────────────────────────────────────────────────────────
# Build structures
# ─────────────────────────────────────────────────────────────────
def create_class_prototypes(dataset):
    """Mean vector per class — used as a fast prototype signal."""
    return {label: np.mean(samples, axis=0) for label, samples in dataset.items()}


def create_centroids(dataset, centroids_per_class=40, random_state=42):
    """
    KMeans centroids per class.
    40 centroids balances coverage vs speed for 30 classes × 1500 samples.
    n_init=10 gives more stable clusters than the default 3.
    """
    if centroids_per_class <= 0:
        raise ValueError("centroids_per_class must be positive")

    centroids = {}
    for label, samples in dataset.items():
        k = min(centroids_per_class, len(samples))
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(samples)
        centroids[label] = km.cluster_centers_
    return centroids


# ─────────────────────────────────────────────────────────────────
# A* Classification
# ─────────────────────────────────────────────────────────────────
def a_star_classification(input_vector, prototypes, centroids_dict):
    """
    Two-level A* over (class → centroid) graph.
    Level-1 heuristic = min centroid distance (admissible).
    First centroid node popped is globally optimal.
    """
    iv = input_vector / (np.linalg.norm(input_vector) + 1e-8)
    open_list = []

    for label, cents in centroids_dict.items():
        h = float(min(cosine_distance(iv, c) for c in cents))
        heapq.heappush(open_list, Node(label, None, 0.0, h, "class"))

    while open_list:
        cur = heapq.heappop(open_list)

        if cur.node_type == "centroid":
            return cur.label, cur.g

        for cent in centroids_dict[cur.label]:
            g = float(cosine_distance(iv, cent))
            heapq.heappush(open_list, Node(cur.label, cent, g, 0.0, "centroid"))

    return None, float("inf")


# ─────────────────────────────────────────────────────────────────
# Ensemble top-K  ← KEY accuracy improvement
# ─────────────────────────────────────────────────────────────────
def ensemble_top_k(input_vector, prototypes, centroids_dict, k=3):
    """
    Combines THREE signals for each class, then rank-aggregates:

      1. Centroid score  – best (min) cosine distance to any centroid
      2. Prototype score – cosine distance to the class mean vector
      3. Top-5 mean      – average distance to the 5 nearest centroids
                          (rewards classes with *dense* nearby clusters,
                           penalises lucky single-centroid matches)

    Each signal is ranked 1…N independently.  Final rank = sum of three
    ranks (Borda count).  The class with the lowest total rank wins.

    Why this works:
    - Centroid score alone can be fooled by a single outlier centroid
      that happens to be close (e.g. an unusual sun drawing matching a
      circle centroid).
    - Prototype score adds a global "is this the right neighbourhood?"
      check.
    - Top-5 mean punishes classes that only have one close centroid and
      rewards classes with a whole cluster nearby — much more reliable
      for visually rich categories like flower, cat, guitar.
    """
    iv = input_vector / (np.linalg.norm(input_vector) + 1e-8)
    labels = list(centroids_dict.keys())

    centroid_scores  = {}
    prototype_scores = {}
    top5_mean_scores = {}

    for label in labels:
        cents = centroids_dict[label]
        dists = sorted(cosine_distance(iv, c) for c in cents)

        centroid_scores[label]  = dists[0]
        prototype_scores[label] = cosine_distance(iv, prototypes[label])
        top5_mean_scores[label] = float(np.mean(dists[:5]))

    def ranks(score_dict):
        sorted_labels = sorted(score_dict, key=score_dict.get)
        return {lbl: i for i, lbl in enumerate(sorted_labels)}

    r1 = ranks(centroid_scores)
    r2 = ranks(prototype_scores)
    r3 = ranks(top5_mean_scores)

    borda = {lbl: r1[lbl] + r2[lbl] + r3[lbl] for lbl in labels}
    sorted_labels = sorted(borda, key=borda.get)

    # Return (label, cosine_distance) pairs for top-k
    return [(lbl, centroid_scores[lbl]) for lbl in sorted_labels[:k]]


# ─────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────
def a_star_top_k(input_vector, prototypes, centroids_dict, k=3):
    """Public wrapper used by main.py predict functions."""
    return ensemble_top_k(input_vector, prototypes, centroids_dict, k=k)