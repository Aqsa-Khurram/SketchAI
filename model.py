import numpy as np
from sklearn.decomposition import PCA
from load_data import load_data
from search import create_class_prototypes, create_centroids

class SketchModel:
    def __init__(self):
        self.pca = None
        self.prototypes = None
        self.centroids = None

    def train(self):
        dataset = load_data(samples_per_class=2000)

        train_data = {}
        for label, samples in dataset.items():
            split = int(0.75 * len(samples))
            train_data[label] = samples[:split]

        # Train PCA
        all_train = np.vstack(list(train_data.values()))
        self.pca = PCA(n_components=100)
        self.pca.fit(all_train)

        # Transform data
        for label in train_data:
            transformed = self.pca.transform(train_data[label])
            norm = np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8
            train_data[label] = transformed / norm

        # Create structures
        self.prototypes = create_class_prototypes(train_data)
        self.centroids = create_centroids(train_data, centroids_per_class=25)

    def predict(self, x, a_star_classification):
        x = self.pca.transform([x])[0]
        x = x / (np.linalg.norm(x) + 1e-8)

        label, _ = a_star_classification(x, self.prototypes, self.centroids)
        return label