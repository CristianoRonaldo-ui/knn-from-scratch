"""A k-Nearest Neighbors classifier written from scratch."""

from collections import Counter
from distance import euclidean_distance


class KNNClassifier:
    """Classify a point by majority vote among its k closest training points."""

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X_train: list[list[float]], y_train: list[str]) -> None:
        """Store the training data. k-NN does no other work at training time."""
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")
        if len(X_train) < self.k:
            raise ValueError("Need at least k training samples.")
        self.X_train = X_train
        self.y_train = y_train

    def predict_one(self, point: list[float]) -> str:
        if not self.X_train:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        distances = []
        for train_point, label in zip(self.X_train, self.y_train):
            distance = euclidean_distance(point, train_point)
            distances.append((distance, label))

        distances.sort(key=lambda pair: pair[0])
        nearest_labels = [label for _, label in distances[: self.k]]

        votes = Counter(nearest_labels)
        return votes.most_common(1)[0][0]

    def predict(self, points: list[list[float]]) -> list[str]:
        """Predict labels for a list of points."""
        return [self.predict_one(point) for point in points]
