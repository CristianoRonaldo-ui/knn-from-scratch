"""A k-Nearest Neighbors classifier written from scratch."""

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