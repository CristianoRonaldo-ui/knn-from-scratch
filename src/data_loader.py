"""Load a numeric dataset with a label column from a CSV file."""

import csv
import random


def load_dataset(file_path: str) -> tuple[list[list[float]], list[str]]:
    """Read a CSV file and return (features, labels).

    Each row must contain numeric feature columns followed by one label column.
    The first row is treated as a header and skipped.
    """
    features = []
    labels = []

    with open(file_path, newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip the header row

        for row in reader:
            if not row:
                continue  # skip empty lines
            features.append([float(value) for value in row[:-1]])
            labels.append(row[-1])

    return features, labels


def train_test_split(
    features: list[list[float]],
    labels: list[str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[list[float]], list[list[float]], list[str], list[str]]:
    """Shuffle the samples and split them into training and test sets.

    Returns (X_train, X_test, y_train, y_test), the same order as scikit-learn.
    """
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same length.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    indices = list(range(len(features)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_size = int(len(indices) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = [features[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
