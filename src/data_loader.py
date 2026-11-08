"""Load a numeric dataset with a label column from a CSV file."""

import csv


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