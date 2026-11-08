"""Distance functions used by the k-NN classifier."""

import math


def euclidean_distance(point_a: list[float], point_b: list[float]) -> float:
    """Return the straight-line distance between two points of the same dimension."""
    if len(point_a) != len(point_b):
        raise ValueError("Points must have the same number of features.")

    total = 0.0
    for a, b in zip(point_a, point_b):
        total += (a - b) ** 2
    return math.sqrt(total)