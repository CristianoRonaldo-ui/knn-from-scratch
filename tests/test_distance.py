"""Tests for distance.py."""

import pytest

from distance import euclidean_distance


def test_distance_3_4_5_triangle():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0


def test_distance_same_point_is_zero():
    assert euclidean_distance([5.1, 3.5, 1.4, 0.2], [5.1, 3.5, 1.4, 0.2]) == 0.0


def test_distance_four_dimensions():
    assert euclidean_distance([0, 0, 0, 0], [1, 1, 1, 1]) == 2.0


def test_mismatched_dimensions_raises():
    with pytest.raises(ValueError):
        euclidean_distance([1, 2], [1, 2, 3]) 