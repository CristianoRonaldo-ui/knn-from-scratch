"""Tests for knn.py."""

import pytest

from knn import KNNClassifier

# Two well-separated groups: "red" near (0, 0) and "blue" near (10, 10).
X_TRAIN = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]
Y_TRAIN = ["red", "red", "red", "blue", "blue", "blue"]


def test_predict_two_clear_clusters():
    model = KNNClassifier(k=3)
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.predict([[0.5, 0.5], [10.5, 10.5]]) == ["red", "blue"]


def test_k_equals_one_uses_single_nearest_neighbor():
    model = KNNClassifier(k=1)
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.predict_one([9, 9]) == "blue"


def test_majority_vote_beats_single_nearest_neighbor():
    model = KNNClassifier(k=3)
    model.fit([[0], [2], [2.5]], ["a", "b", "b"])
    # The nearest point is "a", but 2 of the 3 neighbors are "b".
    assert model.predict_one([0.1]) == "b"


def test_tie_is_broken_by_nearest_neighbor():
    model = KNNClassifier(k=3)
    model.fit([[1], [2], [3]], ["a", "b", "c"])
    # Each label gets one vote, so the closest neighbor ("a") wins.
    assert model.predict_one([0]) == "a"


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)


def test_predict_before_fit_raises():
    model = KNNClassifier(k=3)
    with pytest.raises(ValueError):
        model.predict_one([1, 2])


def test_fit_with_fewer_samples_than_k_raises():
    model = KNNClassifier(k=5)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [1, 1]], ["a", "b"])