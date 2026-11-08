"""Tests for metrics.py."""

import pytest

from metrics import accuracy


def test_accuracy_all_correct():
    assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_accuracy_half_correct():
    assert accuracy(["a", "b", "a", "b"], ["a", "a", "a", "a"]) == 0.5


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a"])


def test_accuracy_empty_list_raises():
    with pytest.raises(ValueError):
        accuracy([], [])