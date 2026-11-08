"""Tests for data_loader.py."""

import pytest

from data_loader import load_dataset, train_test_split

# Feature value i always belongs to label "i", so pairing mistakes are easy to spot.
FEATURES = [[float(i)] for i in range(10)]
LABELS = [str(i) for i in range(10)]


def test_load_dataset_skips_header_and_empty_lines(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("f1,f2,label\n1.0,2.0,cat\n3.5,4.5,dog\n\n")

    features, labels = load_dataset(str(csv_file))

    assert features == [[1.0, 2.0], [3.5, 4.5]]
    assert labels == ["cat", "dog"]


def test_split_sizes_and_no_lost_samples():
    X_train, X_test, y_train, y_test = train_test_split(FEATURES, LABELS, test_ratio=0.3)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert sorted(y_train + y_test) == sorted(LABELS)


def test_split_keeps_features_and_labels_paired():
    X_train, X_test, y_train, y_test = train_test_split(FEATURES, LABELS)

    for point, label in zip(X_train + X_test, y_train + y_test):
        assert str(int(point[0])) == label


def test_split_same_seed_gives_same_result():
    first = train_test_split(FEATURES, LABELS, seed=1)
    second = train_test_split(FEATURES, LABELS, seed=1)
    assert first == second


def test_split_invalid_ratio_raises():
    with pytest.raises(ValueError):
        train_test_split(FEATURES, LABELS, test_ratio=1.5)