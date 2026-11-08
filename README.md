# k-NN From Scratch

A k-Nearest Neighbors classifier implemented from scratch in pure Python, trained and evaluated on the classic Iris dataset. No NumPy, no scikit-learn: every step of the algorithm (distance, neighbor search, voting, train/test split, accuracy) is written by hand and covered by tests.

## Features

- `KNNClassifier` with a scikit-learn style API (`fit`, `predict`)
- Euclidean distance written by hand
- Majority vote with deterministic tie-breaking (the nearest neighbor wins a tie)
- Reproducible train/test split with a random seed
- Accuracy metric
- Command-line interface to try different values of `k`
- 20 pytest tests, including edge cases (ties, invalid `k`, predicting before fitting)
- Core code uses only the Python standard library

## Project Structure

```
knn-from-scratch/
├── data/
│   └── iris.csv            # 150 flowers, 4 features, 3 species
├── src/
│   ├── distance.py         # euclidean_distance
│   ├── data_loader.py      # load_dataset, train_test_split
│   ├── knn.py              # KNNClassifier (fit, predict_one, predict)
│   ├── metrics.py          # accuracy
│   └── cli.py              # command-line entry point
├── tests/
│   ├── test_distance.py
│   ├── test_data_loader.py
│   ├── test_knn.py
│   └── test_metrics.py
├── pytest.ini
└── requirements.txt
```

## How to Run

Requires Python 3.10+.

```bash
git clone https://github.com/CristianoRonaldo-ui/knn-from-scratch.git
cd knn-from-scratch
python3 src/cli.py
```

Options:

```bash
python3 src/cli.py --k 5              # number of neighbors that vote
python3 src/cli.py --test-ratio 0.3   # fraction of samples used for testing
python3 src/cli.py --seed 7           # different random split
python3 src/cli.py --help
```

Run commands from the project root, because the default dataset path (`data/iris.csv`) is relative to it.

## Example Output

```
$ python3 src/cli.py
Dataset: data/iris.csv (150 samples)
Train/test split: 120 / 30 (seed=42)
k = 3
Accuracy: 93.33% (28/30 correct)
```

Results on the same split (seed = 42) for different values of `k`:

| k | Accuracy |
|---|----------|
| 1 | 90.00% (27/30) |
| 3 | 93.33% (28/30) |
| 5 | 96.67% (29/30) |
| 7 | 96.67% (29/30) |

## How It Works

k-NN is a **lazy learner**: `fit` only stores the training data, and all the work happens at prediction time. To predict the label of a new point:

1. **Measure** the Euclidean distance from the new point to every training point.
2. **Sort** the distances and keep the `k` closest neighbors.
3. **Vote**: the most common label among those neighbors is the prediction. If there is a tie, the label of the nearest neighbor wins, because `Counter` keeps labels in the order they were added (closest first).

Before training, the dataset is shuffled and split into training and test sets. Shuffling is essential because `iris.csv` is sorted by species; without it, the test set would contain only one species. Indices are shuffled (not the lists themselves) so each feature row always stays paired with its label.

## Complexity Analysis

`n` = training samples, `d` = features per sample, `m` = test samples.

| Function | Time | Notes |
|----------|------|-------|
| `euclidean_distance` | O(d) | One pass over the features |
| `fit` | O(1) | Only stores references to the data |
| `predict_one` | O(n·d + n log n) | n distances, then sort them |
| `predict` | O(m · (n·d + n log n)) | `predict_one` for each test point |
| `train_test_split` | O(n) | Shuffle and slice indices |
| `accuracy` | O(m) | One pass over the predictions |

Memory: the model keeps the whole training set, O(n·d). This is why k-NN is fast to train but slow to predict on large datasets.

## Testing

```bash
pip install -r requirements.txt
pytest -v
```

All 20 tests pass. Tests use small hand-made datasets where the correct answer can be checked by hand, plus `tmp_path` for file-reading tests.

## What I Learned

- **k-NN does almost nothing in `fit`.** Writing it myself made the difference between a lazy learner (k-NN) and an eager learner (like Naive Bayes in my [sms-spam-classifier](https://github.com/CristianoRonaldo-ui/sms-spam-classifier)) concrete.
- **Why shuffling matters.** Looking at the raw CSV first showed it is sorted by species, so splitting without shuffling would have produced a useless test set.
- **Shuffle indices, not parallel lists.** Shuffling features and labels separately would silently attach the wrong label to each flower. A test now checks that every feature stays paired with its label.
- **What `random_state` really does.** A fixed seed gives the same shuffle every time, so a change in accuracy comes from a code change, not from luck.
- **Tie-breaking is a design decision.** A test documents that the nearest neighbor wins when the vote is tied.
- **The value of `k` changes the result.** On this split, `k = 1` scored 90% and `k = 5` scored 96.67%, which is a first taste of hyperparameter tuning.

## Future Improvements

- Stratified train/test split so each species keeps the same proportion in both sets
- Feature scaling (min-max or z-score) so no single feature dominates the distance
- Find the k nearest neighbors with `heapq.nsmallest` in O(n log k) instead of a full sort
- Cross-validation to choose the best `k` instead of using a single split
- Compare results with scikit-learn's `KNeighborsClassifier`