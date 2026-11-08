"""Command-line interface for training and evaluating the k-NN classifier."""

import argparse

from data_loader import load_dataset, train_test_split
from knn import KNNClassifier
from metrics import accuracy


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line options."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a k-NN classifier written from scratch."
    )
    parser.add_argument("--data", default="data/iris.csv", help="Path to the CSV dataset.")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors that vote.")
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Fraction of samples used for testing."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        features, labels = load_dataset(args.data)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_ratio=args.test_ratio, seed=args.seed
        )

        model = KNNClassifier(k=args.k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy(y_test, predictions)
    except FileNotFoundError:
        print(f"Error: file not found: {args.data}")
        return
    except ValueError as error:
        print(f"Error: {error}")
        return

    correct = sum(1 for true, pred in zip(y_test, predictions) if true == pred)

    print(f"Dataset: {args.data} ({len(features)} samples)")
    print(f"Train/test split: {len(X_train)} / {len(X_test)} (seed={args.seed})")
    print(f"k = {args.k}")
    print(f"Accuracy: {score:.2%} ({correct}/{len(y_test)} correct)")


if __name__ == "__main__":
    main()