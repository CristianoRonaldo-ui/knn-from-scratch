"""Evaluation metrics for classification."""


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Return the fraction of predictions that match the true labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Cannot compute accuracy of an empty list.")

    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)
