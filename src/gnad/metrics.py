"""Accuracy, macro-F1, per-class scores and confusion matrix, shared by both models."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def evaluate(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict:
    labels = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
