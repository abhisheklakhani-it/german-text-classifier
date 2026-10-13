"""Helpers for comparing models on the test split: prediction files, error picking, agreement."""

import json
from collections import Counter
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def most_confident_errors(predictions: list[dict], n: int) -> list[dict]:
    """The n wrong predictions the model was most sure about: the most telling mistakes."""
    wrong = [p for p in predictions if p["pred"] != p["label"]]
    return sorted(wrong, key=lambda p: (-p["confidence"], p["index"]))[:n]


def top_confusions(matrix: list[list[int]], label_names: list[str], n: int) -> list[tuple[str, str, int]]:
    """The n most frequent (true topic, predicted topic, count) mistakes, read off a confusion matrix."""
    pairs = [(label_names[t], label_names[p], count) for t, row in enumerate(matrix)
             for p, count in enumerate(row) if t != p and count]
    return sorted(pairs, key=lambda x: -x[2])[:n]


def agreement(a: list[dict], b: list[dict]) -> dict:
    """How two models' right/wrong answers overlap on the same test articles."""
    assert [p["index"] for p in a] == [p["index"] for p in b], "prediction files are not aligned"
    counts = Counter((x["pred"] == x["label"], y["pred"] == y["label"]) for x, y in zip(a, b))
    return {"both_right": counts[True, True], "only_first_right": counts[True, False],
            "only_second_right": counts[False, True], "both_wrong": counts[False, False]}
