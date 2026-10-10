"""Embeddings + Qdrant nearest-neighbour classifier: choose k on validation, score once on test.

    python scripts/embed.py       # once, caches the vectors
    python scripts/train_knn.py

Also fits a logistic regression on the same embeddings ("linear probe") as a reference:
it shows how much a trained classifier head adds over a pure vector-database lookup.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnad.data import load  # noqa: E402
from gnad.knn import QdrantKNN, vote  # noqa: E402
from gnad.metrics import evaluate  # noqa: E402

EMB = ROOT / "data" / "embeddings"
K_GRID = [1, 3, 5, 10, 20, 30, 50]


def save_predictions(path: Path, labels, pred, confidence) -> None:
    with open(path, "w") as f:
        for i, (gold, p, c) in enumerate(zip(labels, pred, confidence)):
            f.write(json.dumps({"index": i, "label": int(gold), "pred": int(p), "confidence": round(float(c), 4)}) + "\n")


def main() -> None:
    names = load().label_names
    X = {s: np.load(EMB / f"{s}.npy") for s in ("train", "validation", "test")}
    y = {s: np.load(EMB / f"{s}_labels.npy").tolist() for s in ("train", "validation", "test")}

    knn = QdrantKNN(path=str(ROOT / "data" / "qdrant"))
    start = time.perf_counter()
    knn.index(X["train"], y["train"])
    index_seconds = time.perf_counter() - start

    val_neighbours = knn.neighbours(X["validation"], limit=max(K_GRID))
    search = []
    for k, weighted in itertools.product(K_GRID, (False, True)):
        pred = [vote(n, k, weighted)[0] for n in val_neighbours]
        scores = evaluate(y["validation"], pred, names)
        search.append({"k": k, "weighted": weighted, "val_accuracy": round(scores["accuracy"], 4), "val_macro_f1": round(scores["macro_f1"], 4)})
    best = max(search, key=lambda r: r["val_macro_f1"])
    print("best on validation:", best)

    start = time.perf_counter()
    test_neighbours = knn.neighbours(X["test"], limit=best["k"])
    query_seconds = time.perf_counter() - start
    votes = [vote(n, best["k"], best["weighted"]) for n in test_neighbours]
    pred = [v[0] for v in votes]
    result = evaluate(y["test"], pred, names)
    result.update(best_params=best, grid_search=search, index_seconds=round(index_seconds, 1),
                  ms_per_query=round(1000 * query_seconds / len(pred), 2), indexed_articles=len(y["train"]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "knn.json").write_text(json.dumps(result, indent=2) + "\n")
    save_predictions(ROOT / "results" / "knn.test_predictions.jsonl", y["test"], pred, [v[1] for v in votes])
    print(f"Qdrant kNN test: accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}")

    # Reference: a trained linear head on the same embeddings. C chosen on validation.
    probe_search = []
    for C in (0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=C, max_iter=3000).fit(X["train"], y["train"])
        scores = evaluate(y["validation"], clf.predict(X["validation"]).tolist(), names)
        probe_search.append({"C": C, "val_macro_f1": round(scores["macro_f1"], 4)})
    best_c = max(probe_search, key=lambda r: r["val_macro_f1"])["C"]
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(X["train"], y["train"])
    proba = clf.predict_proba(X["test"])
    probe = evaluate(y["test"], proba.argmax(axis=1).tolist(), names)
    probe.update(best_params={"C": best_c}, grid_search=probe_search)
    (ROOT / "results" / "linear_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    save_predictions(ROOT / "results" / "linear_probe.test_predictions.jsonl", y["test"], proba.argmax(axis=1), proba.max(axis=1))
    print(f"Linear probe test: accuracy {probe['accuracy']:.4f}  macro-F1 {probe['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
