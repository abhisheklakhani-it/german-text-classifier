"""Compare every trained model on the test split -> results/comparison.json, docs/*.png.

    python scripts/compare.py

Uses whichever result files exist (knn, linear_probe, transformer). Also writes
results/error_analysis.jsonl: the 10 test articles the best model got wrong with the
highest confidence, with what the other models predicted for them.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnad.analysis import agreement, load_predictions, most_confident_errors, top_confusions  # noqa: E402
from gnad.data import load  # noqa: E402
from gnad.plotting import BLUE, INK, INK_2, MUTED, ORANGE, SURFACE, style_axis  # noqa: E402

MODELS = {  # result file tag -> display name
    "knn": "e5 + Qdrant kNN",
    "linear_probe": "e5 + logistic regression",
    "transformer": "Fine-tuned DistilBERT",
}
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def main() -> None:
    runs = {tag: json.loads((RESULTS / f"{tag}.json").read_text()) for tag in MODELS if (RESULTS / f"{tag}.json").exists()}
    preds = {tag: load_predictions(RESULTS / f"{tag}.test_predictions.jsonl") for tag in runs}
    data = load()
    names = data.label_names

    table = {tag: {"model": MODELS[tag], "accuracy": round(r["accuracy"], 4), "macro_f1": round(r["macro_f1"], 4),
                   "top_confusions": top_confusions(r["confusion_matrix"], names, 3)} for tag, r in runs.items()}
    best = max(runs, key=lambda t: runs[t]["macro_f1"])
    baseline = "knn"
    summary = {"test_articles": len(data.test), "models": table, "best": best}
    if best != baseline:
        summary["agreement_" + f"{baseline}_vs_{best}"] = agreement(preds[baseline], preds[best])
    (RESULTS / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    others = [t for t in runs if t != best]
    with open(RESULTS / "error_analysis.jsonl", "w") as f:
        for p in most_confident_errors(preds[best], 10):
            row = {"index": p["index"], "true": names[p["label"]], "predicted": names[p["pred"]],
                   "confidence": p["confidence"],
                   **{f"{t}_predicted": names[preds[t][p["index"]]["pred"]] for t in others},
                   "text": data.test.texts[p["index"]][:600]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    plot_confusion(runs, names)
    plot_per_class(runs, names)
    for tag, row in table.items():
        print(f"{row['model']:<26} accuracy {row['accuracy']:.4f}  macro-F1 {row['macro_f1']:.4f}")
    print("best:", MODELS[best], "| error analysis -> results/error_analysis.jsonl")


def plot_confusion(runs: dict, names: list[str]) -> None:
    shown = [t for t in ("knn", "transformer") if t in runs]
    fig, axes = plt.subplots(1, len(shown), figsize=(6.2 * len(shown), 5.6), facecolor=SURFACE, squeeze=False)
    for ax, tag in zip(axes[0], shown):
        m = np.array(runs[tag]["confusion_matrix"])
        share = m / m.sum(axis=1, keepdims=True)  # each row sums to 1: the recall view
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for t in range(len(names)):
            for p in range(len(names)):
                if m[t, p]:
                    ax.text(p, t, m[t, p], ha="center", va="center", fontsize=8,
                            color="white" if share[t, p] > 0.6 else INK)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(names)), names, fontsize=8, color=INK_2)
        ax.set_xlabel("predicted topic", color=MUTED, fontsize=9)
        ax.set_ylabel("true topic", color=MUTED, fontsize=9)
        ax.set_title(f"{MODELS[tag]} (test, n={m.sum()})", loc="left", color=INK, fontsize=11)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrices.png", dpi=160, facecolor=SURFACE)


def plot_per_class(runs: dict, names: list[str]) -> None:
    shown = [t for t in ("knn", "transformer") if t in runs]
    colors = {"knn": BLUE, "transformer": ORANGE}
    order = sorted(names, key=lambda n: runs[shown[0]]["per_class"][n]["f1"])
    y = np.arange(len(order))
    height = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    for i, tag in enumerate(shown):
        f1 = [runs[tag]["per_class"][n]["f1"] for n in order]
        offset = (i - (len(shown) - 1) / 2) * height
        ax.barh(y + offset, f1, height=height * 0.9, color=colors[tag], label=MODELS[tag], zorder=2)
    ax.set_yticks(y, order, fontsize=9, color=INK_2)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("F1 on the test split", color=MUTED, fontsize=9)
    ax.set_title("Per-topic F1", loc="left", color=INK, fontsize=11)
    style_axis(ax, grid="x")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(DOCS / "per_class_f1.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
