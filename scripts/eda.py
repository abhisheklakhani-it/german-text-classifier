"""Class balance and text lengths for 10kGNAD -> results/eda.json and docs/eda.png."""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnad.data import load  # noqa: E402
from gnad.plotting import BLUE, INK, INK_2, MUTED, style_axis, SURFACE  # noqa: E402


def main() -> None:
    data = load()
    names = data.label_names
    stats = {"label_names": names, "splits": {}}
    for split_name in ("train", "validation", "test"):
        split = getattr(data, split_name)
        words = [len(t.split()) for t in split.texts]
        counts = Counter(split.labels)
        stats["splits"][split_name] = {
            "articles": len(split),
            "per_class": {names[i]: counts[i] for i in range(len(names))},
            "words": {"median": statistics.median(words), "mean": round(statistics.mean(words), 1),
                      "p95": sorted(words)[int(0.95 * len(words))], "max": max(words), "min": min(words)},
        }
    # Rough share of articles longer than 512 subword tokens (about 1.4 tokens per German word).
    train_words = [len(t.split()) for t in data.train.texts]
    stats["share_over_365_words"] = round(sum(w > 365 for w in train_words) / len(train_words), 3)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eda.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    train = stats["splits"]["train"]["per_class"]
    order = sorted(train, key=train.get)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    a.barh(order, [train[n] for n in order], color=BLUE, height=0.6, zorder=2)
    for y, n in enumerate(order):
        a.text(train[n] + 15, y, str(train[n]), va="center", color=INK_2, fontsize=9)
    a.set_title("Articles per topic (training split)", loc="left", color=INK, fontsize=11)
    style_axis(a, grid="x")
    b.hist(train_words, bins=60, color=BLUE, zorder=2)
    median = stats["splits"]["train"]["words"]["median"]
    b.axvline(median, color=INK_2, linewidth=1, linestyle="--")
    b.text(median + 15, b.get_ylim()[1] * 0.9, f"median {median:.0f} words", color=INK_2, fontsize=9)
    b.set_title("Article length in words (training split)", loc="left", color=INK, fontsize=11)
    b.set_xlim(0, 1500)
    style_axis(b, grid="y")
    b.tick_params(axis="x", colors=MUTED)
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "eda.png", dpi=160, facecolor=SURFACE)
    print(json.dumps({k: v for k, v in stats["splits"].items()}, indent=1, ensure_ascii=False))
    print("share over ~512 tokens:", stats["share_over_365_words"])


if __name__ == "__main__":
    main()
