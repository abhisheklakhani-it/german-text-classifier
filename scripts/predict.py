"""Classify German news text from the command line.

    python scripts/predict.py "Der FC Bayern gewinnt das Spiel mit 3:0."
    python scripts/predict.py --model knn "Die Inflation im Euroraum steigt weiter."
    echo "Ein Artikel pro Zeile" | python scripts/predict.py -

Prints one JSON line per text: {"text": ..., "label": ..., "confidence": ...}
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnad.predict import KNNClassifier, TransformerClassifier  # noqa: E402

LABEL_NAMES = json.loads((ROOT / "results" / "eda.json").read_text())["label_names"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="+", help='article text(s), or "-" to read one article per line from stdin')
    parser.add_argument("--model", choices=["distilbert", "knn"], default="distilbert")
    args = parser.parse_args()
    texts = [line.strip() for line in sys.stdin if line.strip()] if args.texts == ["-"] else args.texts

    try:
        classifier = TransformerClassifier.load() if args.model == "distilbert" else KNNClassifier.load(LABEL_NAMES)
    except FileNotFoundError as error:
        sys.exit(f"error: {error}")
    for text, result in zip(texts, classifier.predict(texts)):
        print(json.dumps({"text": text[:80], **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
