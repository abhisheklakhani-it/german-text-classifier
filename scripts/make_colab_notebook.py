"""Build notebooks/finetune_distilbert_colab.ipynb from the project's own source files.

The notebook writes data.py, metrics.py and train_transformer.py into the Colab machine
unchanged, so Colab trains with exactly the same split, settings and metrics as a local run.

    python scripts/make_colab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = "transformers==5.19.0 datasets==5.1.0 accelerate==1.15.0 scikit-learn==1.9.1"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(keepends=True)}


def write_file(path: str) -> dict:
    return code(f"%%writefile {path}\n" + (ROOT / path).read_text())


cells = [
    md("""
# Fine-tune German DistilBERT on 10kGNAD (Colab)

Part of [german-text-classifier](https://github.com/abhisheklakhani-it/german-text-classifier).

1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
2. Training takes about 15 minutes. The last cell downloads `distilbert_results.zip`.
3. Unzip it into the project folder: it contains `models/distilbert/` and `results/transformer*.json(l)`.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code(f"!pip -q install {PINS}"),
    code("!mkdir -p src/gnad scripts results models"),
    code("%%writefile src/gnad/__init__.py\n"),
    write_file("src/gnad/data.py"),
    write_file("src/gnad/metrics.py"),
    write_file("scripts/train_transformer.py"),
    md("Quick check that the whole pipeline runs (20 steps on 200 articles, about 1 minute):"),
    code("!python scripts/train_transformer.py --smoke"),
    md("The real run: 3 epochs, best epoch chosen by validation macro-F1, then one score on the test split."),
    code("!python scripts/train_transformer.py"),
    code("!cat results/transformer.json | head -20"),
    code("""
!zip -qr distilbert_results.zip models/distilbert results/transformer.json results/transformer.test_predictions.jsonl
from google.colab import files
files.download("distilbert_results.zip")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = ROOT / "notebooks" / "finetune_distilbert_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")
