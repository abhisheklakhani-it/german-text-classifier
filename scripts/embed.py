"""Embed every article once and cache the vectors in data/embeddings/ (about 10 min on an M2).

    python scripts/embed.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnad.data import load  # noqa: E402
from gnad.embeddings import EMBED_MODEL, embed, load_model  # noqa: E402

OUT = ROOT / "data" / "embeddings"


def main() -> None:
    data = load()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(device)
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name in ("train", "validation", "test"):
        split = getattr(data, name)
        start = time.perf_counter()
        vectors = embed(model, split.texts, show_progress=True)
        timings[name] = round(time.perf_counter() - start, 1)
        np.save(OUT / f"{name}.npy", vectors)
        np.save(OUT / f"{name}_labels.npy", np.array(split.labels))
        print(f"{name}: {vectors.shape} in {timings[name]} s", flush=True)
    tokens = [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in data.test.texts]
    info = {"model": EMBED_MODEL, "device": device, "dimensions": int(vectors.shape[1]), "seconds": timings,
            "test_articles_over_512_tokens": sum(n > 512 for n in tokens) / len(tokens)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "embeddings.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
