"""Classify new articles with either saved model.

Both classifiers return the same shape, one dict per text:
    {"label": "Sport", "confidence": 0.93}

- KNNClassifier: embeds the text with multilingual-e5-small and votes among its nearest
  training articles in the Qdrant collection built by scripts/train_knn.py.
- TransformerClassifier: the fine-tuned DistilBERT saved in models/distilbert/.
"""

from pathlib import Path

import numpy as np

from gnad.knn import QdrantKNN, vote

ROOT = Path(__file__).resolve().parents[2]
KNN_SETTINGS = ROOT / "results" / "knn.json"
QDRANT_PATH = ROOT / "data" / "qdrant"
DISTILBERT_PATH = ROOT / "models" / "distilbert"


class KNNClassifier:
    def __init__(self, embed_fn, knn: QdrantKNN, label_names: list[str], k: int, weighted: bool):
        self.embed_fn = embed_fn  # list[str] -> normalised vectors, shape (n, dim)
        self.knn = knn
        self.label_names = label_names
        self.k = k
        self.weighted = weighted

    @classmethod
    def load(cls, label_names: list[str]) -> "KNNClassifier":
        import json

        from gnad.embeddings import embed, load_model

        if not QDRANT_PATH.exists():
            raise FileNotFoundError("No Qdrant index found: run scripts/embed.py and scripts/train_knn.py first")
        best = json.loads(KNN_SETTINGS.read_text())["best_params"]
        model = load_model()
        return cls(lambda texts: embed(model, texts), QdrantKNN(path=str(QDRANT_PATH)), label_names,
                   k=best["k"], weighted=best["weighted"])

    def predict(self, texts: list[str]) -> list[dict]:
        neighbours = self.knn.neighbours(np.asarray(self.embed_fn(texts)), limit=self.k)
        out = []
        for n in neighbours:
            label, share = vote(n, self.k, self.weighted)
            out.append({"label": self.label_names[label], "confidence": round(float(share), 4)})
        return out


class TransformerClassifier:
    def __init__(self, model, tokenizer, max_length: int = 512, batch_size: int = 16):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def load(cls, path: Path = DISTILBERT_PATH) -> "TransformerClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not path.exists():
            raise FileNotFoundError(f"No fine-tuned model in {path}: run the Colab notebook and unzip its output here")
        return cls(AutoModelForSequenceClassification.from_pretrained(path), AutoTokenizer.from_pretrained(path))

    def predict(self, texts: list[str]) -> list[dict]:
        import torch

        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], truncation=True, max_length=self.max_length,
                                   padding=True, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(self.model(**batch).logits, dim=-1)
            for p in probs:
                i = int(p.argmax())
                out.append({"label": self.model.config.id2label[i], "confidence": round(float(p[i]), 4)})
        return out
