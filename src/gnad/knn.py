"""k-nearest-neighbour topic classifier backed by Qdrant.

The training articles' embeddings are stored in a Qdrant collection with their topic as
payload. A new article is classified by a vote among its k most similar training
articles. Qdrant runs in local mode (inside this Python process, no server or Docker).
"""

from collections import Counter

import numpy as np
from qdrant_client import QdrantClient, models

COLLECTION = "gnad_train"


class QdrantKNN:
    def __init__(self, path: str | None = None, collection: str = COLLECTION):
        self.client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        self.collection = collection

    def index(self, vectors: np.ndarray, labels: list[int], batch_size: int = 512) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE)
        )
        for start in range(0, len(vectors), batch_size):
            batch = range(start, min(start + batch_size, len(vectors)))
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"label": int(labels[i])}) for i in batch
            ])

    def neighbours(self, vectors: np.ndarray, limit: int, batch_size: int = 64) -> list[list[tuple[int, float]]]:
        """For each query vector, the (label, cosine similarity) of its `limit` nearest training articles."""
        out = []
        for start in range(0, len(vectors), batch_size):
            requests = [models.QueryRequest(query=v.tolist(), limit=limit, with_payload=True) for v in vectors[start:start + batch_size]]
            for response in self.client.query_batch_points(self.collection, requests=requests):
                out.append([(p.payload["label"], p.score) for p in response.points])
        return out


def vote(neighbours: list[tuple[int, float]], k: int, weighted: bool) -> tuple[int, float]:
    """Majority (or similarity-weighted) vote among the first k neighbours -> (label, vote share)."""
    scores: Counter = Counter()
    for label, similarity in neighbours[:k]:
        scores[label] += similarity if weighted else 1.0
    label, best = max(scores.items(), key=lambda item: (item[1], -item[0]))
    return label, best / sum(scores.values())
