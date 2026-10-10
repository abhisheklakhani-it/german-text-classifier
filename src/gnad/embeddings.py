"""Turn articles into normalised sentence embeddings with multilingual-e5-small."""

import numpy as np

EMBED_MODEL = "intfloat/multilingual-e5-small"
# e5 models expect a prefix. For symmetric tasks such as classification, the model card
# says to use "query: " on every text.
PREFIX = "query: "


def load_model(device: str | None = None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.max_seq_length = 512
    return model


def embed(model, texts: list[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
    return model.encode([PREFIX + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=show_progress).astype(np.float32)
