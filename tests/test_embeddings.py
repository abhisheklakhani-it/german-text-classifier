import numpy as np

from gnad.embeddings import PREFIX, embed


class FakeModel:
    def __init__(self):
        self.seen = None

    def encode(self, texts, **kwargs):
        self.seen = (texts, kwargs)
        return np.ones((len(texts), 4), dtype=np.float64)


def test_embed_adds_e5_prefix_and_normalises():
    model = FakeModel()
    vectors = embed(model, ["Ein Artikel", "Noch einer"])
    texts, kwargs = model.seen
    assert texts == [PREFIX + "Ein Artikel", PREFIX + "Noch einer"]
    assert kwargs["normalize_embeddings"] is True
    assert vectors.dtype == np.float32 and vectors.shape == (2, 4)
