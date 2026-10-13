"""Both classifiers, built from tiny local parts: no downloads, no saved models needed."""

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

from gnad.knn import QdrantKNN
from gnad.predict import KNNClassifier, TransformerClassifier

NAMES = ["Sport", "Wirtschaft"]


def fake_embed(texts):
    # "Tor" points one way, everything else the other way.
    return np.array([[1.0, 0.0] if "Tor" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


def test_knn_classifier_labels_by_nearest_articles():
    knn = QdrantKNN()
    knn.index(fake_embed(["Tor!", "Tor in der 90.", "Aktien fallen", "Zinsen steigen"]), [0, 0, 1, 1])
    clf = KNNClassifier(fake_embed, knn, NAMES, k=2, weighted=False)
    assert clf.predict(["Ein spätes Tor", "Die Börse"]) == [
        {"label": "Sport", "confidence": 1.0},
        {"label": "Wirtschaft", "confidence": 1.0},
    ]


def tiny_transformer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "tor": 4, "aktie": 5}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]")
    torch.manual_seed(0)
    config = DistilBertConfig(vocab_size=len(vocab), dim=16, hidden_dim=32, n_layers=1, n_heads=2,
                              num_labels=2, id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)})
    return DistilBertForSequenceClassification(config), tokenizer


def test_transformer_classifier_returns_label_names_and_probabilities():
    model, tokenizer = tiny_transformer()
    clf = TransformerClassifier(model, tokenizer, batch_size=2)
    out = clf.predict(["tor tor", "aktie", "tor aktie unbekannt"])  # 3 texts, batches of 2
    assert len(out) == 3
    for row in out:
        assert row["label"] in NAMES
        assert 0.5 <= row["confidence"] <= 1.0  # the winner of 2 classes has at least half the probability
