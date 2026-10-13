"""Load 10kGNAD and split it into train / validation / test.

The test split is the official one (1,028 articles) and is only used for the final
comparison. Model choices (k for kNN, best epoch) use a stratified 10% validation split
carved out of the official training data.
"""

from dataclasses import dataclass

from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET = "community-datasets/gnad10"
SEED = 42


@dataclass
class Split:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class GNAD:
    train: Split
    validation: Split
    test: Split
    label_names: list[str]


def load(validation_size: float = 0.1) -> GNAD:
    ds = load_dataset(DATASET)
    names = ds["train"].features["label"].names
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        ds["train"]["text"], ds["train"]["label"], test_size=validation_size, stratify=ds["train"]["label"], random_state=SEED
    )
    return GNAD(
        train=Split(list(train_texts), list(train_labels)),
        validation=Split(list(val_texts), list(val_labels)),
        test=Split(list(ds["test"]["text"]), list(ds["test"]["label"])),
        label_names=names,
    )
