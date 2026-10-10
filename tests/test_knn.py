import numpy as np
import pytest

from gnad.knn import QdrantKNN, vote


def test_majority_vote():
    assert vote([(1, 0.9), (2, 0.8), (2, 0.7)], k=3, weighted=False) == (2, pytest.approx(2 / 3))


def test_only_first_k_neighbours_vote():
    assert vote([(1, 0.9), (2, 0.8), (2, 0.7)], k=1, weighted=False)[0] == 1


def test_weighted_vote_prefers_closer_neighbours():
    neighbours = [(1, 0.95), (2, 0.40), (2, 0.40)]
    assert vote(neighbours, k=3, weighted=False)[0] == 2
    assert vote(neighbours, k=3, weighted=True)[0] == 1


def test_tie_goes_to_lower_label_id():
    assert vote([(3, 0.5), (1, 0.5)], k=2, weighted=False)[0] == 1


def unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_qdrant_index_and_search_in_memory():
    train = np.stack([unit([1, 0, 0]), unit([0.9, 0.1, 0]), unit([0, 1, 0]), unit([0, 0.9, 0.1])])
    knn = QdrantKNN()  # in-memory Qdrant, no files, no server
    knn.index(train, [0, 0, 1, 1])
    neighbours = knn.neighbours(np.stack([unit([1, 0.05, 0]), unit([0.05, 1, 0])]), limit=2)
    assert [label for label, _ in neighbours[0]] == [0, 0]
    assert [label for label, _ in neighbours[1]] == [1, 1]
    assert neighbours[0][0][1] > neighbours[0][1][1]  # sorted by similarity


def test_reindexing_replaces_the_collection():
    knn = QdrantKNN()
    knn.index(np.stack([unit([1, 0])]), [0])
    knn.index(np.stack([unit([0, 1])]), [5])
    assert knn.neighbours(np.stack([unit([1, 0])]), limit=5)[0][0][0] == 5
