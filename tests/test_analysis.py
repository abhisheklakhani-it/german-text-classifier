import pytest

from gnad.analysis import agreement, most_confident_errors, top_confusions


def p(index, label, pred, confidence=0.5):
    return {"index": index, "label": label, "pred": pred, "confidence": confidence}


def test_most_confident_errors_skips_correct_and_sorts_by_confidence():
    preds = [p(0, 1, 1, 0.99), p(1, 1, 2, 0.6), p(2, 0, 2, 0.9), p(3, 2, 0, 0.7)]
    assert [e["index"] for e in most_confident_errors(preds, 2)] == [2, 3]


def test_top_confusions_ignores_the_diagonal():
    matrix = [[9, 3, 0],
              [1, 8, 0],
              [0, 5, 7]]
    assert top_confusions(matrix, ["A", "B", "C"], 2) == [("C", "B", 5), ("A", "B", 3)]


def test_agreement_counts_every_combination():
    a = [p(0, 1, 1), p(1, 1, 1), p(2, 1, 0), p(3, 1, 0)]
    b = [p(0, 1, 1), p(1, 1, 0), p(2, 1, 1), p(3, 1, 0)]
    assert agreement(a, b) == {"both_right": 1, "only_first_right": 1, "only_second_right": 1, "both_wrong": 1}


def test_agreement_refuses_misaligned_files():
    with pytest.raises(AssertionError):
        agreement([p(0, 1, 1)], [p(1, 1, 1)])
