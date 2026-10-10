import pytest

from gnad.metrics import evaluate

NAMES = ["Sport", "Kultur", "Web"]


def test_perfect_predictions():
    result = evaluate([0, 1, 2], [0, 1, 2], NAMES)
    assert result["accuracy"] == 1 and result["macro_f1"] == 1


def test_macro_f1_weights_small_classes_equally():
    # 8 Sport articles all right, the single Kultur article wrong: accuracy is high, macro-F1 is not.
    y_true = [0] * 8 + [1, 2]
    y_pred = [0] * 8 + [0, 2]
    result = evaluate(y_true, y_pred, NAMES)
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["per_class"]["Kultur"]["f1"] == 0
    assert result["macro_f1"] < 0.7


def test_confusion_matrix_rows_are_true_labels():
    matrix = evaluate([0, 1, 1], [0, 0, 1], NAMES)["confusion_matrix"]
    assert matrix[1] == [1, 1, 0]  # one Kultur article predicted as Sport, one correct
