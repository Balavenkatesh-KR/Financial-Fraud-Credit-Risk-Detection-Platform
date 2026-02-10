import numpy as np

from src.model_evaluation.evaluate import evaluate_binary_classifier


def test_evaluation_outputs_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])
    result = evaluate_binary_classifier(y_true, proba)
    assert result.roc_auc > 0.8
    assert 0.05 <= result.best_threshold <= 0.95
