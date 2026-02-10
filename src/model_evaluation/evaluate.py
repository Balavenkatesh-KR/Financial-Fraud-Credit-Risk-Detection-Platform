from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    roc_auc: float
    pr_auc: float
    ks_stat: float
    best_threshold: float
    confusion: list[list[int]]
    expected_cost: float


def optimize_threshold(y_true, proba, fn_cost: float = 10.0, fp_cost: float = 1.0) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thr, min_cost = 0.5, float("inf")
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < min_cost:
            best_thr, min_cost = float(thr), float(cost)
    return best_thr, min_cost


def evaluate_binary_classifier(y_true, proba, threshold: float | None = None) -> EvaluationResult:
    roc_auc = float(roc_auc_score(y_true, proba))
    pr_auc = float(average_precision_score(y_true, proba))
    ks = float(ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic)

    if threshold is None:
        threshold, expected_cost = optimize_threshold(y_true, proba)
    else:
        expected_cost = float("nan")

    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred).tolist()

    return EvaluationResult(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        ks_stat=ks,
        best_threshold=threshold,
        confusion=cm,
        expected_cost=expected_cost,
    )


def precision_recall_points(y_true, proba):
    precision, recall, threshold = precision_recall_curve(y_true, proba)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": threshold.tolist()}
