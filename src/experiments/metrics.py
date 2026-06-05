"""Classification metrics and aggregation utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict:
    """Return accuracy, precision, recall and F1 (binary, zero_division=0)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def aggregate(metric_dicts: list[dict]) -> dict:
    """Compute per-metric (mean, std) over a list of metric dicts.

    Returns {metric_name: (mean, std)}.
    """
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {
        k: (
            float(np.mean([d[k] for d in metric_dicts])),
            float(np.std([d[k] for d in metric_dicts])),
        )
        for k in keys
    }
