"""Statistical significance tests for paired model comparisons.

wilcoxon_test  -- paired signed-rank test on F1 scores across seeds/folds
mcnemar_test   -- paired per-sample correct/incorrect comparison
"""
from __future__ import annotations

import numpy as np


def wilcoxon_test(scores_a: list | np.ndarray, scores_b: list | np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired performance scores (e.g. F1 per fold).

    Returns {"statistic": float, "pvalue": float}.
    When all differences are zero, returns pvalue=1.0 (no significant difference).
    """
    from scipy.stats import wilcoxon

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b

    if np.all(diff == 0):
        return {"statistic": 0.0, "pvalue": 1.0}

    stat, pvalue = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def mcnemar_test(
    y_true: list | np.ndarray,
    pred_a: list | np.ndarray,
    pred_b: list | np.ndarray,
) -> dict:
    """McNemar test on per-sample correctness of two classifiers.

    Builds the 2×2 contingency table from discordant pairs (a-correct/b-wrong and
    a-wrong/b-correct) and applies the exact binomial test.

    Returns {"statistic": float, "pvalue": float}.
    When both models agree on every sample, returns pvalue=1.0.
    """
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))

    if n10 + n01 == 0:
        return {"statistic": 0.0, "pvalue": 1.0}

    table = np.array([[n11, n10], [n01, n00]])
    result = sm_mcnemar(table, exact=True)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
