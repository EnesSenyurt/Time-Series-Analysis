"""Tests for src/experiments/stats_tests.py"""
import numpy as np
import pytest

from src.experiments.stats_tests import mcnemar_test, wilcoxon_test


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------


def test_wilcoxon_identical_scores_pvalue_one():
    result = wilcoxon_test([0.8, 0.9, 0.7], [0.8, 0.9, 0.7])
    assert result["pvalue"] == 1.0


def test_wilcoxon_returns_dict_keys():
    result = wilcoxon_test([0.6, 0.7, 0.8], [0.5, 0.6, 0.7])
    assert "statistic" in result and "pvalue" in result


def test_wilcoxon_clearly_different_scores_low_pvalue():
    # a clearly better than b → should be significant with enough samples
    a = [0.9] * 10
    b = [0.1] * 10
    result = wilcoxon_test(a, b)
    assert result["pvalue"] < 0.05


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_mcnemar_identical_predictions_pvalue_high():
    p = mcnemar_test([1, 0, 1], [1, 0, 1], [1, 0, 1])["pvalue"]
    assert p > 0.05


def test_mcnemar_returns_dict_keys():
    result = mcnemar_test([1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0])
    assert "statistic" in result and "pvalue" in result


def test_mcnemar_zero_discordant_pairs():
    # Both models agree on every sample → pvalue = 1.0
    result = mcnemar_test([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0])
    assert result["pvalue"] == 1.0


def test_mcnemar_pvalue_in_unit_interval():
    result = mcnemar_test([1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [0, 0, 1, 0, 1])
    assert 0.0 <= result["pvalue"] <= 1.0
