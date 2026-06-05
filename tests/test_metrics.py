"""Tests for src/experiments/metrics.py"""
import numpy as np
import pytest

from src.experiments.metrics import aggregate, classification_metrics


def test_classification_metrics_known():
    m = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0])
    assert m["accuracy"] == 0.75
    assert m["recall"] == 0.5


def test_classification_metrics_perfect():
    m = classification_metrics([1, 0, 1], [1, 0, 1])
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_classification_metrics_all_wrong():
    m = classification_metrics([1, 1, 0], [0, 0, 1])
    assert m["accuracy"] == 0.0


def test_classification_metrics_zero_division_safe():
    # all predicted negative → precision undefined → returns 0.0
    m = classification_metrics([1, 1, 0], [0, 0, 0])
    assert m["precision"] == 0.0


def test_aggregate_mean_std():
    agg = aggregate([{"f1": 0.8}, {"f1": 1.0}])
    assert abs(agg["f1"][0] - 0.9) < 1e-9


def test_aggregate_single_entry():
    agg = aggregate([{"accuracy": 0.75, "f1": 0.6}])
    assert agg["accuracy"] == (0.75, 0.0)
    assert agg["f1"] == (0.6, 0.0)


def test_aggregate_empty_returns_empty():
    assert aggregate([]) == {}


def test_aggregate_multiple_metrics():
    dicts = [{"accuracy": 0.8, "f1": 0.7}, {"accuracy": 0.9, "f1": 0.9}]
    agg = aggregate(dicts)
    assert abs(agg["accuracy"][0] - 0.85) < 1e-9
    assert abs(agg["f1"][0] - 0.8) < 1e-9
