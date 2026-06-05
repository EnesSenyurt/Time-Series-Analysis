"""Tests for src/experiments/scenarios.py"""
import numpy as np
import pytest

from src.experiments.scenarios import apply_scenario, make_unseen_report


# ---------------------------------------------------------------------------
# apply_scenario
# ---------------------------------------------------------------------------


def test_original_returns_copy(cfg):
    X = np.ones((5, 3))
    rng = np.random.default_rng(0)
    Xo = apply_scenario("original", X, cfg, rng)
    assert np.allclose(Xo, X)
    assert Xo is not X  # must be a copy


def test_unseen_returns_copy(cfg):
    X = np.ones((5, 3))
    rng = np.random.default_rng(0)
    Xu = apply_scenario("unseen", X, cfg, rng)
    assert np.allclose(Xu, X)
    assert Xu is not X


def test_noise_changes_data_but_not_shape(cfg):
    X = np.ones((5, 3))
    rng = np.random.default_rng(0)
    Xn = apply_scenario("noise", X, cfg, rng)
    assert Xn.shape == X.shape
    assert not np.allclose(Xn, X)


def test_noise_with_explicit_train_std(cfg):
    X = np.zeros((10, 2))
    rng = np.random.default_rng(1)
    train_std = np.array([1.0, 2.0])
    Xn = apply_scenario("noise", X, cfg, rng, train_std=train_std)
    assert Xn.shape == X.shape
    assert not np.allclose(Xn, X)


def test_noise_is_seed_reproducible(cfg):
    X = np.ones((5, 3))
    Xn1 = apply_scenario("noise", X, cfg, np.random.default_rng(42))
    Xn2 = apply_scenario("noise", X, cfg, np.random.default_rng(42))
    assert np.allclose(Xn1, Xn2)


def test_unknown_scenario_raises(cfg):
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario("invalid", np.ones((3, 2)), cfg, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# make_unseen_report
# ---------------------------------------------------------------------------


class _FakeAutomaton:
    """Minimal automaton stub for testing make_unseen_report."""
    def __init__(self, vocab):
        self.vocab = set(vocab)


def test_unseen_report_all_seen():
    auto = _FakeAutomaton(["aab", "abc", "bcc"])
    report = make_unseen_report(auto, ["aab", "abc", "bcc", "aab"])
    assert report["n_unseen"] == 0
    assert report["unseen_rate"] == 0.0


def test_unseen_report_all_unseen():
    auto = _FakeAutomaton(["aab"])
    report = make_unseen_report(auto, ["xyz", "zzz"])
    assert report["n_unseen"] == 2
    assert report["unseen_rate"] == 1.0


def test_unseen_report_partial():
    auto = _FakeAutomaton(["aab", "abc"])
    report = make_unseen_report(auto, ["aab", "xyz", "abc", "qqq"])
    assert report["n_unseen"] == 2
    assert abs(report["unseen_rate"] - 0.5) < 1e-9


def test_unseen_report_empty_patterns():
    auto = _FakeAutomaton(["aab"])
    report = make_unseen_report(auto, [])
    assert report["unseen_rate"] == 0.0
    assert report["n_total"] == 0
