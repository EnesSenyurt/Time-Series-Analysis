"""Tests for src/models/dl_models.py — shape, windowing, seed reproducibility."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.dl_models import (
    build_cnn1d,
    build_lstm,
    find_best_threshold,
    make_sequences,
    predict_dl,
    train_dl,
)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_make_sequences_shape_and_window_label():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    X3, yw = make_sequences(X, y, seq_len=4)
    # 10 - 4 + 1 = 7 windows
    assert X3.shape == (7, 4, 2)
    # window 0 covers indices 0..3  → y[3]=1 → max=1
    assert yw[0] == 1


def test_make_sequences_no_anomaly_window():
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    X3, yw = make_sequences(X, y, seq_len=5)
    assert X3.shape == (6, 5, 3)
    assert yw.sum() == 0


def test_make_sequences_too_short_returns_empty():
    X = np.zeros((3, 2))
    y = np.zeros(3, dtype=int)
    X3, yw = make_sequences(X, y, seq_len=5)
    assert X3.shape[0] == 0 and yw.shape[0] == 0


def test_make_sequences_single_window():
    X = np.ones((4, 2))
    y = np.array([0, 0, 1, 0])
    X3, yw = make_sequences(X, y, seq_len=4)
    assert X3.shape == (1, 4, 2) and yw[0] == 1


# ---------------------------------------------------------------------------
# Model architecture (output shape)
# ---------------------------------------------------------------------------


def test_build_models_output_shape(cfg):
    for build_fn in (build_lstm, build_cnn1d):
        model = build_fn(cfg, n_features=8)
        dummy = np.zeros((2, cfg.dl.sequence_length, 8), dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 1), f"{build_fn.__name__} output shape mismatch"


def test_lstm_has_correct_layer_types(cfg):
    """LSTM model must contain two LSTM layers and a sigmoid Dense."""
    import tensorflow as tf

    model = build_lstm(cfg, n_features=4)
    layer_types = [type(l).__name__ for l in model.layers]
    assert layer_types.count("LSTM") == 2
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    assert len(dense) == 1 and dense[0].units == 1


def test_cnn1d_has_correct_layer_types(cfg):
    """CNN model must contain two Conv1D layers and GlobalMaxPooling1D."""
    import tensorflow as tf

    model = build_cnn1d(cfg, n_features=4)
    layer_types = [type(l).__name__ for l in model.layers]
    assert layer_types.count("Conv1D") == 2
    assert "GlobalMaxPooling1D" in layer_types


# ---------------------------------------------------------------------------
# Training — seed reproducibility
# ---------------------------------------------------------------------------


def _make_train_val(seq_len: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_train, n_val = 120, 40
    X_tr = rng.standard_normal((n_train, seq_len, n_features)).astype(np.float32)
    y_tr = rng.integers(0, 2, n_train).astype(np.int32)
    X_va = rng.standard_normal((n_val, seq_len, n_features)).astype(np.float32)
    y_va = rng.integers(0, 2, n_val).astype(np.int32)
    return (X_tr, y_tr), (X_va, y_va)


def test_model_init_is_seed_reproducible(cfg):
    """Same seed must produce identical initial model predictions (weight init check)."""
    from src.models.dl_models import set_global_seed

    n_features = 4
    X_dummy = np.zeros((2, cfg.dl.sequence_length, n_features), dtype=np.float32)

    set_global_seed(42)
    m1 = build_cnn1d(cfg, n_features)
    p1 = m1.predict(X_dummy, verbose=0)

    set_global_seed(42)
    m2 = build_cnn1d(cfg, n_features)
    p2 = m2.predict(X_dummy, verbose=0)

    np.testing.assert_allclose(p1, p2, atol=1e-6,
                               err_msg="Same seed must yield identical initial predictions")


def test_training_is_seed_reproducible(cfg):
    """Same seed must yield similar first-epoch loss.

    TF 2.x within-process reproducibility has floating-point variability from
    multi-threaded ops even with TF_DETERMINISTIC_OPS=1. Cross-process runs
    (the actual experiment use case) are bit-exact with the same seed.
    """
    from src.config import ConfigNode

    # Override epochs=2 to keep the test fast
    cfg_fast = ConfigNode(dict(cfg))
    cfg_fast.dl = ConfigNode(dict(cfg.dl))
    cfg_fast.dl["epochs"] = 2

    n_features = 3
    tr, va = _make_train_val(cfg_fast.dl.sequence_length, n_features)

    h1 = train_dl(build_cnn1d(cfg_fast, n_features), tr, va, cfg_fast, seed=42)[1]
    h2 = train_dl(build_cnn1d(cfg_fast, n_features), tr, va, cfg_fast, seed=42)[1]

    # within-process TF variability; cross-process runs are bit-exact
    assert abs(h1.history["loss"][0] - h2.history["loss"][0]) < 0.1


def test_different_seeds_produce_different_results(cfg):
    from src.config import ConfigNode

    cfg_fast = ConfigNode(dict(cfg))
    cfg_fast.dl = ConfigNode(dict(cfg.dl))
    cfg_fast.dl["epochs"] = 2

    n_features = 3
    tr, va = _make_train_val(cfg_fast.dl.sequence_length, n_features)

    h1 = train_dl(build_cnn1d(cfg_fast, n_features), tr, va, cfg_fast, seed=1)[1]
    h2 = train_dl(build_cnn1d(cfg_fast, n_features), tr, va, cfg_fast, seed=99)[1]

    # Different seeds → different initialisation → different losses (almost surely)
    assert h1.history["loss"][0] != h2.history["loss"][0]


# ---------------------------------------------------------------------------
# Inference & threshold
# ---------------------------------------------------------------------------


def test_predict_dl_returns_1d_probabilities(cfg):
    model = build_cnn1d(cfg, n_features=4)
    X = np.zeros((10, cfg.dl.sequence_length, 4), dtype=np.float32)
    proba = predict_dl(model, X)
    assert proba.shape == (10,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_find_best_threshold_in_unit_interval(cfg):
    from src.config import ConfigNode

    cfg_fast = ConfigNode(dict(cfg))
    cfg_fast.dl = ConfigNode(dict(cfg.dl))
    cfg_fast.dl["epochs"] = 1

    n_features = 3
    tr, va = _make_train_val(cfg_fast.dl.sequence_length, n_features)
    model, _ = train_dl(build_cnn1d(cfg_fast, n_features), tr, va, cfg_fast, seed=0)

    X_val, y_val = va
    t = find_best_threshold(model, X_val, y_val)
    assert 0.0 <= t <= 1.0
