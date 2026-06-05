"""Deep learning models for time-series anomaly detection.

Provides:
    make_sequences       -- sliding-window segmentation
    build_lstm           -- two-layer LSTM classifier (config-driven)
    build_cnn1d          -- two-layer 1D-CNN classifier (config-driven)
    train_dl             -- seeded training with EarlyStopping + class balancing
    predict_dl           -- probability inference
    find_best_threshold  -- sweep thresholds to maximise val F1
"""
from __future__ import annotations

import os
import random

import numpy as np

# Disable oneDNN fused ops so Conv/matmul results are bit-exact across runs.
# Must be set before TensorFlow is imported anywhere in the process.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    # set_random_seed covers Python random, NumPy, and TF global + op seeds
    tf.keras.utils.set_random_seed(seed)


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window segmentation.

    Parameters
    ----------
    X       : (n_samples, n_features)
    y       : (n_samples,)
    seq_len : lookback window length

    Returns
    -------
    X3d  : (n_windows, seq_len, n_features)
    ywin : (n_windows,)  max label in each window (1 if any anomaly present)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    n = len(X)
    if n < seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    n_windows = n - seq_len + 1
    X3d = np.stack([X[i : i + seq_len] for i in range(n_windows)])
    ywin = np.array([int(y[i : i + seq_len].max()) for i in range(n_windows)], dtype=np.int32)
    return X3d, ywin


def build_lstm(cfg, n_features: int):
    """Build a two-layer LSTM classifier from config."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    seq_len = cfg.dl.sequence_length
    units = cfg.dl.lstm.units
    dropout = cfg.dl.lstm.dropout

    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(units[0], return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units[1])(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_cnn1d(cfg, n_features: int):
    """Build a two-layer 1D-CNN classifier from config."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    seq_len = cfg.dl.sequence_length
    filters = cfg.dl.cnn1d.filters
    kernel_size = cfg.dl.cnn1d.kernel_size
    dropout = cfg.dl.cnn1d.dropout

    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(filters[0], kernel_size, padding="same", activation="relu")(inp)
    x = layers.Conv1D(filters[1], kernel_size, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_dl(
    model,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    cfg,
    seed: int,
):
    """Train a Keras model with seeding, balanced class weights, and EarlyStopping.

    Parameters
    ----------
    model      : compiled Keras model
    train_data : (X_train, y_train)  X shape (n, seq_len, n_features)
    val_data   : (X_val, y_val)
    cfg        : ConfigNode
    seed       : int

    Returns
    -------
    (model, history)
    """
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping

    set_global_seed(seed)

    X_train, y_train = train_data
    X_val, y_val = val_data

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))

    es_cfg = cfg.dl.early_stopping
    callbacks = [
        EarlyStopping(
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            restore_best_weights=es_cfg.restore_best_weights,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.dl.epochs,
        batch_size=cfg.dl.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def predict_dl(model, X: np.ndarray) -> np.ndarray:
    """Return sigmoid probabilities, shape (n,)."""
    return model.predict(X, verbose=0).ravel()


def find_best_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_thresholds: int = 101,
) -> float:
    """Sweep thresholds in [0, 1] and return the one that maximises F1 on val set."""
    from sklearn.metrics import f1_score

    proba = predict_dl(model, X_val)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.0, 1.0, n_thresholds):
        f1 = f1_score(y_val, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
