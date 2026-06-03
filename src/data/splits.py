"""Leakage-safe data splitting strategies.

SKAB: file-based cross-validation (``StratifiedGroupKFold`` with a ``GroupKFold``
fallback) so that records from the same ``source_file`` never appear in both the
train and the test fold.

BATADAL: a strictly time-ordered 60/20/20 train/val/test split -- no random
row-level shuffling, preserving the temporal dependency of the series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# Folds are deterministic across DL seeds (seed variation lives in model init).
_FOLD_RANDOM_STATE = 42


def skab_folds(df: pd.DataFrame, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) folds grouped by ``source_file``."""
    cv = cfg.datasets.skab.cv
    target = cfg.datasets.skab.target
    groups = df[cv.group_col].to_numpy()
    y = df[target].to_numpy()

    n_groups = len(np.unique(groups))
    n_splits = min(cv.n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("SKAB CV icin en az 2 farkli source_file gerekir")

    if cv.strategy == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=_FOLD_RANDOM_STATE
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)

    return list(splitter.split(df, y, groups))


def batadal_split(
    df: pd.DataFrame, cfg
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered 60/20/20 train/val/test split (positional, no shuffle)."""
    split = cfg.datasets.batadal.split
    time_col = cfg.datasets.batadal.time_col

    ordered = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_train = int(n * split.train)
    n_val = int(n * split.val)

    train = ordered.iloc[:n_train].reset_index(drop=True)
    val = ordered.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = ordered.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test
