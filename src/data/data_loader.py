"""Dataset loaders for SKAB and BATADAL, faithful to the project spec.

SKAB:  only ``valve1`` and ``valve2`` folders are concatenated. Two metadata
columns are added -- ``source_group`` (folder) and ``source_file`` (folder-
qualified file name, kept unique so file-based grouping does not merge
``valve1/0.csv`` with ``valve2/0.csv``). The target is ``anomaly``.

BATADAL: only ``BATADAL_dataset04.csv`` (Training Dataset 2) is used. The label
column is ``ATT_FLAG``; the ``-999`` (concealed/unlabeled) entries are handled
according to ``unlabeled_policy`` and the final target is binarised to {0, 1}.
"""
from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT


# --------------------------------------------------------------------------- #
# SKAB
# --------------------------------------------------------------------------- #
def load_skab(cfg) -> pd.DataFrame:
    """Load and concatenate SKAB ``valve1`` + ``valve2`` CSVs."""
    skab = cfg.datasets.skab
    root = PROJECT_ROOT / skab.dir
    frames = []
    for folder in skab.use_folders:
        for csv_path in sorted((root / folder).glob("*.csv")):
            frame = pd.read_csv(csv_path, sep=skab.sep)
            frame["source_group"] = folder
            frame["source_file"] = f"{folder}/{csv_path.name}"
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"SKAB CSV bulunamadi: {root} / {skab.use_folders}")

    df = pd.concat(frames, ignore_index=True)
    df[skab.target] = df[skab.target].astype(float).round().astype(int)
    return df


def get_skab_features(df: pd.DataFrame, cfg) -> list[str]:
    """Sensor feature columns for SKAB (meta + target excluded)."""
    excluded = set(cfg.datasets.skab.drop_cols) | {cfg.datasets.skab.target}
    return [c for c in df.columns if c not in excluded]


# --------------------------------------------------------------------------- #
# BATADAL
# --------------------------------------------------------------------------- #
def apply_unlabeled_policy(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Resolve ``-999`` entries in ATT_FLAG and binarise the label to {0, 1}."""
    bat = cfg.datasets.batadal
    target = bat.target
    out = df.copy()
    if bat.unlabeled_policy == "as_normal":
        out[target] = out[target].replace(bat.unlabeled_value, 0)
    elif bat.unlabeled_policy == "drop":
        out = out[out[target] != bat.unlabeled_value].copy()
    else:
        raise ValueError(f"Bilinmeyen unlabeled_policy: {bat.unlabeled_policy}")
    out[target] = (out[target] > 0).astype(int)
    return out


def load_batadal(cfg) -> pd.DataFrame:
    """Load BATADAL Training Dataset 2 (dataset04) with cleaned columns/labels."""
    bat = cfg.datasets.batadal
    path = PROJECT_ROOT / bat.file
    df = pd.read_csv(path, sep=bat.sep, skipinitialspace=bat.skipinitialspace)
    df.columns = df.columns.str.strip()
    df[bat.time_col] = pd.to_datetime(
        df[bat.time_col].astype(str).str.strip(),
        format="%d/%m/%y %H",
        errors="coerce",
    )
    return apply_unlabeled_policy(df, cfg)


def get_batadal_features(df: pd.DataFrame, cfg) -> list[str]:
    """Sensor/system feature columns for BATADAL (time + target excluded)."""
    bat = cfg.datasets.batadal
    excluded = {bat.time_col, bat.target}
    return [c for c in df.columns if c not in excluded]
