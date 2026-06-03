from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATADAL_DIR = PROJECT_ROOT / "BATADAL"
SKAB_DIR = PROJECT_ROOT / "SKAB"


def load_batadal_file(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load a single BATADAL CSV file."""
    csv_path = Path(path)
    kwargs = {}
    if parse_dates:
        kwargs["parse_dates"] = ["DATETIME"]

    return pd.read_csv(csv_path, **kwargs)


def load_batadal_all(root: str | Path = BATADAL_DIR, parse_dates: bool = True) -> pd.DataFrame:
    """Load all BATADAL CSV files and keep their original file names."""
    root_path = Path(root)
    frames = []

    for csv_path in sorted(root_path.glob("*.csv")):
        frame = load_batadal_file(csv_path, parse_dates=parse_dates)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No BATADAL CSV files found in {root_path}")

    return pd.concat(frames, ignore_index=True)


def iter_skab_files(
    root: str | Path = SKAB_DIR,
    include_anomaly_free: bool = True,
) -> Iterable[Path]:
    """Yield SKAB CSV files from the local SKAB folder."""
    root_path = Path(root)
    scenario_dirs = ["valve1", "valve2", "other"]

    if include_anomaly_free:
        anomaly_free = root_path / "anomaly-free" / "anomaly-free.csv"
        if anomaly_free.exists():
            yield anomaly_free

    for scenario in scenario_dirs:
        yield from sorted((root_path / scenario).glob("*.csv"))


def load_skab_file(
    path: str | Path,
    parse_dates: bool = True,
    add_missing_labels: bool = True,
) -> pd.DataFrame:
    """Load a single SKAB CSV file."""
    csv_path = Path(path)
    kwargs = {"sep": ";"}
    if parse_dates:
        kwargs["parse_dates"] = ["datetime"]

    frame = pd.read_csv(csv_path, **kwargs)

    if add_missing_labels:
        if "anomaly" not in frame.columns:
            frame["anomaly"] = 0
        if "changepoint" not in frame.columns:
            frame["changepoint"] = 0

    return frame


def load_skab_all(
    root: str | Path = SKAB_DIR,
    include_anomaly_free: bool = True,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load all SKAB CSV files and add source metadata columns."""
    root_path = Path(root)
    frames = []

    for csv_path in iter_skab_files(root_path, include_anomaly_free=include_anomaly_free):
        frame = load_skab_file(csv_path, parse_dates=parse_dates)
        frame["scenario"] = csv_path.parent.name
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No SKAB CSV files found in {root_path}")

    return pd.concat(frames, ignore_index=True)
