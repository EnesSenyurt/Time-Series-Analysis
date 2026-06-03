"""Shared pytest fixtures: config + small synthetic datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def skab_sample_df():
    """Synthetic SKAB-like frame with 4 source files across two valve groups."""
    rng = np.random.default_rng(0)
    rows = []
    for source_file in ["valve1/0.csv", "valve1/1.csv", "valve2/0.csv", "valve2/1.csv"]:
        group = source_file.split("/")[0]
        for i in range(60):
            rows.append(
                {
                    "datetime": pd.Timestamp("2020-03-09") + pd.Timedelta(seconds=i),
                    "Current": rng.normal(),
                    "Pressure": rng.normal(),
                    "anomaly": int(rng.random() < 0.4),
                    "changepoint": 0,
                    "source_group": group,
                    "source_file": source_file,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def batadal_sample_df():
    """Synthetic BATADAL-like time-ordered frame (100 hourly rows)."""
    n = 100
    t0 = pd.Timestamp("2016-07-04")
    return pd.DataFrame(
        {
            "DATETIME": [t0 + pd.Timedelta(hours=i) for i in range(n)],
            "L_T1": np.linspace(0.0, 1.0, n),
            "F_PU1": np.linspace(1.0, 2.0, n),
            "ATT_FLAG": [0] * 70 + [1] * 15 + [0] * 15,
        }
    )


@pytest.fixture
def train_X():
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))


@pytest.fixture
def test_X():
    rng = np.random.default_rng(2)
    return pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
