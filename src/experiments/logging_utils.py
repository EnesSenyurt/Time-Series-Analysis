"""Experiment run logging to JSONL and summary aggregation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


_DEFAULT_RUNS_PATH = Path("results/metrics/runs.jsonl")


def _runs_path(cfg=None) -> Path:
    if cfg is not None:
        return Path(cfg.paths.metrics) / "runs.jsonl"
    return _DEFAULT_RUNS_PATH


def log_run(record: dict, cfg=None) -> None:
    """Append one experiment record to results/metrics/runs.jsonl.

    Expected keys: dataset, model, scenario, window_size, alphabet_size,
    seed, fold, metrics (dict), n_states, transition_density, unseen_rate.
    A timestamp is added automatically.
    """
    path = _runs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(record)
    entry["timestamp"] = datetime.now().isoformat()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_runs(cfg=None) -> pd.DataFrame:
    """Load all logged runs into a DataFrame. Returns empty frame if file missing."""
    path = _runs_path(cfg)
    if not path.exists():
        return pd.DataFrame()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    records = [json.loads(ln) for ln in lines]
    df = pd.json_normalize(records, sep="_")
    return df


def summary_table(cfg=None) -> pd.DataFrame:
    """Aggregate runs: mean ± std of F1 grouped by dataset/model/scenario.

    Returns a DataFrame with columns: dataset, model, scenario,
    window_size, alphabet_size, f1_mean, f1_std, n_runs.
    """
    df = load_runs(cfg)
    if df.empty:
        return df

    f1_col = "metrics_f1"
    if f1_col not in df.columns:
        return df

    group_cols = [c for c in ["dataset", "model", "scenario", "window_size", "alphabet_size"] if c in df.columns]
    agg = (
        df.groupby(group_cols)[f1_col]
        .agg(f1_mean="mean", f1_std="std", n_runs="count")
        .reset_index()
    )
    return agg.sort_values(["dataset", "model", "scenario"])
