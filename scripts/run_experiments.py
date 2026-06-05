"""CLI entry point for running all experiments.

Usage examples:
    python scripts/run_experiments.py --only smoke
    python scripts/run_experiments.py --only main --datasets skab
    python scripts/run_experiments.py --only grid --seeds 42 123
    python scripts/run_experiments.py          # runs main + grid (all)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when called as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.experiments.logging_utils import summary_table
from src.experiments.runner import run_grid, run_main, run_smoke


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="YazLab2 — run time-series anomaly detection experiments"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--only",
        choices=["main", "grid", "smoke", "all"],
        default="all",
        help="Which experiment set to run (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["skab", "batadal"],
        default=None,
        help="Datasets to include (default: both)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seed list (e.g. --seeds 42 123)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    progress = not args.no_progress
    kwargs = dict(datasets=args.datasets, seeds=args.seeds, progress=progress)

    if args.only == "smoke":
        print("Running smoke test…")
        run_smoke(cfg, progress=progress)

    elif args.only == "main":
        print("Running main comparison…")
        run_main(cfg, **kwargs)

    elif args.only == "grid":
        print("Running parameter grid…")
        run_grid(cfg, **kwargs)

    else:  # "all"
        print("Running main comparison…")
        run_main(cfg, **kwargs)
        print("Running parameter grid…")
        run_grid(cfg, **kwargs)

    print("\n=== Summary ===")
    tbl = summary_table(cfg)
    if not tbl.empty:
        print(tbl.to_string(index=False))
    else:
        print("(no runs logged yet)")


if __name__ == "__main__":
    main()
