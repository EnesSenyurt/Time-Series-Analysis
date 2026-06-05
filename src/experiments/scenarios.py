"""Scenario transformations for robustness experiments.

Three scenarios:
  original  -- no change (copy)
  noise     -- Gaussian noise added to each feature column of the test set
  unseen    -- DL receives original X; automaton unseen-rate is tracked separately
"""
from __future__ import annotations

import numpy as np


def apply_scenario(
    name: str,
    X: np.ndarray,
    cfg,
    rng: np.random.Generator,
    train_std: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a scenario transform to X (test data, already normalised).

    Parameters
    ----------
    name       : "original" | "noise" | "unseen"
    X          : (n_samples, n_features) — typically StandardScaler-normalised
    cfg        : ConfigNode
    rng        : numpy Generator (for reproducibility)
    train_std  : per-feature std from training data; defaults to 1.0 per feature
                 (appropriate when X is already standard-scaled)
    """
    X = np.asarray(X, dtype=float)

    if name in ("original", "unseen"):
        return X.copy()

    if name == "noise":
        if train_std is None:
            # For already-scaled X, each feature has std ≈ 1; clip zeros for
            # constant features so noise is always non-zero.
            raw_std = np.std(X, axis=0)
            std = np.where(raw_std > 1e-8, raw_std, 1.0)
        else:
            std = np.asarray(train_std, dtype=float)
        sigma = cfg.noise.sigma_ratio * std
        return X + rng.normal(0.0, sigma, X.shape)

    raise ValueError(f"Unknown scenario: {name!r}. Expected one of: original, noise, unseen.")


def make_unseen_report(automaton, test_patterns: list[str]) -> dict:
    """Count patterns in test_patterns that are absent from the automaton vocabulary.

    Returns
    -------
    {"n_total": int, "n_unseen": int, "unseen_rate": float}
    """
    n_total = len(test_patterns)
    n_unseen = sum(1 for p in test_patterns if p not in automaton.vocab)
    return {
        "n_total": n_total,
        "n_unseen": n_unseen,
        "unseen_rate": n_unseen / n_total if n_total > 0 else 0.0,
    }
