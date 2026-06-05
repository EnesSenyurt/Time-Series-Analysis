"""Experiment orchestration: loads data, runs DL and automaton pipelines,
logs every result to results/metrics/runs.jsonl.

Key optimisation: DL models and the automaton are trained ONCE per (seed, fold).
Each trained model/automaton is then scored on all three scenarios independently,
so training cost does not multiply with the number of scenarios.

Public entry points:
    run_main(cfg)   -- fixed params, all datasets × scenarios × seeds
    run_grid(cfg)   -- automaton param grid (window × alphabet)
    run_smoke(cfg)  -- quick sanity check (1 seed, 2 folds, short epochs)
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.experiments.logging_utils import log_run
from src.experiments.metrics import classification_metrics
from src.experiments.scenarios import apply_scenario, make_unseen_report


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _opt_threshold(scores: np.ndarray, y_true: np.ndarray, n_grid: int = 101) -> float:
    """Sweep thresholds in [min, max] of scores and return the F1-maximising one."""
    from sklearn.metrics import f1_score

    if len(scores) == 0:
        return 0.5
    best_t, best_f1 = float(scores.min()), 0.0
    for t in np.linspace(scores.min(), scores.max(), n_grid):
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _automaton_scores_and_labels(
    automaton, params, pc1, y, cfg
) -> tuple[np.ndarray, np.ndarray]:
    """Score a PC1 series and align scores with window-level labels.

    Returns (scores, y_labels) — both length len(patterns)-1.
    score[i] corresponds to the transition into patterns[i+1].
    """
    from src.automata.automaton import anomaly_scores, pattern_sequence

    pats = pattern_sequence(pc1, params)
    if len(pats) < 2:
        return np.array([]), np.array([], dtype=int)

    scores = anomaly_scores(automaton, pats, cfg.automaton.path_horizon)
    y = np.asarray(y)
    w = params.window_size
    y_win = np.array(
        [int(y[j : min(j + w, len(y))].max()) for j in range(len(pats))]
    )
    return scores, y_win[1:]


def _base_record(dataset, model, scenario, window_size, alphabet_size, seed, fold):
    return {
        "dataset": dataset,
        "model": model,
        "scenario": scenario,
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "seed": seed,
        "fold": fold,
    }


# --------------------------------------------------------------------------- #
# Trained-model wrappers (train once, score many times)
# --------------------------------------------------------------------------- #

def _train_dl(model_name, X_tr, y_tr, X_va, y_va, cfg, seed):
    """Train one DL model; return (model, threshold, n_features)."""
    from src.models.dl_models import (
        build_cnn1d, build_lstm, find_best_threshold,
        make_sequences, predict_dl, train_dl,
    )

    seq_len = cfg.dl.sequence_length
    X3_tr, y3_tr = make_sequences(X_tr, y_tr, seq_len)
    X3_va, y3_va = make_sequences(X_va, y_va, seq_len)
    if min(len(X3_tr), len(X3_va)) == 0:
        return None, None
    build_fn = {"lstm": build_lstm, "cnn1d": build_cnn1d}[model_name]
    model = build_fn(cfg, X_tr.shape[1])
    model, _ = train_dl(model, (X3_tr, y3_tr), (X3_va, y3_va), cfg, seed)
    threshold = find_best_threshold(model, X3_va, y3_va)
    return model, threshold


def _score_dl(model, threshold, X_te, y_te, cfg):
    """Window-level metrics for an already-trained DL model on new test data."""
    from src.models.dl_models import make_sequences, predict_dl

    if model is None:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    X3_te, y3_te = make_sequences(X_te, y_te, cfg.dl.sequence_length)
    if len(X3_te) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    proba = predict_dl(model, X3_te)
    return classification_metrics(y3_te, (proba >= threshold).astype(int))


def _build_automaton(pc1_tr, y_tr, cfg, window_size, alphabet_size):
    """Fit automaton on normal-only train; also fit val threshold."""
    from src.automata.automaton import build_automaton_from_segments

    if cfg.automaton.build_on == "normal_only":
        mask = np.asarray(y_tr) == 0
        segs = [pc1_tr[mask]]
    else:
        segs = [pc1_tr]

    if len(segs[0]) < window_size + 2:
        return None, None
    return build_automaton_from_segments(segs, cfg, window_size, alphabet_size)


def _score_automaton(automaton, params, pc1_va, y_va, pc1_te, y_te, cfg):
    """Threshold on val, score on test; return metrics + metadata dict."""
    from src.automata.automaton import pattern_sequence

    if automaton is None:
        empty = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"metrics": empty, "n_states": 0, "transition_density": 0.0, "unseen_rate": 0.0}

    val_scores, val_labels = _automaton_scores_and_labels(automaton, params, pc1_va, y_va, cfg)
    thr = _opt_threshold(val_scores, val_labels) if len(val_scores) > 0 else 0.5

    te_scores, te_labels = _automaton_scores_and_labels(automaton, params, pc1_te, y_te, cfg)
    y_pred = (te_scores >= thr).astype(int) if len(te_scores) > 0 else np.array([])
    metrics = (
        classification_metrics(te_labels, y_pred)
        if len(y_pred) > 0
        else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    )
    test_pats = pattern_sequence(pc1_te, params)
    unseen_rate = make_unseen_report(automaton, test_pats)["unseen_rate"]
    return {
        "metrics": metrics,
        "n_states": automaton.num_states,
        "transition_density": automaton.transition_density(),
        "unseen_rate": unseen_rate,
    }


# --------------------------------------------------------------------------- #
# SKAB pipeline
# --------------------------------------------------------------------------- #

def _run_skab(cfg, window_size, alphabet_size, seeds, scenarios, progress, cfg_dl=None):
    from src.data.data_loader import get_skab_features, load_skab
    from src.data.preprocess import fit_preprocess
    from src.data.splits import skab_folds

    if cfg_dl is None:
        cfg_dl = cfg

    df = load_skab(cfg)
    target = cfg.datasets.skab.target
    feat_cols = get_skab_features(df, cfg)
    folds = skab_folds(df, cfg)

    for seed in tqdm(seeds, desc="SKAB seeds", disable=not progress):
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            df_tr_full = df.iloc[train_idx].reset_index(drop=True)
            df_te = df.iloc[test_idx].reset_index(drop=True)
            n = len(df_tr_full)
            n_val = max(1, int(n * 0.2))
            df_tr = df_tr_full.iloc[: n - n_val]
            df_va = df_tr_full.iloc[n - n_val :]

            X_tr_raw = df_tr[feat_cols].values.astype(float)
            y_tr = df_tr[target].values.astype(int)
            X_va_raw = df_va[feat_cols].values.astype(float)
            y_va = df_va[target].values.astype(int)
            X_te_raw = df_te[feat_cols].values.astype(float)
            y_te = df_te[target].values.astype(int)

            pre = fit_preprocess(X_tr_raw, cfg)
            X_tr = pre.transform_multivariate(X_tr_raw)
            X_va = pre.transform_multivariate(X_va_raw)

            # --- train once per (seed, fold) ---
            dl_trained = {}
            for model_name in cfg.dl.models:
                dl_trained[model_name] = _train_dl(model_name, X_tr, y_tr, X_va, y_va, cfg_dl, seed)

            pc1_tr = pre.pca.transform(X_tr)[:, 0]
            pc1_va = pre.pca.transform(X_va)[:, 0]
            automaton, params = _build_automaton(pc1_tr, y_tr, cfg, window_size, alphabet_size)

            # --- score each scenario separately ---
            for scenario in scenarios:
                rng = np.random.default_rng(seed)
                X_te_s = apply_scenario(scenario, pre.transform_multivariate(X_te_raw), cfg, rng)
                pc1_te_s = pre.pca.transform(X_te_s)[:, 0]

                for model_name, (model, thr) in dl_trained.items():
                    metrics = _score_dl(model, thr, X_te_s, y_te, cfg_dl)
                    log_run(
                        {**_base_record("skab", model_name, scenario, window_size, alphabet_size, seed, fold_idx),
                         "metrics": metrics, "n_states": None, "transition_density": None, "unseen_rate": None},
                        cfg,
                    )

                result = _score_automaton(automaton, params, pc1_va, y_va, pc1_te_s, y_te, cfg)
                log_run(
                    {**_base_record("skab", "automaton", scenario, window_size, alphabet_size, seed, fold_idx),
                     **{k: result[k] for k in ("metrics", "n_states", "transition_density", "unseen_rate")}},
                    cfg,
                )


# --------------------------------------------------------------------------- #
# BATADAL pipeline
# --------------------------------------------------------------------------- #

def _run_batadal(cfg, window_size, alphabet_size, seeds, scenarios, progress, cfg_dl=None):
    from src.data.data_loader import get_batadal_features, load_batadal
    from src.data.preprocess import fit_preprocess
    from src.data.splits import batadal_split

    if cfg_dl is None:
        cfg_dl = cfg

    df = load_batadal(cfg)
    target = cfg.datasets.batadal.target
    feat_cols = get_batadal_features(df, cfg)
    df_tr, df_va, df_te = batadal_split(df, cfg)

    X_tr_raw = df_tr[feat_cols].values.astype(float)
    y_tr = df_tr[target].values.astype(int)
    X_va_raw = df_va[feat_cols].values.astype(float)
    y_va = df_va[target].values.astype(int)
    X_te_raw = df_te[feat_cols].values.astype(float)
    y_te = df_te[target].values.astype(int)

    pre = fit_preprocess(X_tr_raw, cfg)
    X_tr = pre.transform_multivariate(X_tr_raw)
    X_va = pre.transform_multivariate(X_va_raw)
    pc1_tr = pre.pca.transform(X_tr)[:, 0]
    pc1_va = pre.pca.transform(X_va)[:, 0]

    for seed in tqdm(seeds, desc="BATADAL seeds", disable=not progress):
        # --- train once per seed ---
        dl_trained = {}
        for model_name in cfg.dl.models:
            dl_trained[model_name] = _train_dl(model_name, X_tr, y_tr, X_va, y_va, cfg_dl, seed)

        automaton, params = _build_automaton(pc1_tr, y_tr, cfg, window_size, alphabet_size)

        for scenario in scenarios:
            rng = np.random.default_rng(seed)
            X_te_s = apply_scenario(scenario, pre.transform_multivariate(X_te_raw), cfg, rng)
            pc1_te_s = pre.pca.transform(X_te_s)[:, 0]

            for model_name, (model, thr) in dl_trained.items():
                metrics = _score_dl(model, thr, X_te_s, y_te, cfg_dl)
                log_run(
                    {**_base_record("batadal", model_name, scenario, window_size, alphabet_size, seed, 0),
                     "metrics": metrics, "n_states": None, "transition_density": None, "unseen_rate": None},
                    cfg,
                )

            result = _score_automaton(automaton, params, pc1_va, y_va, pc1_te_s, y_te, cfg)
            log_run(
                {**_base_record("batadal", "automaton", scenario, window_size, alphabet_size, seed, 0),
                 **{k: result[k] for k in ("metrics", "n_states", "transition_density", "unseen_rate")}},
                cfg,
            )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_main(cfg, datasets=None, seeds=None, progress=True):
    """Fixed params: all datasets × scenarios × seeds. Models trained once per fold."""
    w = cfg.fixed_params.window_size
    a = cfg.fixed_params.alphabet_size
    seeds = seeds or list(cfg.seed_list)
    scenarios = list(cfg.scenarios)
    ds = datasets or ["skab", "batadal"]

    if "skab" in ds:
        _run_skab(cfg, w, a, seeds, scenarios, progress)
    if "batadal" in ds:
        _run_batadal(cfg, w, a, seeds, scenarios, progress)


def run_grid(cfg, datasets=None, seeds=None, progress=True):
    """Automaton-only parameter grid: window × alphabet × datasets × seeds."""
    from src.data.data_loader import (
        get_batadal_features, get_skab_features, load_batadal, load_skab,
    )
    from src.data.preprocess import fit_preprocess
    from src.data.splits import batadal_split, skab_folds

    seeds = seeds or list(cfg.seed_list)
    ds = datasets or ["skab", "batadal"]
    combos = [(w, a) for w in cfg.param_grid.window_size for a in cfg.param_grid.alphabet_size]

    if "skab" in ds:
        df = load_skab(cfg)
        target = cfg.datasets.skab.target
        feat_cols = get_skab_features(df, cfg)
        folds = skab_folds(df, cfg)

        for seed in tqdm(seeds, desc="Grid SKAB seeds", disable=not progress):
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                df_tr_full = df.iloc[train_idx].reset_index(drop=True)
                df_te = df.iloc[test_idx].reset_index(drop=True)
                n = len(df_tr_full)
                n_val = max(1, int(n * 0.2))
                df_tr = df_tr_full.iloc[: n - n_val]
                df_va = df_tr_full.iloc[n - n_val :]

                pre = fit_preprocess(df_tr[feat_cols].values.astype(float), cfg)
                pc1_tr = pre.pca.transform(pre.transform_multivariate(df_tr[feat_cols].values.astype(float)))[:, 0]
                pc1_va = pre.pca.transform(pre.transform_multivariate(df_va[feat_cols].values.astype(float)))[:, 0]
                pc1_te = pre.pca.transform(pre.transform_multivariate(df_te[feat_cols].values.astype(float)))[:, 0]
                y_tr = df_tr[target].values.astype(int)
                y_va = df_va[target].values.astype(int)
                y_te = df_te[target].values.astype(int)

                for w, a in combos:
                    auto, params = _build_automaton(pc1_tr, y_tr, cfg, w, a)
                    result = _score_automaton(auto, params, pc1_va, y_va, pc1_te, y_te, cfg)
                    log_run(
                        {**_base_record("skab", "automaton", "original", w, a, seed, fold_idx),
                         **{k: result[k] for k in ("metrics", "n_states", "transition_density", "unseen_rate")}},
                        cfg,
                    )

    if "batadal" in ds:
        df = load_batadal(cfg)
        target = cfg.datasets.batadal.target
        feat_cols = get_batadal_features(df, cfg)
        df_tr, df_va, df_te = batadal_split(df, cfg)

        pre = fit_preprocess(df_tr[feat_cols].values.astype(float), cfg)
        pc1_tr = pre.pca.transform(pre.transform_multivariate(df_tr[feat_cols].values.astype(float)))[:, 0]
        pc1_va = pre.pca.transform(pre.transform_multivariate(df_va[feat_cols].values.astype(float)))[:, 0]
        pc1_te = pre.pca.transform(pre.transform_multivariate(df_te[feat_cols].values.astype(float)))[:, 0]
        y_tr = df_tr[target].values.astype(int)
        y_va = df_va[target].values.astype(int)
        y_te = df_te[target].values.astype(int)

        for seed in tqdm(seeds, desc="Grid BATADAL seeds", disable=not progress):
            for w, a in combos:
                auto, params = _build_automaton(pc1_tr, y_tr, cfg, w, a)
                result = _score_automaton(auto, params, pc1_va, y_va, pc1_te, y_te, cfg)
                log_run(
                    {**_base_record("batadal", "automaton", "original", w, a, seed, 0),
                     **{k: result[k] for k in ("metrics", "n_states", "transition_density", "unseen_rate")}},
                    cfg,
                )


def run_smoke(cfg, progress=True):
    """Minimal sanity check: 1 seed, ≤2 folds, 2 DL epochs, original scenario only."""
    from src.config import ConfigNode

    cfg_smoke = ConfigNode(dict(cfg))
    cfg_smoke.dl = ConfigNode(dict(cfg.dl))
    cfg_smoke.dl["epochs"] = 2
    cfg_smoke.datasets = ConfigNode(dict(cfg.datasets))
    cfg_smoke.datasets.skab = ConfigNode(dict(cfg.datasets.skab))
    cfg_smoke.datasets.skab.cv = ConfigNode(dict(cfg.datasets.skab.cv))
    cfg_smoke.datasets.skab.cv["n_splits"] = 2

    seeds = [cfg.seed_list[0]]
    w = cfg.fixed_params.window_size
    a = cfg.fixed_params.alphabet_size

    _run_skab(cfg_smoke, w, a, seeds, ["original"], progress, cfg_dl=cfg_smoke)
    _run_batadal(cfg_smoke, w, a, seeds, ["original"], progress, cfg_dl=cfg_smoke)
