"""Generate all report figures and save to results/figures/.

Runs a single-seed, single-fold evaluation to obtain predictions/scores, then
produces every figure type required by the rubric:

  CM         confusion_matrix_{dataset}_{model}.png
  ROC/PR     roc_pr_{dataset}_{model}.png
  Automata   automaton_diagram_{dataset}.png
             transition_heatmap_{dataset}.png
             mermaid_{dataset}.txt
  Sensitivity param_sensitivity.png  (from runs.jsonl grid data)

Usage:
    python scripts/make_report_assets.py [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.experiments.logging_utils import load_runs


def _quick_eval(cfg, dataset_name: str, seed: int):
    """Run one fold/split and return {model: (y_true, y_pred, y_score)} + automaton."""
    from src.automata.automaton import (
        anomaly_scores,
        build_automaton_from_segments,
        pattern_sequence,
    )
    from src.data.preprocess import fit_preprocess
    from src.experiments.runner import _automaton_scores_and_labels, _opt_threshold
    from src.models.dl_models import (
        build_cnn1d,
        build_lstm,
        find_best_threshold,
        make_sequences,
        predict_dl,
        train_dl,
    )

    if dataset_name == "skab":
        from src.data.data_loader import get_skab_features, load_skab
        from src.data.splits import skab_folds

        df = load_skab(cfg)
        target = cfg.datasets.skab.target
        feat_cols = get_skab_features(df, cfg)
        folds = skab_folds(df, cfg)
        train_idx, test_idx = folds[0]
        df_tr_full = df.iloc[train_idx].reset_index(drop=True)
        df_te = df.iloc[test_idx].reset_index(drop=True)
        n = len(df_tr_full)
        n_val = max(1, int(n * 0.2))
        df_tr = df_tr_full.iloc[: n - n_val]
        df_va = df_tr_full.iloc[n - n_val :]
    else:
        from src.data.data_loader import get_batadal_features, load_batadal
        from src.data.splits import batadal_split

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
    X_te = pre.transform_multivariate(X_te_raw)

    seq_len = cfg.dl.sequence_length
    results = {}

    for model_name in cfg.dl.models:
        X3_tr, y3_tr = make_sequences(X_tr, y_tr, seq_len)
        X3_va, y3_va = make_sequences(X_va, y_va, seq_len)
        X3_te, y3_te = make_sequences(X_te, y_te, seq_len)
        if min(len(X3_tr), len(X3_va), len(X3_te)) == 0:
            continue
        build_fn = {"lstm": build_lstm, "cnn1d": build_cnn1d}[model_name]

        # Use fast config for asset generation (avoid 50-epoch wait)
        from src.config import ConfigNode
        cfg_fast = ConfigNode(dict(cfg))
        cfg_fast.dl = ConfigNode(dict(cfg.dl))
        cfg_fast.dl["epochs"] = 5

        model = build_fn(cfg_fast, X_tr.shape[1])
        model, _ = train_dl(model, (X3_tr, y3_tr), (X3_va, y3_va), cfg_fast, seed)
        threshold = find_best_threshold(model, X3_va, y3_va)
        proba = predict_dl(model, X3_te)
        y_pred = (proba >= threshold).astype(int)
        results[model_name] = {"y_true": y3_te, "y_pred": y_pred, "y_score": proba}

    w = cfg.fixed_params.window_size
    a = cfg.fixed_params.alphabet_size
    pc1_tr = pre.pca.transform(X_tr)[:, 0]
    pc1_va = pre.pca.transform(X_va)[:, 0]
    pc1_te = pre.pca.transform(X_te)[:, 0]
    normal_mask = y_tr == 0
    automaton, params = build_automaton_from_segments([pc1_tr[normal_mask]], cfg, w, a)
    val_scores, val_labels = _automaton_scores_and_labels(automaton, params, pc1_va, y_va, cfg)
    test_scores, test_labels = _automaton_scores_and_labels(automaton, params, pc1_te, y_te, cfg)
    thr = _opt_threshold(val_scores, val_labels) if len(val_scores) > 0 else 0.5
    y_pred_auto = (test_scores >= thr).astype(int) if len(test_scores) > 0 else np.array([])
    results["automaton"] = {
        "y_true": test_labels,
        "y_pred": y_pred_auto,
        "y_score": test_scores,
        "automaton_obj": automaton,
    }

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate all report figures.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    from src.viz.automata_plots import plot_automaton, plot_transition_heatmap, to_mermaid
    from src.viz.plots import plot_confusion_matrix, plot_param_sensitivity, plot_roc_pr

    for ds in ["skab", "batadal"]:
        print(f"\n[{ds.upper()}] figürler üretiliyor…")
        try:
            results = _quick_eval(cfg, ds, args.seed)
        except Exception as exc:
            print(f"  HATA: {exc}")
            continue

        for model_name, res in results.items():
            y_true = res["y_true"]
            y_pred = res["y_pred"]
            y_score = res["y_score"]
            if len(y_true) == 0:
                continue

            cm_path = fig_dir / f"confusion_matrix_{ds}_{model_name}.png"
            plot_confusion_matrix(
                y_true, y_pred, str(cm_path),
                title=f"CM — {ds.upper()} / {model_name}",
            )
            print(f"  ✓ {cm_path.name}")

            roc_path = fig_dir / f"roc_pr_{ds}_{model_name}.png"
            plot_roc_pr(y_true, y_score, str(roc_path), label=f"{ds.upper()} / {model_name}")
            print(f"  ✓ {roc_path.name}")

        if "automaton" in results and "automaton_obj" in results["automaton"]:
            auto = results["automaton"]["automaton_obj"]
            diag_path = fig_dir / f"automaton_diagram_{ds}.png"
            plot_automaton(auto, str(diag_path))
            print(f"  ✓ {diag_path.name}")

            heat_path = fig_dir / f"transition_heatmap_{ds}.png"
            plot_transition_heatmap(auto, str(heat_path))
            print(f"  ✓ {heat_path.name}")

            mermaid_path = fig_dir / f"mermaid_{ds}.txt"
            mermaid_path.write_text(to_mermaid(auto), encoding="utf-8")
            print(f"  ✓ {mermaid_path.name}")

    print("\n[SENSITIVITY] parametre duyarlilik grafiği üretiliyor…")
    runs_df = load_runs(cfg)
    sens_path = fig_dir / "param_sensitivity.png"
    plot_param_sensitivity(runs_df, str(sens_path))
    print(f"  ✓ {sens_path.name}")

    print(f"\nTüm figürler kaydedildi: {fig_dir}")


if __name__ == "__main__":
    main()
