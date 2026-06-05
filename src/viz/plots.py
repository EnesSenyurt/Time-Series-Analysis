"""Evaluation visualisations (Enes's part of the viz layer).

Produces:
* Confusion matrix (sklearn ConfusionMatrixDisplay)
* ROC + Precision-Recall curves (joint figure)
* Parameter sensitivity heatmaps (window × alphabet → F1 / n_states / density)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 6.1  Confusion matrix
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(
    y_true,
    y_pred,
    path: str,
    title: str = "Confusion Matrix",
    labels: list[str] | None = None,
) -> str:
    """Save a confusion-matrix figure to *path* and return the path."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels or ["Normal", "Anomali"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 6.2  ROC + Precision-Recall (joint figure)
# --------------------------------------------------------------------------- #

def plot_roc_pr(
    y_true,
    y_score,
    path: str,
    label: str = "Model",
) -> str:
    """Save a 1×2 figure with ROC (left) and Precision-Recall (right) curves.

    Parameters
    ----------
    y_score : continuous anomaly score (higher = more anomalous).
              Works with both DL sigmoid probabilities and automaton
              neg-log-path scores.
    """
    from sklearn.metrics import (
        PrecisionRecallDisplay,
        RocCurveDisplay,
        average_precision_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))

    try:
        auc = roc_auc_score(y_true, y_score)
        RocCurveDisplay.from_predictions(
            y_true, y_score, name=f"{label} (AUC={auc:.3f})", ax=ax_roc
        )
    except ValueError:
        ax_roc.text(0.5, 0.5, "ROC hesaplanamadi", ha="center", transform=ax_roc.transAxes)
    ax_roc.set_title("ROC Egrisi")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)

    try:
        ap = average_precision_score(y_true, y_score)
        PrecisionRecallDisplay.from_predictions(
            y_true, y_score, name=f"{label} (AP={ap:.3f})", ax=ax_pr
        )
    except ValueError:
        ax_pr.text(0.5, 0.5, "PR hesaplanamadi", ha="center", transform=ax_pr.transAxes)
    ax_pr.set_title("Precision-Recall Egrisi")

    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 6.5  Parameter sensitivity
# --------------------------------------------------------------------------- #

def plot_param_sensitivity(runs_df: pd.DataFrame, path: str) -> str:
    """Save a sensitivity figure: window × alphabet heatmaps for F1, n_states,
    transition_density — one column per dataset.

    Expects *runs_df* to contain at least: dataset, window_size, alphabet_size,
    metrics_f1 (and optionally n_states, transition_density).
    """
    required = {"dataset", "window_size", "alphabet_size", "metrics_f1"}
    if not required.issubset(runs_df.columns):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Parametre grid verisi bulunamadi", ha="center")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    df = runs_df[runs_df["model"] == "automaton"].copy() if "model" in runs_df.columns else runs_df.copy()
    datasets = sorted(df["dataset"].unique())
    metrics_cols = ["metrics_f1", "n_states", "transition_density"]
    metric_labels = {"metrics_f1": "F1", "n_states": "N States", "transition_density": "Gecis Yogunlugu"}
    available_metrics = [c for c in metrics_cols if c in df.columns]

    n_rows = len(available_metrics)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for col_idx, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        agg = (
            sub.groupby(["window_size", "alphabet_size"])[available_metrics]
            .mean()
            .reset_index()
        )
        for row_idx, metric in enumerate(available_metrics):
            ax = axes[row_idx][col_idx]
            pivot = agg.pivot(index="window_size", columns="alphabet_size", values=metric)
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Alfabe Boyutu")
            ax.set_ylabel("Pencere Boyutu")
            ax.set_title(f"{ds.upper()} — {metric_labels.get(metric, metric)}")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
            fig.colorbar(im, ax=ax)

    fig.suptitle("Parametre Duyarlilik Analizi (Otomata)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
