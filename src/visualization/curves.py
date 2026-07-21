"""Training curves, ROC/PR curves and calibration plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .style import ieee_figure, save_figure, figure_size, apply_ieee_style


def plot_training_curves(history: List[Dict], path: str | Path,
                         cfg: Optional[Dict] = None, best_epoch: Optional[int] = None):
    """Loss and MAcc against epoch, for train and validation.

    This figure is the evidence behind any statement about overfitting. We mark
    the early-stopping epoch so the reader can see that the reported weights
    come from the best validation point, not the last one.
    """
    apply_ieee_style(cfg)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(cfg, columns=2, height=2.4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_macc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_macc"] for h in history], label="validation")
    axes[1].axhline(0.5, color="grey", ls=":", lw=0.7, label="chance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAcc")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, color="k", ls="--", lw=0.6, alpha=0.6)
        axes[1].annotate(f"best epoch {best_epoch}", xy=(best_epoch, 0.45),
                         fontsize=6, rotation=90, va="bottom", ha="right")

    fig.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)


def plot_roc_curves(results: Dict[str, Dict], path: str | Path,
                    cfg: Optional[Dict] = None):
    """Overlaid ROC curves.

    Under class imbalance ROC is optimistic - it uses the true-negative rate,
    and true negatives are abundant here - so we always publish the PR curve
    next to it rather than instead of it.
    """
    from sklearn.metrics import roc_curve, auc

    with ieee_figure(cfg, columns=1, height=2.8) as (fig, ax):
        for name, payload in results.items():
            y_true, y_prob = payload["y_true"], payload["y_prob"]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.7, label="chance")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", fontsize=6)
    return save_figure(fig, path, cfg)


def plot_pr_curves(results: Dict[str, Dict], path: str | Path,
                   cfg: Optional[Dict] = None):
    """Precision-recall curves with the prevalence baseline marked."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    with ieee_figure(cfg, columns=1, height=2.8) as (fig, ax):
        prevalence = None
        for name, payload in results.items():
            y_true, y_prob = payload["y_true"], payload["y_prob"]
            if len(np.unique(y_true)) < 2:
                continue
            prevalence = float(np.mean(y_true))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
        if prevalence is not None:
            ax.axhline(prevalence, color="grey", ls=":", lw=0.7,
                       label=f"prevalence ({prevalence:.2f})")
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision (PPV)")
        ax.legend(loc="lower left", fontsize=6)
    return save_figure(fig, path, cfg)


def plot_calibration(calibrations: Dict[str, Dict], path: str | Path,
                     cfg: Optional[Dict] = None):
    """Reliability diagram. Perfect calibration lies on the diagonal."""
    with ieee_figure(cfg, columns=1, height=2.8) as (fig, ax):
        for name, calibration in calibrations.items():
            ax.plot(calibration["confidence"], calibration["accuracy"],
                    marker="o", label=f"{name} (ECE={calibration['ece']:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.7, label="perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency of abnormal")
        ax.legend(fontsize=6)
    return save_figure(fig, path, cfg)


def plot_metric_comparison(rows: List[Dict], path: str | Path,
                           metric: str = "macc", cfg: Optional[Dict] = None):
    """Bar chart of one metric per model, with bootstrap CIs as error bars."""
    with ieee_figure(cfg, columns=1, height=2.6) as (fig, ax):
        names = [r["model"] for r in rows]
        values = [r.get(metric, 0) or 0 for r in rows]
        positions = np.arange(len(names))

        errors = None
        if all("ci_lower" in r and r["ci_lower"] is not None for r in rows):
            lower = [v - r["ci_lower"] for v, r in zip(values, rows)]
            upper = [r["ci_upper"] - v for v, r in zip(values, rows)]
            errors = np.array([lower, upper])

        ax.barh(positions, values, xerr=errors, color="#2E86AB",
                error_kw={"lw": 0.7, "capsize": 2})
        ax.axvline(0.5, color="grey", ls=":", lw=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_xlabel(metric.upper())
        ax.set_xlim(0.4, 1.0)
        ax.invert_yaxis()
    return save_figure(fig, path, cfg)
