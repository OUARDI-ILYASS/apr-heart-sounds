"""Confusion matrices and other heatmap-style result figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .style import ieee_figure, save_figure, figure_size, apply_ieee_style


def plot_confusion_matrix(matrix: np.ndarray, path: str | Path,
                          class_names: Optional[List[str]] = None,
                          cfg: Optional[Dict] = None, title: str = "",
                          normalized: bool = False):
    """Annotated confusion matrix.

    We show row-normalised values (per-class recall) with the raw counts in
    parentheses. Under 3:1 imbalance a raw-count matrix is visually dominated
    by the Normal row and hides the minority-class performance entirely, which
    is exactly what a reader needs to see.
    """
    class_names = class_names or ["Normal", "Abnormal"]
    matrix = np.asarray(matrix)

    with ieee_figure(cfg, columns=1, height=2.6) as (fig, ax):
        display = (matrix.astype(float) /
                   np.maximum(matrix.sum(axis=1, keepdims=True), 1)
                   if not normalized else matrix)
        image = ax.imshow(display, cmap="Blues", vmin=0, vmax=1)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = (f"{display[i, j]:.2f}\n({int(matrix[i, j])})"
                        if not normalized else f"{matrix[i, j]:.2f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if display[i, j] > 0.5 else "black")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        if title:
            ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                     label="Fraction of true class")
    return save_figure(fig, path, cfg)


def plot_confusion_grid(matrices: Dict[str, np.ndarray], path: str | Path,
                        class_names: Optional[List[str]] = None,
                        cfg: Optional[Dict] = None):
    """All models' confusion matrices in one figure, for the results section."""
    apply_ieee_style(cfg)
    class_names = class_names or ["Normal", "Abnormal"]
    names = list(matrices)
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=figure_size(cfg, columns=2, height=2.0),
                             squeeze=False)
    for index, name in enumerate(names):
        ax = axes[0][index]
        matrix = np.asarray(matrices[name], dtype=float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if normalized[i, j] > 0.5 else "black")
        ax.set_title(name, fontsize=7)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=6)
        ax.set_yticklabels(class_names if index == 0 else [], fontsize=6)
        ax.set_xlabel("Predicted", fontsize=6)
        if index == 0:
            ax.set_ylabel("True", fontsize=6)
        ax.grid(False)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)


def plot_per_group_heatmap(rows: List[Dict], path: str | Path,
                           metric: str = "macc", cfg: Optional[Dict] = None):
    """Per-sub-database performance heatmap - the shortcut-learning check.

    A uniform row means the model generalises across sites. A row with one
    bright cell means it has learned that site's prevalence.
    """
    rows = [r for r in rows if metric in r]
    if not rows:
        return []

    with ieee_figure(cfg, columns=1, height=2.0) as (fig, ax):
        groups = [r["group"] for r in rows]
        values = np.array([[r[metric] for r in rows]], dtype=float)
        image = ax.imshow(values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
        for j, value in enumerate(values[0]):
            ax.text(j, 0, f"{value:.3f}\nn={rows[j].get('n', '?')}",
                    ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric.upper())
    return save_figure(fig, path, cfg)
