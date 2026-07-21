"""Cluster and projection visualisations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .style import ieee_figure, save_figure, figure_size, apply_ieee_style, class_colors


def plot_projection(coords: np.ndarray, labels: np.ndarray, path: str | Path,
                    title: str = "", cfg: Optional[Dict] = None,
                    label_names: Optional[List[str]] = None,
                    axis_labels: tuple = ("Dim 1", "Dim 2")):
    """2-D scatter coloured by a categorical variable."""
    label_names = label_names or ["Normal", "Abnormal"]
    colors = class_colors(cfg)

    with ieee_figure(cfg, columns=1, height=2.8) as (fig, ax):
        for value in np.unique(labels):
            mask = labels == value
            name = label_names[int(value)] if int(value) < len(label_names) else str(value)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=3, alpha=0.45, linewidths=0,
                       c=colors[int(value) % len(colors)], label=name)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if title:
            ax.set_title(title)
        ax.legend(markerscale=3, fontsize=6)
    return save_figure(fig, path, cfg)


def plot_projection_grid(projections: Dict[str, Dict], labels: np.ndarray,
                         path: str | Path, cfg: Optional[Dict] = None,
                         label_names: Optional[List[str]] = None):
    """One row per feature domain, PCA and t-SNE side by side.

    This is the course-required clustering figure. Laying the three feature
    domains out in a grid is what makes the comparison legible: the reader can
    see at a glance that none of the three separates the classes, which is the
    point being made.
    """
    apply_ieee_style(cfg)
    label_names = label_names or ["Normal", "Abnormal"]
    colors = class_colors(cfg)
    domains = list(projections)

    fig, axes = plt.subplots(len(domains), 2,
                             figsize=figure_size(cfg, columns=2,
                                                 height=2.4 * len(domains)),
                             squeeze=False)

    for row, domain in enumerate(domains):
        payload = projections[domain]
        for col, key in enumerate(["pca_coords", "tsne_coords"]):
            ax = axes[row][col]
            coords = np.asarray(payload[key])
            # t-SNE is fitted on a subsample, so its labels must be indexed
            # with the same subsample indices or the colours are meaningless.
            point_labels = (labels[payload["tsne_indices"]]
                            if key == "tsne_coords" and "tsne_indices" in payload
                            else labels[:len(coords)])
            for value in np.unique(point_labels):
                mask = point_labels == value
                ax.scatter(coords[mask, 0], coords[mask, 1], s=2, alpha=0.4,
                           linewidths=0, c=colors[int(value) % len(colors)],
                           label=label_names[int(value)] if row == 0 and col == 0 else None)
            method = "PCA" if col == 0 else "t-SNE"
            ax.set_title(f"{domain.upper()} — {method}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0 and "pca_report" in payload:
                variance = payload["pca_report"]["variance_pc1_pc2"]
                ax.set_xlabel(f"PC1+PC2 explain {100 * variance:.1f}% of variance",
                              fontsize=6)

    axes[0][0].legend(markerscale=4, fontsize=6, loc="best")
    fig.tight_layout(pad=0.4)
    return save_figure(fig, path, cfg)


def plot_kmeans_sweep(rows: List[Dict], path: str | Path,
                      cfg: Optional[Dict] = None, elbow_k: Optional[int] = None):
    """Inertia, silhouette and ARI against k.

    Showing all three together is deliberate: inertia always decreases,
    silhouette measures geometry, ARI measures agreement with the diagnosis.
    Plotting them on one axis row makes it obvious when geometry and diagnosis
    disagree - which is our main clustering finding.
    """
    apply_ieee_style(cfg)
    k_values = [r["k"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=figure_size(cfg, columns=2, height=2.0))

    axes[0].plot(k_values, [r["inertia"] for r in rows], marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    if elbow_k is not None:
        axes[0].axvline(elbow_k, color="r", ls="--", lw=0.7)
        axes[0].annotate(f"elbow k={elbow_k}", xy=(elbow_k, max(r["inertia"] for r in rows)),
                         fontsize=6, color="r")

    axes[1].plot(k_values, [r.get("silhouette", np.nan) for r in rows],
                 marker="o", color="#2E86AB")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    axes[2].plot(k_values, [r.get("ari", np.nan) for r in rows],
                 marker="o", label="vs diagnosis", color="#D7263D")
    if "ari_vs_site" in rows[0]:
        axes[2].plot(k_values, [r.get("ari_vs_site", np.nan) for r in rows],
                     marker="s", ls="--", label="vs site", color="#666666")
    axes[2].axhline(0, color="grey", ls=":", lw=0.6)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("ARI")
    axes[2].legend(fontsize=6)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)


def plot_cluster_composition(crosstab: List[Dict], path: str | Path,
                             cfg: Optional[Dict] = None,
                             label_names: Optional[List[str]] = None):
    """Stacked bars showing the class composition of each cluster."""
    label_names = label_names or ["Normal", "Abnormal"]
    colors = class_colors(cfg)

    with ieee_figure(cfg, columns=1, height=2.4) as (fig, ax):
        clusters = [str(r["cluster"]) for r in crosstab]
        bottom = np.zeros(len(crosstab))
        for index, name in enumerate(label_names):
            values = np.array([r.get(name, 0) for r in crosstab], dtype=float)
            ax.bar(clusters, values, bottom=bottom, label=name,
                   color=colors[index % len(colors)])
            bottom += values
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of segments")
        ax.legend(fontsize=6)
    return save_figure(fig, path, cfg)


def plot_pca_variance(report: Dict, path: str | Path, cfg: Optional[Dict] = None):
    """Scree plot with the 90/95% thresholds marked."""
    with ieee_figure(cfg, columns=1, height=2.4) as (fig, ax):
        cumulative = np.asarray(report["cumulative_variance"])
        components = np.arange(1, len(cumulative) + 1)
        ax.plot(components, cumulative, marker="o", ms=2)
        for threshold, style in [(0.90, ":"), (0.95, "--")]:
            ax.axhline(threshold, color="grey", ls=style, lw=0.6)
            ax.annotate(f"{int(100 * threshold)}%", xy=(components[-1], threshold),
                        fontsize=6, ha="right", va="bottom")
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_ylim(0, 1.02)
    return save_figure(fig, path, cfg)
