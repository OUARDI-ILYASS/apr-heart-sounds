"""Cluster validity indices - internal (no labels) and external (with labels).

Internal indices measure geometric quality; external indices measure agreement
with a known partition. Reporting only one of the two is the usual mistake:
internal indices can look excellent for clusters that have nothing to do with
the diagnosis, which is precisely the situation we expect and want to detect.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def purity_score(y_true: np.ndarray, y_cluster: np.ndarray) -> float:
    """Fraction of samples in the majority class of their assigned cluster.

    Purity rises monotonically with k, so it is only meaningful when comparing
    the same k. Reported alongside ARI/NMI, which are chance-corrected.
    """
    y_true = np.asarray(y_true)
    y_cluster = np.asarray(y_cluster)
    total = 0
    for cluster in np.unique(y_cluster):
        mask = y_cluster == cluster
        if mask.sum():
            counts = np.bincount(y_true[mask].astype(int))
            total += int(counts.max())
    return float(total / max(1, len(y_true)))


def cluster_validity(X: np.ndarray, labels: np.ndarray,
                     y_true: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None,
                     sample_size: int = 5000,
                     seed: int = 42) -> Dict[str, float]:
    """Compute all validity indices for one clustering.

    ``sample_size`` caps the silhouette computation, which is O(N^2) in memory
    and would otherwise be the slowest step in the whole pipeline.
    """
    from sklearn.metrics import (
        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
        adjusted_rand_score, normalized_mutual_info_score,
    )

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    out: Dict[str, float] = {}

    if len(np.unique(labels)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}

    if len(X) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample, labels_sample = X[idx], labels[idx]
    else:
        X_sample, labels_sample = X, labels

    # --- internal (geometry only) ----------------------------------------
    if len(np.unique(labels_sample)) >= 2:
        # Higher is better; range [-1, 1].
        out["silhouette"] = float(silhouette_score(X_sample, labels_sample))
        # Lower is better; ratio of within- to between-cluster scatter.
        out["davies_bouldin"] = float(davies_bouldin_score(X_sample, labels_sample))
        # Higher is better; variance ratio criterion.
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_sample, labels_sample))

    # --- external vs the diagnosis ---------------------------------------
    if y_true is not None:
        y_true = np.asarray(y_true)
        # ARI and NMI are chance-corrected, so 0 means "no better than random".
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))
        out["purity"] = purity_score(y_true, labels)

    # --- external vs the recording site ----------------------------------
    # The diagnostic test for confounding: if clusters track the acquisition
    # site more strongly than the diagnosis, the feature space is organised by
    # equipment, not pathology.
    if groups is not None:
        groups = np.asarray(groups)
        codes = {g: i for i, g in enumerate(np.unique(groups))}
        group_codes = np.array([codes[g] for g in groups])
        out["ari_vs_site"] = float(adjusted_rand_score(group_codes, labels))
        out["nmi_vs_site"] = float(normalized_mutual_info_score(group_codes, labels))
        out["purity_vs_site"] = purity_score(group_codes, labels)

    return out


def interpret_validity(row: Dict[str, float]) -> str:
    """One-sentence, human-readable reading of a validity row.

    Written into the phase summary so the clustering section of the paper has a
    defensible interpretation attached to the numbers rather than a bare table.
    """
    ari = row.get("ari")
    ari_site = row.get("ari_vs_site")
    silhouette = row.get("silhouette")

    parts = []
    if silhouette is not None:
        if silhouette < 0.1:
            parts.append(
                f"Silhouette {silhouette:.3f} indicates essentially no compact "
                "cluster structure - the feature cloud is close to unimodal."
            )
        elif silhouette < 0.25:
            parts.append(f"Silhouette {silhouette:.3f} indicates weak cluster structure.")
        else:
            parts.append(f"Silhouette {silhouette:.3f} indicates reasonably separated clusters.")

    if ari is not None:
        if ari < 0.05:
            parts.append(
                f"ARI vs diagnosis {ari:.3f} is near chance: the unsupervised "
                "partition does not recover the Normal/Abnormal split."
            )
        elif ari < 0.2:
            parts.append(f"ARI vs diagnosis {ari:.3f} shows weak but non-random alignment.")
        else:
            parts.append(f"ARI vs diagnosis {ari:.3f} shows substantial alignment.")

    if ari is not None and ari_site is not None and ari_site > ari + 0.05:
        parts.append(
            f"Crucially, alignment with recording site (ARI {ari_site:.3f}) exceeds "
            "alignment with diagnosis, so the dominant variance in this feature "
            "space is acquisition condition rather than pathology."
        )

    return " ".join(parts)
