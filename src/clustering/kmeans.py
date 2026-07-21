"""K-means clustering of the feature spaces (course requirement).

PROFESSOR Q: "What is unsupervised clustering *for* in a supervised project?"
A: It answers a question the classifiers cannot: is the Normal/Abnormal
   distinction *geometrically present* in each feature space, before any label
   is used? If k=2 k-means recovers clusters that align with the diagnosis
   (high ARI/NMI), the representation itself separates the classes and a
   classifier is mostly reading off existing structure. If it does not - which
   is what we expect here - then the class boundary is a thin, supervised
   direction inside a space whose dominant variance is something else
   (recording site, noise level, heart rate). That is a genuinely useful
   negative result: it tells you the classifier is doing real work rather than
   thresholding an obvious cluster, and it sets expectations for how much a
   purely unsupervised approach could ever achieve on this task.

PROFESSOR Q: "Why should I not read a low ARI as failure?"
A: Because k-means optimises within-cluster variance, and the dominant
   variance in PCG features is not pathology - it is acquisition. We test that
   interpretation directly by also measuring cluster alignment with
   sub-database (site). If clusters align better with site than with diagnosis,
   we have identified *what* the feature space is actually organised by, which
   is a stronger statement than "clustering did not work".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42,
               n_init: int = 10, max_iter: int = 300):
    """Fit k-means. ``n_init`` restarts guard against poor initialisation."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init,
                   max_iter=max_iter, init="k-means++")
    model.fit(np.asarray(X, dtype=np.float64))
    return model


def sweep_k(X: np.ndarray, k_range: List[int], seed: int = 42,
            n_init: int = 10, max_iter: int = 300,
            y_true: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    """Fit k-means for each k and score every validity index.

    Returns one row per k, ready to be written straight into a results table.
    """
    from .validity import cluster_validity

    X = np.asarray(X, dtype=np.float64)
    rows: List[Dict[str, object]] = []

    for k in k_range:
        model = fit_kmeans(X, k, seed=seed, n_init=n_init, max_iter=max_iter)
        labels = model.labels_
        row: Dict[str, object] = {
            "k": int(k),
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
        }
        row.update(cluster_validity(X, labels, y_true=y_true, groups=groups))
        rows.append(row)

    return rows


def elbow_point(k_values: List[int], inertias: List[float]) -> int:
    """Pick the elbow by maximum distance to the line joining the endpoints.

    The classic 'look at the plot' elbow rule, made reproducible. We report it
    but we do not lean on it: on high-dimensional audio features the inertia
    curve is usually smooth and the elbow is weak, which is itself worth
    stating rather than pretending a clean elbow exists.
    """
    k = np.asarray(k_values, dtype=float)
    inertia = np.asarray(inertias, dtype=float)
    if len(k) < 3:
        return int(k[0])

    # Normalise both axes so the "distance to the chord" is scale-free.
    k_norm = (k - k.min()) / max(k.ptp(), 1e-12)
    i_norm = (inertia - inertia.min()) / max(inertia.ptp(), 1e-12)

    start = np.array([k_norm[0], i_norm[0]])
    end = np.array([k_norm[-1], i_norm[-1]])
    line = end - start
    line = line / (np.linalg.norm(line) + 1e-12)

    distances = []
    for point in np.stack([k_norm, i_norm], axis=1):
        vec = point - start
        projection = np.dot(vec, line) * line
        distances.append(float(np.linalg.norm(vec - projection)))

    return int(k[int(np.argmax(distances))])


def cluster_label_crosstab(cluster_labels: np.ndarray, y_true: np.ndarray,
                           class_names: Optional[List[str]] = None
                           ) -> List[Dict[str, object]]:
    """Contingency table of cluster vs true class, as table-ready rows."""
    cluster_labels = np.asarray(cluster_labels)
    y_true = np.asarray(y_true)
    class_names = class_names or [str(c) for c in np.unique(y_true)]

    rows = []
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        row: Dict[str, object] = {"cluster": int(cluster), "n": int(mask.sum())}
        for class_index, name in enumerate(class_names):
            count = int(np.sum(y_true[mask] == class_index))
            row[name] = count
            row[f"pct_{name}"] = round(100.0 * count / max(1, int(mask.sum())), 1)
        rows.append(row)
    return rows
