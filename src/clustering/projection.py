"""Dimensionality reduction for visualising the feature spaces: PCA and t-SNE.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def fit_pca(X: np.ndarray, n_components: Optional[int] = None,
            variance_target: Optional[float] = None, seed: int = 42):
    """Fit PCA either to a fixed rank or to a target explained variance."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    if variance_target is not None:
        model = PCA(n_components=variance_target, svd_solver="full",
                    random_state=seed)
    else:
        model = PCA(n_components=n_components, random_state=seed)
    model.fit(X)
    return model


def pca_report(model) -> Dict[str, object]:
    """Explained-variance diagnostics.

    ``n_components_for_95pct`` is the headline number: it says how many
    effective dimensions the feature space really has. A 234-dimensional MFCC
    vector whose variance is 95% captured by 20 components is telling you the
    aggregation statistics are highly redundant.
    """
    ratios = np.asarray(model.explained_variance_ratio_)
    cumulative = np.cumsum(ratios)
    return {
        "n_components": int(model.n_components_),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "variance_pc1": float(ratios[0]) if ratios.size else 0.0,
        "variance_pc1_pc2": float(cumulative[1]) if ratios.size > 1 else 0.0,
        "n_components_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "n_components_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "n_components_for_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def fit_tsne(X: np.ndarray, cfg: Dict, seed: int = 42,
             subsample: Optional[int] = None
             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run t-SNE, returning (embedding, indices_used, metadata).

    Returning the indices is important: everything plotted alongside the
    embedding (colours, cluster assignments) must be subsampled identically or
    the plot silently mislabels points.
    """
    from sklearn.manifold import TSNE

    X = np.asarray(X, dtype=np.float64)
    tsne_cfg = cfg["clustering"]["tsne"]
    limit = subsample if subsample is not None else int(tsne_cfg.get("subsample", 3000))

    if len(X) > limit:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(X), limit, replace=False))
        X_sub = X[indices]
    else:
        indices = np.arange(len(X))
        X_sub = X

    # Perplexity must be < n_samples; sklearn raises otherwise.
    perplexity = min(float(tsne_cfg.get("perplexity", 30)), max(5.0, len(X_sub) / 4.0))

    # Recent sklearn renamed n_iter -> max_iter. Try the new name first so the
    # code works across versions without pinning.
    kwargs = dict(
        n_components=int(tsne_cfg.get("n_components", 2)),
        perplexity=perplexity,
        learning_rate=tsne_cfg.get("learning_rate", "auto"),
        init=str(tsne_cfg.get("init", "pca")),
        random_state=seed,
    )
    n_iter = int(tsne_cfg.get("n_iter", 1000))
    try:
        model = TSNE(max_iter=n_iter, **kwargs)
    except TypeError:
        model = TSNE(n_iter=n_iter, **kwargs)

    embedding = model.fit_transform(X_sub)
    meta = {
        "n_points": int(len(X_sub)),
        "subsampled": bool(len(X) > limit),
        "perplexity": float(perplexity),
        "kl_divergence": float(getattr(model, "kl_divergence_", np.nan)),
        "n_iter": n_iter,
        "init": str(tsne_cfg.get("init", "pca")),
    }
    return embedding, indices, meta


def project_for_plot(X: np.ndarray, cfg: Dict, seed: int = 42
                     ) -> Dict[str, object]:
    """Produce both 2-D projections plus their diagnostics in one call."""
    pca_model = fit_pca(X, n_components=min(50, X.shape[1], len(X)), seed=seed)
    pca_coords = pca_model.transform(np.asarray(X, dtype=np.float64))

    tsne_coords, tsne_indices, tsne_meta = fit_tsne(X, cfg, seed=seed)

    return {
        "pca_coords": pca_coords[:, :2],
        "pca_full": pca_coords,
        "pca_report": pca_report(pca_model),
        "pca_model": pca_model,
        "tsne_coords": tsne_coords,
        "tsne_indices": tsne_indices,
        "tsne_meta": tsne_meta,
    }


def reduce_for_clustering(X: np.ndarray, n_components: int, seed: int = 42
                          ) -> Tuple[np.ndarray, object]:
    """PCA-reduce a very high-dimensional space before k-means.

    Used for the flattened log-Mel maps (32 x 188 = 6016 dimensions). Euclidean
    distance in 6016 dimensions is dominated by noise accumulated across
    dimensions - the concentration-of-distances problem - so k-means there is
    close to meaningless. Reducing to ~50 components keeps the informative
    variance and makes the distances behave.
    """
    model = fit_pca(X, n_components=min(n_components, X.shape[1], len(X) - 1), seed=seed)
    return model.transform(np.asarray(X, dtype=np.float64)), model
