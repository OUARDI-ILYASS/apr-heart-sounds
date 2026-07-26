"""Phase 03 - K-means clustering and PCA / t-SNE visualisation (course requirement).

Answers: is the Normal/Abnormal distinction present in the feature geometry
*before* any label is used? And if the clusters are not the classes, what are
they? We test the obvious alternative - recording site - explicitly.

Clustering is fitted on the TRAINING split only. This is an unsupervised
analysis, but it is still part of the modelling process, and fitting it on the
full dataset would let test-set structure influence a figure in the paper.

Run:  python scripts/03_cluster_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.clustering.kmeans import sweep_k, fit_kmeans, elbow_point, cluster_label_crosstab
from src.clustering.projection import project_for_plot, reduce_for_clustering, fit_pca, pca_report
from src.clustering.validity import interpret_validity
from src.visualization.clusters import (
    plot_projection_grid, plot_kmeans_sweep, plot_cluster_composition, plot_pca_variance,
)
from src.utils.io import load_npy, save_json, require_artifacts, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "03_cluster_features"


def main() -> int:
    cfg, logger, args = setup(PHASE, "K-means clustering and feature-space visualisation")
    summary = PhaseSummary(PHASE, cfg, next_phase="04_train_classical")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "clustering")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])

    require_artifacts([interim / "segment_index.csv"], phase=PHASE)
    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    train_index = index[index["split"] == "train"].reset_index(drop=True)

    labels = train_index["label"].to_numpy()
    sites = train_index["subdb"].to_numpy()
    summary.touch_split("train")
    logger.info(f"Clustering on {len(labels)} training segments "
                f"({100 * labels.mean():.1f}% abnormal)")

    seed = int(cfg["experiment"]["seed"])
    k_range = list(cfg["clustering"]["k_range"])
    domains = ["mfcc", "logmel", "pwp"]
    projections = {}
    all_results = {}

    for domain in domains:
        feature_path = processed / domain / f"features_train.npy"
        if not feature_path.exists():
            logger.warning(f"Skipping {domain}: {feature_path} not found")
            continue

        logger.info(f"--- {domain} " + "-" * 50)
        X = load_npy(feature_path)
        summary.add_input(feature_path, shape=list(X.shape))

        # log-Mel is 2-D; flatten and PCA-reduce so k-means is meaningful.
        if X.ndim == 3:
            n = len(X)
            X_flat = X.reshape(n, -1)
            logger.info(f"  flattened {X.shape} -> {X_flat.shape}")
            with watch.section(f"{domain}_pca_reduce"):
                X_work, _ = reduce_for_clustering(
                    X_flat, int(cfg["clustering"]["logmel_flatten_pca_dim"]), seed
                )
            logger.info(f"  PCA-reduced to {X_work.shape} "
                        "(Euclidean distance in 6000-D is dominated by noise)")
            summary.add_note(
                f"log-Mel maps were flattened and PCA-reduced to "
                f"{X_work.shape[1]} dimensions before k-means, because k-means in "
                f"{X_flat.shape[1]}-dimensional space suffers from distance "
                "concentration."
            )
        else:
            X_work = X

        # ---- k sweep ----------------------------------------------------
        with watch.section(f"{domain}_kmeans"):
            rows = sweep_k(X_work, k_range, seed=seed,
                           n_init=int(cfg["clustering"]["n_init"]),
                           max_iter=int(cfg["clustering"]["max_iter"]),
                           y_true=labels, groups=sites)
        elbow = elbow_point([r["k"] for r in rows], [r["inertia"] for r in rows])

        for row in rows:
            logger.info(
                f"  k={row['k']}: silhouette={row.get('silhouette', float('nan')):.3f} "
                f"DB={row.get('davies_bouldin', float('nan')):.3f} "
                f"ARI(class)={row.get('ari', float('nan')):.3f} "
                f"ARI(site)={row.get('ari_vs_site', float('nan')):.3f}"
            )

        row_k2 = next(r for r in rows if r["k"] == 2)
        interpretation = interpret_validity(row_k2)
        logger.info(f"  k=2 reading: {interpretation}")

        # ---- k=2 detail --------------------------------------------------
        model_k2 = fit_kmeans(X_work, 2, seed=seed,
                              n_init=int(cfg["clustering"]["n_init"]))
        crosstab = cluster_label_crosstab(model_k2.labels_, labels,
                                          list(cfg["dataset"]["class_names"]))

        # ---- projections -------------------------------------------------
        with watch.section(f"{domain}_projection"):
            projection = project_for_plot(X_work, cfg, seed=seed)
        projections[domain] = projection

        logger.info(f"  PC1+PC2 explain {100 * projection['pca_report']['variance_pc1_pc2']:.1f}% "
                    f"of variance; {projection['pca_report']['n_components_for_95pct']} "
                    "components needed for 95%")

        # ---- persist -----------------------------------------------------
        payload = {
            "domain": domain,
            "n_samples": int(len(X_work)),
            "n_features_used": int(X_work.shape[1]),
            "sweep": rows,
            "elbow_k": elbow,
            "k2_crosstab": crosstab,
            "k2_interpretation": interpretation,
            "pca": projection["pca_report"],
            "tsne": projection["tsne_meta"],
        }
        save_json(results_dir / f"{domain}_clustering.json", payload)
        summary.add_artifact(results_dir / f"{domain}_clustering.json")
        all_results[domain] = payload

        plot_kmeans_sweep(rows, figures_dir / f"fig_kmeans_sweep_{domain}", cfg,
                          elbow_k=elbow)
        plot_cluster_composition(crosstab, figures_dir / f"fig_cluster_composition_{domain}",
                                 cfg, list(cfg["dataset"]["class_names"]))
        plot_pca_variance(projection["pca_report"],
                          figures_dir / f"fig_pca_variance_{domain}", cfg)

        summary.add_finding(f"{domain}_silhouette_k2", round(row_k2.get("silhouette", float("nan")), 4))
        summary.add_finding(f"{domain}_ari_class_k2", round(row_k2.get("ari", float("nan")), 4))
        summary.add_finding(f"{domain}_ari_site_k2", round(row_k2.get("ari_vs_site", float("nan")), 4))
        summary.add_finding(f"{domain}_pca_95pct_components",
                            projection["pca_report"]["n_components_for_95pct"])

    if not projections:
        logger.error("No feature domains were available to cluster.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    # ---- combined figure ---------------------------------------------------
    plot_projection_grid(projections, labels, figures_dir / "fig_feature_projections",
                         cfg, list(cfg["dataset"]["class_names"]))
    summary.add_artifact(figures_dir / "fig_feature_projections.pdf")

    # ---- claim verdict -----------------------------------------------------
    best_ari = max(
        (r["sweep"][k_range.index(2)].get("ari", 0.0) for r in all_results.values()),
        default=0.0,
    )
    site_aris = [r["sweep"][k_range.index(2)].get("ari_vs_site", 0.0) for r in all_results.values()]
    best_site_ari = max(site_aris) if site_aris else 0.0

    if best_ari > 0.3:
        verdict, evidence = "supported", f"best ARI vs diagnosis at k=2 is {best_ari:.3f}"
    elif best_ari > 0.1:
        verdict, evidence = "weak", f"best ARI vs diagnosis at k=2 is only {best_ari:.3f}"
    else:
        verdict, evidence = "contradicted", (
            f"best ARI vs diagnosis at k=2 is {best_ari:.3f}, close to chance; "
            f"ARI vs recording site reaches {best_site_ari:.3f}"
        )

    summary.add_claim_verdict(
        "C1",
        "The Normal/Abnormal distinction is recoverable by unsupervised clustering "
        "of the feature spaces.",
        verdict, evidence,
        "A contradicted verdict is a useful result: it establishes that the class "
        "boundary is a thin supervised direction rather than the dominant structure, "
        "which is what makes the supervised comparison in phases 04-06 non-trivial.",
    )

    summary.add_table("clustering_k2_summary", [
        {"domain": d,
         "silhouette": round(r["sweep"][k_range.index(2)].get("silhouette", float("nan")), 4),
         "ari_vs_class": round(r["sweep"][k_range.index(2)].get("ari", float("nan")), 4),
         "ari_vs_site": round(r["sweep"][k_range.index(2)].get("ari_vs_site", float("nan")), 4),
         "elbow_k": r["elbow_k"],
         "pca_95pct": r["pca"]["n_components_for_95pct"]}
        for d, r in all_results.items()
    ])

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
