"""Phase 06 - Evaluate every model on the held-out TEST split.

This is the first and only phase that opens the test data. Everything it
consumes - hyperparameters, thresholds, CNN weights - was frozen in phases 04
and 05 using training and validation data only. Nothing is tuned here; the
phase reads models, computes numbers, and stops.

What it produces:
    * recording-level and segment-level metrics for every model
    * bootstrap confidence intervals (resampled over recordings)
    * pairwise McNemar tests with Holm-Bonferroni correction
    * per-sub-database breakdown (the shortcut-learning check)
    * calibration curves and error analysis
    * a literature comparison table, with the non-comparability caveat attached

Run:  python scripts/06_evaluate_models.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.models.inference import (
    predict_classical, predict_cnn, aggregate_to_recording, segment_agreement,
)
from src.models.cnn import rebuild_from_checkpoint, resolve_device
from src.evaluation.metrics import (
    compute_metrics, baseline_metrics, per_group_metrics, compare_to_literature,
)
from src.evaluation.statistical import (
    bootstrap_ci, mcnemar_test, holm_bonferroni, effect_size_cohens_h,
)
from src.evaluation.confusion import confusion_matrix, error_analysis, calibration_curve
from src.evaluation.aggregation import (
    build_comparison_table, build_pairwise_table, rank_models,
)
from src.visualization.matrices import plot_confusion_grid, plot_per_group_heatmap
from src.visualization.curves import (
    plot_roc_curves, plot_pr_curves, plot_calibration, plot_metric_comparison,
)
from src.utils.io import (
    load_npy, load_json, load_joblib, load_checkpoint, save_json,
    require_artifacts, ensure_dir,
)
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "06_evaluate_models"


def main() -> int:
    cfg, logger, args = setup(PHASE, "Evaluate all models on the held-out test set")
    summary = PhaseSummary(PHASE, cfg, next_phase="07_explain_shap")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    models_dir = Path(cfg["_abs_paths"]["models_dir"])
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "evaluation")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])

    require_artifacts([interim / "segment_index.csv"], phase=PHASE)
    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    test_index = index[index["split"] == "test"].reset_index(drop=True)

    # THE audit line. Phase 06 is the only phase whose summary records "test".
    summary.touch_split("test")
    summary.add_note(
        "This is the first phase to read the test split. All model and threshold "
        "selection happened in phases 04-05 on training and validation data, as "
        "recorded in the splits_touched field of those phases' summaries."
    )

    y_test_segment = test_index["label"].to_numpy()
    recordings_test = test_index["recording_id"].to_numpy()
    sites_test = test_index["subdb"].to_numpy()

    logger.info(f"Test: {len(y_test_segment)} segments from "
                f"{len(np.unique(recordings_test))} recordings "
                f"({100 * y_test_segment.mean():.1f}% abnormal segments)")

    seed = int(cfg["experiment"]["seed"])
    primary_aggregation = str(cfg["evaluation"]["primary_aggregation"])
    results: dict = {}
    recording_predictions: dict = {}

    # ---- classical models --------------------------------------------------
    classical_dir = models_dir / "classical"
    thresholds = {}
    if (results_dir / "classical_cv.json").exists():
        classical_reports = load_json(results_dir / "classical_cv.json")
        thresholds = {tag: r.get("tuned_threshold", 0.5)
                      for tag, r in classical_reports.items()}

    for model_path in sorted(classical_dir.glob("*.joblib")):
        tag = model_path.stem
        domain = tag.split("_", 1)[1]
        feature_path = processed / domain / "features_test.npy"
        if not feature_path.exists():
            logger.warning(f"Skipping {tag}: {feature_path} missing")
            continue

        logger.info(f"--- {tag} ---")
        X_test = load_npy(feature_path)
        model = load_joblib(model_path)

        with watch.section(f"predict_{tag}"):
            predictions = predict_classical(model, X_test)

        results[tag] = _evaluate_one(
            tag, predictions["y_prob"], y_test_segment, recordings_test, sites_test,
            cfg, seed, threshold=thresholds.get(tag, 0.5), logger=logger,
        )
        recording_predictions[tag] = results[tag]["_recording_arrays"]
        summary.add_input(model_path)

    # ---- CNN ---------------------------------------------------------------
    checkpoint_path = models_dir / "cnn" / "cnn_logmel.pt"
    if checkpoint_path.exists():
        logger.info("--- cnn_logmel ---")
        X_test = load_npy(processed / "logmel" / "features_test.npy")
        checkpoint = load_checkpoint(checkpoint_path)
        device = resolve_device(str(cfg["cnn"]["training"]["device"]))
        model = rebuild_from_checkpoint(checkpoint)

        with watch.section("predict_cnn"):
            predictions = predict_cnn(model, X_test, device)

        cnn_threshold = 0.5
        if (results_dir / "cnn_training.json").exists():
            cnn_threshold = load_json(results_dir / "cnn_training.json").get(
                "tuned_threshold", 0.5
            )

        results["cnn_logmel"] = _evaluate_one(
            "cnn_logmel", predictions["y_prob"], y_test_segment, recordings_test,
            sites_test, cfg, seed, threshold=cnn_threshold, logger=logger,
        )
        recording_predictions["cnn_logmel"] = results["cnn_logmel"]["_recording_arrays"]
        summary.add_input(checkpoint_path)
    else:
        logger.warning("No CNN checkpoint found; skipping the deep model.")
        summary.add_warning("CNN checkpoint missing - the deep branch was not evaluated.")

    if not results:
        logger.error("No models were available to evaluate.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    # ---- trivial baselines -------------------------------------------------
    any_arrays = next(iter(recording_predictions.values()))
    y_test_recording = any_arrays["y_true"]
    baselines = baseline_metrics(y_test_recording)
    logger.info("Trivial baselines (recording level):")
    for name, metrics in baselines.items():
        logger.info(f"  {name:20s} acc={metrics['accuracy']:.4f} "
                    f"MAcc={metrics['macc']:.4f}")
    summary.add_table("trivial_baselines", [
        {"baseline": name, "accuracy": round(m["accuracy"], 4),
         "sensitivity": round(m["sensitivity"], 4),
         "specificity": round(m["specificity"], 4),
         "macc": round(m["macc"], 4)}
        for name, m in baselines.items()
    ])
    summary.add_finding(
        "majority_class_accuracy", round(baselines["always_normal"]["accuracy"], 4),
        "What a model that always predicts Normal achieves. Any reported accuracy "
        "must be read against this number, which is why MAcc is our primary metric."
    )

    # ---- pairwise significance ---------------------------------------------
    mcnemar_results = {}
    p_values = {}
    for a, b in combinations(sorted(recording_predictions), 2):
        arrays_a, arrays_b = recording_predictions[a], recording_predictions[b]
        result = mcnemar_test(arrays_a["y_true"], arrays_a["y_pred"], arrays_b["y_pred"],
                              correction=bool(cfg["evaluation"]["mcnemar"]["correction"]))
        result["effect_size_h"] = effect_size_cohens_h(
            results[a]["recording"]["macc"], results[b]["recording"]["macc"]
        )
        key = f"{a} vs {b}"
        mcnemar_results[key] = result
        p_values[key] = result["p_value"]
        logger.info(f"McNemar {key}: p={result['p_value']:.4f} "
                    f"({result['a_only_correct']} vs {result['b_only_correct']} discordant)")

    corrected = holm_bonferroni(p_values) if p_values else {}
    n_significant = sum(1 for v in corrected.values() if v["significant"])
    logger.info(f"{n_significant}/{len(corrected)} comparisons remain significant "
                "after Holm-Bonferroni correction")

    # ---- ranking and claim verdicts ----------------------------------------
    ranking = rank_models(results, metric="macc", level="recording")
    best = ranking[0]
    logger.info(f"Best model: {best['model']} (MAcc {best['macc']:.4f})")

    classical_best = max(
        (r for name, r in results.items() if not name.startswith("cnn")),
        key=lambda r: r["recording"]["macc"], default=None,
    )
    cnn_result = results.get("cnn_logmel")

    if cnn_result and classical_best:
        difference = cnn_result["recording"]["macc"] - classical_best["recording"]["macc"]
        pair_key = next((k for k in mcnemar_results if "cnn" in k), None)
        p_value = mcnemar_results[pair_key]["p_value"] if pair_key else 1.0
        if p_value < 0.05 and difference > 0:
            verdict = "supported"
        elif abs(difference) < 0.03 and p_value >= 0.05:
            verdict = "weak"
        else:
            verdict = "contradicted" if difference < 0 else "weak"
        summary.add_claim_verdict(
            "C2",
            "The log-Mel CNN outperforms the best classical feature/model pairing.",
            verdict,
            f"CNN MAcc {cnn_result['recording']['macc']:.4f} vs classical "
            f"{classical_best['recording']['macc']:.4f} "
            f"(delta {difference:+.4f}, McNemar p={p_value:.4f})",
            "If the difference is not significant, the honest conclusion is that "
            "the compact CNN and a well-tuned classical model are statistically "
            "indistinguishable on this dataset - which is itself worth reporting, "
            "since the classical model is far cheaper to train and to explain.",
        )

    summary.add_claim_verdict(
        "C3",
        "All models substantially exceed the majority-class baseline on MAcc.",
        "supported" if best["macc"] > 0.65 else "weak",
        f"best MAcc {best['macc']:.4f} vs 0.500 for the always-Normal baseline",
    )

    # ---- per-site check ----------------------------------------------------
    best_name = best["model"]
    best_arrays = recording_predictions[best_name]
    site_by_recording = (
        test_index.groupby("recording_id")["subdb"].first()
        .reindex(best_arrays["recording_ids"]).to_numpy()
    )
    group_rows = per_group_metrics(best_arrays["y_true"], best_arrays["y_pred"],
                                   site_by_recording, best_arrays["y_prob"])
    maccs = [r["macc"] for r in group_rows if "macc" in r]
    site_spread = (max(maccs) - min(maccs)) if maccs else 0.0
    logger.info(f"Per-site MAcc spread for {best_name}: {site_spread:.3f}")

    summary.add_table("per_subdatabase", group_rows)
    summary.add_finding(
        "site_macc_spread", round(site_spread, 4),
        "Range of MAcc across sub-databases for the best model. A large spread "
        "suggests the model exploits site-specific cues rather than pathology."
    )
    summary.add_assertion(
        "no_gross_site_shortcut", site_spread < 0.25,
        f"per-site MAcc spread = {site_spread:.3f}",
    )
    if site_spread >= 0.25:
        summary.add_warning(
            f"Per-site MAcc varies by {site_spread:.3f}. Report this prominently: "
            "it indicates the model does not transfer uniformly across acquisition "
            "conditions, and the pooled metric hides that."
        )

    # ---- figures -----------------------------------------------------------
    plot_confusion_grid(
        {name: confusion_matrix(a["y_true"], a["y_pred"])
         for name, a in recording_predictions.items()},
        figures_dir / "fig_confusion_matrices",
        list(cfg["dataset"]["class_names"]), cfg,
    )
    plot_roc_curves(recording_predictions, figures_dir / "fig_roc_curves", cfg)
    plot_pr_curves(recording_predictions, figures_dir / "fig_pr_curves", cfg)
    plot_metric_comparison(ranking, figures_dir / "fig_model_comparison", "macc", cfg)
    plot_per_group_heatmap(group_rows, figures_dir / "fig_per_site_macc", "macc", cfg)

    calibrations = {
        name: calibration_curve(a["y_true"], a["y_prob"])
        for name, a in recording_predictions.items()
    }
    plot_calibration(calibrations, figures_dir / "fig_calibration", cfg)
    for name, calibration in calibrations.items():
        logger.info(f"  {name}: ECE = {calibration['ece']:.4f}")
        summary.add_finding(f"{name}_ece", round(calibration["ece"], 4))

    for figure in ["fig_confusion_matrices", "fig_roc_curves", "fig_pr_curves",
                   "fig_model_comparison", "fig_per_site_macc", "fig_calibration"]:
        summary.add_artifact(figures_dir / f"{figure}.pdf")

    # ---- literature comparison ---------------------------------------------
    literature_rows = compare_to_literature(
        results[best_name]["recording"],
        {k: dict(v) for k, v in cfg["evaluation"]["literature_reference"].items()},
    )
    summary.add_table("literature_comparison", literature_rows)
    summary.add_note(
        "The literature rows in the comparison table were obtained on the official "
        "PhysioNet/CinC 2016 hidden test set, which was never publicly released. "
        "Our numbers come from a held-out split of the public training data. The "
        "two are therefore NOT directly comparable, and the table carries a "
        "'comparable' column saying so."
    )

    # ---- persist -----------------------------------------------------------
    serialisable = {
        name: {k: v for k, v in payload.items() if not k.startswith("_")}
        for name, payload in results.items()
    }
    save_json(results_dir / "test_results.json", {
        "models": serialisable,
        "baselines": baselines,
        "mcnemar": mcnemar_results,
        "holm_bonferroni": corrected,
        "ranking": ranking,
        "per_subdatabase": group_rows,
        "calibration": calibrations,
        "literature_comparison": literature_rows,
        "n_test_recordings": int(len(y_test_recording)),
        "n_test_segments": int(len(y_test_segment)),
    })
    summary.add_artifact(results_dir / "test_results.json")

    summary.add_table("test_results_recording_level",
                      build_comparison_table(results, level="recording"))
    summary.add_table("pairwise_mcnemar", build_pairwise_table(mcnemar_results))

    for name, payload in results.items():
        summary.add_finding(f"{name}_test_macc", round(payload["recording"]["macc"], 4))
    summary.add_finding("best_model", best_name)
    summary.add_finding("best_test_macc", round(best["macc"], 4))
    summary.add_finding("n_significant_comparisons", n_significant)

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


def _evaluate_one(tag, y_prob_segment, y_true_segment, recording_ids, sites,
                  cfg, seed, threshold, logger):
    """Compute the full metric bundle for one model's test predictions."""
    from sklearn.metrics import balanced_accuracy_score

    segment_metrics = compute_metrics(
        y_true_segment, (y_prob_segment >= 0.5).astype(int), y_prob_segment
    )

    aggregated = aggregate_to_recording(
        recording_ids, y_prob_segment, y_true_segment,
        method=str(cfg["evaluation"]["primary_aggregation"]), threshold=threshold,
    )
    recording_metrics = compute_metrics(
        aggregated["y_true"], aggregated["y_pred"], aggregated["y_prob"]
    )

    # Both aggregation rules, so the reader can see the choice does not decide
    # the conclusion.
    alternative = {}
    for method in cfg["evaluation"]["aggregation"]:
        alt = aggregate_to_recording(recording_ids, y_prob_segment, y_true_segment,
                                     method=str(method), threshold=threshold)
        alternative[str(method)] = compute_metrics(alt["y_true"], alt["y_pred"],
                                                   alt["y_prob"])

    bootstrap = {}
    if bool(cfg["evaluation"]["bootstrap"]["enabled"]):
        bootstrap["macc"] = bootstrap_ci(
            aggregated["y_true"], aggregated["y_pred"], balanced_accuracy_score,
            n_resamples=int(cfg["evaluation"]["bootstrap"]["n_resamples"]),
            ci=float(cfg["evaluation"]["bootstrap"]["ci"]), seed=seed,
        )

    agreement = segment_agreement(recording_ids, (y_prob_segment >= 0.5).astype(int))
    errors = error_analysis(aggregated["recording_ids"], aggregated["y_true"],
                            aggregated["y_pred"], aggregated["y_prob"])

    if logger is not None:
        ci = bootstrap.get("macc", {})
        logger.info(
            f"  test MAcc (recording) = {recording_metrics['macc']:.4f} "
            f"[{ci.get('lower', float('nan')):.3f}, {ci.get('upper', float('nan')):.3f}] "
            f"| Se {recording_metrics['sensitivity']:.3f} "
            f"Sp {recording_metrics['specificity']:.3f} "
            f"AUC {recording_metrics['roc_auc']:.3f}"
        )

    return {
        "segment": segment_metrics,
        "recording": recording_metrics,
        "by_aggregation": alternative,
        "bootstrap": bootstrap,
        "threshold_used": float(threshold),
        "segment_agreement": agreement,
        "error_analysis": errors,
        "_recording_arrays": {
            "recording_ids": aggregated["recording_ids"],
            "y_true": aggregated["y_true"],
            "y_pred": aggregated["y_pred"],
            "y_prob": aggregated["y_prob"],
        },
    }


if __name__ == "__main__":
    sys.exit(main())
