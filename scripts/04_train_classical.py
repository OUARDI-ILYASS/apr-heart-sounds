"""Phase 04 - Train SVM and Random Forest on the MFCC and PWP feature sets.

Hyperparameters are selected by grouped, stratified cross-validation on the
TRAINING split, scored by balanced accuracy (= MAcc). Validation data is used
only to pick the decision threshold. The test split is not opened here.

Artifacts:
    models/classical/<model>_<domain>.joblib
    results/evaluation/classical_cv.json

Run:  python scripts/04_train_classical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.models.classical import run_grid_search, check_overfitting, predict_proba_safe
from src.models.inference import aggregate_to_recording, find_best_threshold
from src.evaluation.metrics import compute_metrics
from src.utils.io import load_npy, save_json, save_joblib, require_artifacts, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "04_train_classical"


def main() -> int:
    cfg, logger, args = setup(PHASE, "Train and tune SVM and Random Forest")
    summary = PhaseSummary(PHASE, cfg, next_phase="05_train_cnn")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    models_dir = ensure_dir(Path(cfg["_abs_paths"]["models_dir"]) / "classical")
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "evaluation")

    require_artifacts([interim / "segment_index.csv"], phase=PHASE)
    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    train_index = index[index["split"] == "train"].reset_index(drop=True)
    val_index = index[index["split"] == "val"].reset_index(drop=True)

    # IMPORTANT: only train and val are read in this phase.
    summary.touch_split("train", "val")

    y_train = train_index["label"].to_numpy()
    groups_train = train_index["recording_id"].to_numpy()
    y_val = val_index["label"].to_numpy()
    recordings_val = val_index["recording_id"].to_numpy()

    logger.info(f"Train: {len(y_train)} segments from {len(np.unique(groups_train))} recordings")
    logger.info(f"Val  : {len(y_val)} segments from {len(np.unique(recordings_val))} recordings")

    seed = int(cfg["experiment"]["seed"])
    feature_sets = list(cfg["classical"]["feature_sets"])
    all_reports = {}

    for domain in feature_sets:
        train_path = processed / domain / "features_train.npy"
        val_path = processed / domain / "features_val.npy"
        if not train_path.exists():
            logger.warning(f"Skipping {domain}: features not found")
            continue

        X_train = load_npy(train_path)
        X_val = load_npy(val_path)
        if X_train.ndim != 2:
            logger.warning(f"Skipping {domain}: expected 2-D features, got {X_train.shape}")
            continue

        summary.add_input(train_path, shape=list(X_train.shape))
        logger.info(f"=== {domain}: {X_train.shape[1]} features " + "=" * 40)

        for model_name in ["svm", "rf"]:
            tag = f"{model_name}_{domain}"
            logger.info(f"--- {tag} ---")

            with watch.section(f"gridsearch_{tag}"):
                model, report = run_grid_search(          # was: search, report
                    model_name, X_train, y_train, groups_train, cfg,
                    seed=seed, verbose=1 if not args.quiet else 0,
                    logger=logger,
                )

            logger.info(f"  best params : {report['best_params']}")
            logger.info(f"  CV MAcc     : {report['best_cv_score']:.4f} "
                        f"± {report['best_cv_std']:.4f}")
            logger.info(f"  train MAcc  : {report['best_train_score']:.4f} "
                        f"(gap {report['overfit_gap']:+.4f})")

            warning = check_overfitting(report)
            if warning:
                logger.warning(f"  {warning}")
                summary.add_warning(warning)

            # ---- validation: segment and recording level -----------------
            # every later `search.best_estimator_` becomes `model`
            y_prob_val = predict_proba_safe(model, X_val)

            

            segment_metrics = compute_metrics(y_val, (y_prob_val >= 0.5).astype(int),
                                              y_prob_val)

            aggregated = aggregate_to_recording(
                recordings_val, y_prob_val, y_val,
                method=str(cfg["evaluation"]["primary_aggregation"]),
            )
            recording_metrics = compute_metrics(
                aggregated["y_true"], aggregated["y_pred"], aggregated["y_prob"]
            )

            # Threshold chosen on VALIDATION only, then frozen for phase 06.
            threshold, threshold_score = find_best_threshold(
                aggregated["y_true"], aggregated["y_prob"], metric="macc"
            )
            logger.info(f"  val MAcc (segment)   : {segment_metrics['macc']:.4f}")
            logger.info(f"  val MAcc (recording) : {recording_metrics['macc']:.4f}")
            logger.info(f"  tuned threshold      : {threshold:.2f} "
                        f"-> val MAcc {threshold_score:.4f}")

            report.update(
                val_segment=segment_metrics,
                val_recording=recording_metrics,
                tuned_threshold=threshold,
                tuned_threshold_val_macc=threshold_score,
                feature_domain=domain,
                n_features=int(X_train.shape[1]),
            )
            all_reports[tag] = report

            model_path = models_dir / f"{tag}.joblib"
            save_joblib(model_path, model)
            save_json(models_dir / f"{tag}_report.json", report)
            summary.add_artifact(model_path)

            summary.add_finding(f"{tag}_cv_macc", round(report["best_cv_score"], 4))
            summary.add_finding(f"{tag}_val_macc_recording",
                                round(recording_metrics["macc"], 4))
            summary.add_finding(f"{tag}_threshold", round(threshold, 3))

    if not all_reports:
        logger.error("No classical models were trained.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    save_json(results_dir / "classical_cv.json", all_reports)
    summary.add_artifact(results_dir / "classical_cv.json")

    summary.add_table("classical_validation", [
        {"model": tag,
         "features": r["feature_domain"],
         "n_features": r["n_features"],
         "cv_macc": round(r["best_cv_score"], 4),
         "cv_std": round(r["best_cv_std"], 4),
         "val_macc_recording": round(r["val_recording"]["macc"], 4),
         "val_sensitivity": round(r["val_recording"]["sensitivity"], 4),
         "val_specificity": round(r["val_recording"]["specificity"], 4),
         "overfit_gap": round(r["overfit_gap"], 4)}
        for tag, r in sorted(all_reports.items(),
                             key=lambda kv: -kv[1]["val_recording"]["macc"])
    ])

    best_tag = max(all_reports, key=lambda t: all_reports[t]["val_recording"]["macc"])
    summary.add_finding("best_classical_model", best_tag)
    summary.add_note(
        "All numbers in this phase are cross-validation or validation scores. "
        "The test split has not been read. Hyperparameter and threshold choices "
        "are now frozen."
    )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
