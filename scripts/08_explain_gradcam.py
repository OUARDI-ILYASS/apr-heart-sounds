#!/usr/bin/env python3
"""Phase 08 - Grad-CAM explanations for the CNN, plus saliency sanity checks.

Produces per-example heatmaps, class-averaged maps, and two checks that decide
whether any of it means anything:

  * the Adebayo model-randomisation test (are the maps driven by learned
    weights or by architecture?)
  * layer sensitivity (does the conclusion survive a different target layer?)

Artifacts:
    results/xai/gradcam/gradcam_results.npz
    results/xai/gradcam/gradcam_analysis.json
    figures/fig_gradcam_*.pdf, figures/fig_average_cams.pdf

Run:  python scripts/08_explain_gradcam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.models.cnn import rebuild_from_checkpoint, resolve_device, HeartSoundCNN
from src.models.inference import predict_cnn
from src.features.logmel import LogMelExtractor
from src.xai.gradcam import (
    compute_gradcam_batch, average_cam_by_class, cam_frequency_profile,
    cam_time_profile, sanity_check_randomization, layer_sensitivity,
)
from src.xai.alignment import energy_confound_check
from src.visualization.explanations import plot_average_cams
from src.utils.io import (
    load_npy, load_checkpoint, save_json, save_npz, require_artifacts, ensure_dir,
)
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "08_explain_gradcam"


def main() -> int:
    cfg, logger, args = setup(PHASE, "Grad-CAM explanations and saliency sanity checks")
    summary = PhaseSummary(PHASE, cfg, next_phase="09_cycle_alignment")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    checkpoint_path = Path(cfg["_abs_paths"]["models_dir"]) / "cnn" / "cnn_logmel.pt"
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "xai" / "gradcam")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])

    require_artifacts([
        checkpoint_path,
        processed / "logmel" / "features_test.npy",
        interim / "segment_index.csv",
        interim / "segments_test.npy",
    ], phase=PHASE)

    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    test_index = index[index["split"] == "test"].reset_index(drop=True)
    summary.touch_split("test")

    X_test = load_npy(processed / "logmel" / "features_test.npy")
    audio_test = load_npy(interim / "segments_test.npy")
    y_test = test_index["label"].to_numpy()

    checkpoint = load_checkpoint(checkpoint_path)
    device = resolve_device(str(cfg["cnn"]["training"]["device"]))
    model = rebuild_from_checkpoint(checkpoint)
    logger.info(f"Loaded CNN from {checkpoint_path} (best epoch "
                f"{checkpoint.get('best_epoch')})")
    logger.info(f"Grad-CAM feature map: {model.feature_map_shape()} "
                "-> upsampled to the log-Mel grid")

    predictions = predict_cnn(model, X_test, device)["y_pred"]
    accuracy = float(np.mean(predictions == y_test))
    logger.info(f"Segment-level test accuracy: {accuracy:.4f}")

    gradcam_cfg = cfg["xai"]["gradcam"]
    sr = float(cfg["preprocessing"]["target_sr"])
    hop_length = int(cfg["features"]["hop_length"])
    extractor = LogMelExtractor(sr, cfg)
    band_frequencies = extractor.band_frequencies()
    frame_times = extractor.frame_times()

    # ---- Grad-CAM over the whole test set ---------------------------------
    logger.info("Computing Grad-CAM for the test set ...")
    with watch.section("gradcam"):
        result = compute_gradcam_batch(
            model, X_test, device, batch_size=32, method="gradcam",
            normalize=str(gradcam_cfg["normalize"]), progress=True,
        )
    cams = result["cams"]
    logger.info(f"Grad-CAM maps: {cams.shape}")

    # Grad-CAM++ as a robustness variant.
    logger.info("Computing Grad-CAM++ as a robustness check ...")
    with watch.section("gradcampp"):
        result_pp = compute_gradcam_batch(
            model, X_test[:500], device, batch_size=32, method="gradcampp",
            normalize=str(gradcam_cfg["normalize"]),
        )
    correlations = [
        float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        for a, b in zip(cams[:500], result_pp["cams"])
        if a.std() > 1e-12 and b.std() > 1e-12
    ]
    variant_agreement = float(np.mean(correlations)) if correlations else float("nan")
    logger.info(f"Grad-CAM vs Grad-CAM++ agreement: r={variant_agreement:.3f}")
    summary.add_finding("gradcam_vs_gradcampp_correlation", round(variant_agreement, 4),
                        "High agreement means the conclusion is not an artefact of "
                        "the specific CAM variant chosen")

    # ---- class-averaged maps ----------------------------------------------
    averages = average_cam_by_class(cams, y_test, predictions)
    for category in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        logger.info(f"  {category}: n={averages[f'{category}_count']}")

    plot_average_cams(averages, figures_dir / "fig_average_cams",
                      band_frequencies, frame_times, cfg)
    summary.add_artifact(figures_dir / "fig_average_cams.pdf")

    # ---- marginal profiles -------------------------------------------------
    profiles = {}
    for category in ["true_positive", "true_negative"]:
        if averages[f"{category}_count"] == 0:
            continue
        profiles[category] = {
            "frequency": cam_frequency_profile(averages[category], band_frequencies),
            "time": cam_time_profile(averages[category], frame_times),
        }
        logger.info(f"  {category}: frequency centroid "
                    f"{profiles[category]['frequency']['centroid_hz']:.0f} Hz, "
                    f"temporal concentration "
                    f"{profiles[category]['time']['concentration']:.3f}")

    # ---- SANITY CHECK 1: model randomisation -------------------------------
    logger.info("Running the Adebayo model-randomisation sanity check ...")
    with watch.section("sanity_randomization"):
        sanity = sanity_check_randomization(HeartSoundCNN, checkpoint, X_test,
                                            device, n_samples=100,
                                            seed=int(cfg["experiment"]["seed"]))
    logger.info(f"  correlation with randomly re-initialised model: "
                f"{sanity['mean_correlation_with_random_model']:.3f}")
    logger.info(f"  {sanity['interpretation']}")

    summary.add_assertion(
        "gradcam_passes_randomization_check", sanity["passes_sanity_check"],
        f"|r| with random model = "
        f"{abs(sanity['mean_correlation_with_random_model']):.3f} (threshold 0.30)",
    )
    if not sanity["passes_sanity_check"]:
        summary.add_warning(
            "Grad-CAM maps correlate strongly with those from a randomly "
            "initialised network. This means they are driven by architecture and "
            "input statistics rather than by learned weights. Every claim in the "
            "explainability section must be withdrawn or heavily qualified."
        )

    # ---- SANITY CHECK 2: layer sensitivity ---------------------------------
    logger.info("Checking sensitivity to the choice of target layer ...")
    with watch.section("layer_sensitivity"):
        layers = layer_sensitivity(model, X_test, device, n_samples=50)
    for row in layers:
        logger.info(f"  {row['layer']}: temporal concentration "
                    f"{row['mean_temporal_concentration']:.3f}, "
                    f"correlation with last layer "
                    f"{row['correlation_with_last']}")
    summary.add_table("layer_sensitivity", layers)

    # ---- SANITY CHECK 3: energy confound ------------------------------------
    logger.info("Checking whether attribution merely tracks signal energy ...")
    with watch.section("energy_confound"):
        confound = energy_confound_check(cams[:500], audio_test[:500],
                                         hop_length, X_test.shape[2])
    logger.info(f"  mean correlation with the RMS envelope: "
                f"{confound.get('mean_correlation_with_envelope', float('nan')):.3f}")
    logger.info(f"  {confound.get('interpretation', '')}")
    summary.add_finding("gradcam_energy_correlation",
                        round(confound.get("mean_correlation_with_envelope", float("nan")), 4),
                        "Correlation between the temporal attribution profile and the "
                        "signal envelope. High values would mean the CNN is an energy "
                        "detector rather than a murmur detector.")
    if confound.get("mean_correlation_with_envelope", 0) > 0.6:
        summary.add_warning(
            "Grad-CAM attribution closely tracks signal energy. Phase-specific "
            "claims about attending to systole are confounded with loudness and "
            "must be stated with that caveat."
        )

    # ---- persist -----------------------------------------------------------
    save_npz(results_dir / "gradcam_results.npz",
             cams=cams.astype(np.float32),
             predictions=predictions.astype(np.int64),
             labels=y_test.astype(np.int64))
    summary.add_artifact(results_dir / "gradcam_results.npz", array=cams)

    save_json(results_dir / "gradcam_analysis.json", {
        "n_segments": int(len(cams)),
        "segment_accuracy": accuracy,
        "feature_map_shape": list(model.feature_map_shape()),
        "category_counts": {k: v for k, v in averages.items() if k.endswith("_count")},
        "profiles": profiles,
        "sanity_check_randomization": sanity,
        "layer_sensitivity": layers,
        "energy_confound": confound,
        "gradcam_vs_gradcampp_correlation": variant_agreement,
        "band_frequencies_hz": band_frequencies.tolist(),
        "frame_times_s": frame_times.tolist(),
    })
    summary.add_artifact(results_dir / "gradcam_analysis.json")

    summary.add_parameters({
        "target_layer": str(gradcam_cfg["target_layer"]),
        "normalize": str(gradcam_cfg["normalize"]),
        "upsample_mode": str(gradcam_cfg["upsample_mode"]),
    })
    summary.add_note(
        "Grad-CAM's native resolution here is "
        f"{model.feature_map_shape()[1]} frequency x {model.feature_map_shape()[2]} "
        "time cells. We therefore make temporal claims only; frequency claims come "
        "from the PWP SHAP analysis in phase 07, where the band mapping is exact."
    )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
