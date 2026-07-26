"""Phase 09 - Cardiac-cycle alignment: the quantitative XAI evaluation.

Segments the test audio into S1 / systole / S2 / diastole with a lightweight
envelope-based segmenter (NOT the Springer HSMM - see src/xai/segmenter.py),
then measures what fraction of each model's attribution mass falls in each
state, against uniform and shuffled nulls.

This is the phase that turns "the model looks at the right places" from an
assertion into a hypothesis test.

Artifacts:
    results/segmentation/test_segmentation.json
    results/xai/alignment.json
    figures/fig_alignment_*.pdf, figures/fig_gradcam_examples_*.pdf

Run:  python scripts/09_cycle_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.features.logmel import LogMelExtractor
from src.xai.segmenter import batch_segment, segmentation_quality_report
from src.xai.alignment import (
    batch_alignment, stratified_alignment, alignment_significance,
    build_alignment_table, uniform_null,
)
from src.visualization.explanations import (
    plot_alignment, plot_alignment_comparison, plot_gradcam_example,
    plot_segmentation_example,
)
from src.utils.io import (
    load_npy, load_npz, load_json, save_json, require_artifacts, ensure_dir,
)
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "09_cycle_alignment"


def main() -> int:
    cfg, logger, args = setup(PHASE, "Cardiac-cycle alignment of model attributions")
    summary = PhaseSummary(PHASE, cfg, next_phase="10_run_ablations")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    results_root = Path(cfg["_abs_paths"]["results_dir"])
    gradcam_path = results_root / "xai" / "gradcam" / "gradcam_results.npz"
    segmentation_dir = ensure_dir(results_root / "segmentation")
    xai_dir = ensure_dir(results_root / "xai")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])
    supplementary_dir = ensure_dir(Path(cfg["_abs_paths"]["figures_dir"]) / "supplementary")

    require_artifacts([
        gradcam_path,
        interim / "segments_test.npy",
        interim / "segment_index.csv",
    ], phase=PHASE)

    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    test_index = index[index["split"] == "test"].reset_index(drop=True)
    summary.touch_split("test")

    audio_test = load_npy(interim / "segments_test.npy")
    logmel_test = load_npy(processed / "logmel" / "features_test.npy")
    gradcam = load_npz(gradcam_path)
    cams = gradcam["cams"]
    predictions = gradcam["predictions"]
    labels = gradcam["labels"]

    sr = float(cfg["preprocessing"]["target_sr"])
    hop_length = int(cfg["features"]["hop_length"])
    n_frames = logmel_test.shape[2]
    seed = int(cfg["experiment"]["seed"])

    # ---- segment the test audio -------------------------------------------
    logger.info(f"Segmenting {len(audio_test)} test windows into cardiac states ...")
    logger.info("NOTE: this is an envelope-based segmenter, not the Springer HSMM. "
                "It is used only for evaluation, never for training.")
    with watch.section("segmentation"):
        segmentations = batch_segment(audio_test, sr, cfg, progress=True)

    quality = segmentation_quality_report(segmentations)
    logger.info(f"  usable: {quality['n_usable']}/{quality['n_segments']} "
                f"({100 * quality['usable_fraction']:.1f}%)")
    logger.info(f"  mean confidence: {quality['mean_confidence']:.3f}")
    logger.info(f"  mean heart rate: {quality['mean_heart_rate_bpm']:.1f} bpm "
                f"(range {quality['heart_rate_range'][0]:.0f}-"
                f"{quality['heart_rate_range'][1]:.0f})")

    save_json(segmentation_dir / "test_segmentation.json", {
        "quality": quality,
        "method": str(cfg["xai"]["segmentation"]["method"]),
        "envelope": str(cfg["xai"]["segmentation"]["envelope"]),
        "confidence_threshold": float(cfg["xai"]["segmentation"]["confidence_threshold"]),
        "per_segment_confidence": [float(s.get("confidence", 0)) for s in segmentations],
    })
    summary.add_artifact(segmentation_dir / "test_segmentation.json")

    summary.add_finding("segmenter_usable_fraction", round(quality["usable_fraction"], 4),
                        "Fraction of test windows the segmenter could label confidently. "
                        "Everything below the threshold is EXCLUDED from the alignment "
                        "analysis, and this number is the honest denominator.")
    summary.add_finding("segmenter_mean_confidence", round(quality["mean_confidence"], 4))
    summary.add_finding("mean_heart_rate_bpm", round(quality["mean_heart_rate_bpm"], 1))

    if quality["usable_fraction"] < 0.5:
        summary.add_warning(
            f"Only {100 * quality['usable_fraction']:.0f}% of test windows could be "
            "segmented confidently. The alignment analysis therefore describes a "
            "subset of the data - report this rate next to every alignment number."
        )

    # Physiological sanity: an implausible mean heart rate means the segmenter
    # is picking up something other than the cardiac cycle.
    mean_hr = quality["mean_heart_rate_bpm"]
    summary.add_assertion(
        "heart_rate_physiologically_plausible",
        bool(np.isfinite(mean_hr) and 50 <= mean_hr <= 120),
        f"mean detected heart rate = {mean_hr:.1f} bpm",
    )

    # ---- alignment for the CNN --------------------------------------------
    logger.info("Computing cardiac-cycle alignment for the CNN (Grad-CAM) ...")
    with watch.section("alignment_cnn"):
        alignment_cnn = batch_alignment(cams, segmentations, hop_length, n_frames,
                                        cfg, labels=labels, predictions=predictions)

    if alignment_cnn.get("n_included", 0) == 0:
        logger.error("No segment survived the confidence threshold.")
        summary.add_warning("Alignment analysis produced no usable segments.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    states = list(cfg["xai"]["alignment"]["states"])
    primary = str(cfg["xai"]["alignment"]["primary_state"])

    logger.info(f"  included {alignment_cnn['n_included']}/{alignment_cnn['n_total']} "
                f"segments ({100 * alignment_cnn['inclusion_rate']:.1f}%)")
    for state in states:
        mass = alignment_cnn[f"mean_mass_{state}"]
        time = alignment_cnn[f"mean_time_{state}"]
        enrichment = alignment_cnn[f"mean_enrichment_{state}"]
        logger.info(f"  {state:9s}: mass {mass:.4f}  time {time:.4f}  "
                    f"enrichment {enrichment:.3f}")

    # ---- significance ------------------------------------------------------
    logger.info(f"Testing '{primary}' enrichment against the shuffled null ...")
    with watch.section("significance"):
        significance = alignment_significance(alignment_cnn, cams, segmentations,
                                              hop_length, n_frames, cfg, seed)
    logger.info(f"  observed mass {significance.get('mean_observed_mass', 0):.4f} "
                f"vs null {significance.get('mean_null_mass', 0):.4f}")
    logger.info(f"  {significance.get('frac_significant_at_0.05', 0) * 100:.1f}% of "
                f"segments significant at p<0.05, "
                f"Cohen's d = {significance.get('effect_size_cohens_d', 0):.3f}")
    logger.info(f"  {significance.get('population_conclusion', '')}")

    # ---- stratified by outcome --------------------------------------------
    stratified = stratified_alignment(alignment_cnn, cfg)
    for category, values in stratified.items():
        if values.get("n", 0):
            logger.info(f"  {category}: n={values['n']}, "
                        f"E_{primary}={values.get(f'mean_enrichment_{primary}', float('nan')):.3f}")
    summary.add_table("alignment_by_outcome", [
        {"category": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                           for kk, vv in v.items()}}
        for k, v in stratified.items()
    ])

    # TP vs FP is the discriminating comparison.
    tp_enrichment = stratified.get("true_positive", {}).get(f"mean_enrichment_{primary}")
    fp_enrichment = stratified.get("false_positive", {}).get(f"mean_enrichment_{primary}")
    if tp_enrichment is not None and fp_enrichment is not None and \
            np.isfinite(tp_enrichment) and np.isfinite(fp_enrichment):
        logger.info(f"  TP vs FP enrichment: {tp_enrichment:.3f} vs {fp_enrichment:.3f}")
        summary.add_finding(
            "tp_minus_fp_enrichment", round(tp_enrichment - fp_enrichment, 4),
            "Positive values mean the model's systolic focus is stronger when it is "
            "right than when it is wrong - evidence that attention tracks real "
            "evidence rather than being a fixed habit of the architecture."
        )

    # ---- results and claim -------------------------------------------------
    alignment_results = {"cnn_logmel_gradcam": alignment_cnn}
    table = build_alignment_table(alignment_results, cfg)

    enrichment = alignment_cnn[f"mean_enrichment_{primary}"]
    fraction_significant = significance.get("frac_significant_at_0.05", 0.0)
    effect_size = significance.get("effect_size_cohens_d", 0.0)

    if enrichment > 1.15 and fraction_significant > 0.5 and effect_size > 0.5:
        verdict = "supported"
    elif enrichment > 1.05 and fraction_significant > 0.25:
        verdict = "weak"
    else:
        verdict = "contradicted"

    summary.add_claim_verdict(
        "C5",
        f"CNN attribution concentrates in {primary} beyond what uniform temporal "
        "attention would produce.",
        verdict,
        f"mean enrichment E_{primary} = {enrichment:.3f} (1.0 = no preference); "
        f"{100 * fraction_significant:.0f}% of segments significant against the "
        f"shuffled null; Cohen's d = {effect_size:.3f}",
        "A contradicted verdict here is publishable and honest: it would mean that "
        "although the CNN classifies well, its evidence is not localised to the "
        "phase where murmurs occur, and the model should not be described as "
        "clinically interpretable.",
    )

    save_json(xai_dir / "alignment.json", {
        "cnn_logmel_gradcam": {k: v for k, v in alignment_cnn.items()
                               if k != "per_segment"},
        "per_segment_count": len(alignment_cnn.get("per_segment", [])),
        "significance": significance,
        "stratified": stratified,
        "table": table,
        "segmenter_quality": quality,
        "primary_state": primary,
    })
    summary.add_artifact(xai_dir / "alignment.json")

    # ---- figures -----------------------------------------------------------
    plot_alignment(alignment_cnn, figures_dir / "fig_alignment_cnn", cfg, states,
                   title="CNN Grad-CAM: attribution mass vs. time budget")
    plot_alignment_comparison(table, figures_dir / "fig_alignment_comparison",
                              primary, cfg)
    summary.add_artifact(figures_dir / "fig_alignment_cnn.pdf")
    summary.add_artifact(figures_dir / "fig_alignment_comparison.pdf")

    # Per-category example panels - the paper's qualitative figure.
    extractor = LogMelExtractor(sr, cfg)
    band_frequencies = extractor.band_frequencies()
    n_examples = int(cfg["xai"]["gradcam"]["n_examples_per_category"])
    rng = np.random.default_rng(seed)

    categories = {
        "true_positive": (labels == 1) & (predictions == 1),
        "true_negative": (labels == 0) & (predictions == 0),
        "false_positive": (labels == 0) & (predictions == 1),
        "false_negative": (labels == 1) & (predictions == 0),
    }
    for category, mask in categories.items():
        # Only show examples the segmenter handled confidently, otherwise the
        # shaded cardiac states in the figure would be misleading.
        candidates = [i for i in np.where(mask)[0]
                      if segmentations[i].get("usable", False)]
        if not candidates:
            logger.warning(f"  no confidently segmented example for {category}")
            continue
        chosen = rng.choice(candidates, min(n_examples, len(candidates)), replace=False)
        for rank, i in enumerate(chosen):
            plot_gradcam_example(
                audio_test[i], logmel_test[i], cams[i], segmentations[i],
                sr, hop_length,
                supplementary_dir / f"fig_gradcam_{category}_{rank}", cfg,
                title=(f"{category.replace('_', ' ')} — "
                       f"E_{primary}=" +
                       (f"{alignment_cnn['per_segment'][0].get(f'enrichment_{primary}', float('nan')):.2f}"
                        if alignment_cnn.get("per_segment") else "n/a")),
                band_frequencies=band_frequencies,
            )
        # Promote the first true-positive example into the main paper figure.
        if category == "true_positive":
            plot_gradcam_example(
                audio_test[chosen[0]], logmel_test[chosen[0]], cams[chosen[0]],
                segmentations[chosen[0]], sr, hop_length,
                figures_dir / "fig_gradcam_example", cfg,
                title="Correctly detected abnormal recording",
                band_frequencies=band_frequencies,
            )
            summary.add_artifact(figures_dir / "fig_gradcam_example.pdf")

    # Segmenter example for the supplementary material - makes the substitution
    # of a simplified segmenter auditable rather than merely disclosed.
    usable_indices = [i for i, s in enumerate(segmentations) if s.get("usable")]
    if usable_indices:
        plot_segmentation_example(audio_test[usable_indices[0]],
                                  segmentations[usable_indices[0]], sr,
                                  supplementary_dir / "fig_segmentation_example", cfg)
        summary.add_artifact(supplementary_dir / "fig_segmentation_example.pdf")

    summary.add_table("alignment_summary", table)
    summary.add_finding(f"cnn_enrichment_{primary}", round(enrichment, 4))
    summary.add_finding("alignment_inclusion_rate",
                        round(alignment_cnn["inclusion_rate"], 4))
    summary.add_finding("frac_segments_significant", round(fraction_significant, 4))
    summary.add_finding("alignment_effect_size_d", round(effect_size, 4))

    summary.add_note(
        "The enrichment statistic is a ratio of attribution mass to time budget, so "
        "1.0 is the exact value produced by uniform temporal attention. This framing "
        "is what makes the number interpretable: 'X% of attention was in systole' is "
        "meaningless until compared against the fraction of time systole occupies."
    )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
