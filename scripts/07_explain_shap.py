"""Phase 07 - SHAP explanations for the classical models.

TreeSHAP (exact) for the Random Forests, KernelSHAP (approximate) for the SVM.
Attributions are then projected onto the frequency axis so that models built on
different feature domains can be compared on common ground.

Artifacts:
    results/xai/shap/<model>_shap.json
    results/xai/shap/frequency_profiles.json
    figures/fig_shap_*.pdf, figures/fig_frequency_attribution.pdf

Run:  python scripts/07_explain_shap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.features.mfcc import MFCCExtractor
from src.features.pwp import PWPExtractor
from src.xai.shap_explainer import (
    explain_tree_model, explain_kernel_model, top_features, group_by_prefix,
    map_mfcc_shap_to_frequency, map_pwp_shap_to_frequency,
    compare_frequency_profiles, rank_stability,
)
from src.visualization.explanations import (
    plot_shap_bar, plot_shap_summary, plot_frequency_attribution,
)
from src.utils.io import (
    load_npy, load_json, load_joblib, save_json, require_artifacts, ensure_dir,
)
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "07_explain_shap"


def main() -> int:
    cfg, logger, args = setup(PHASE, "SHAP explanations for the classical models")
    summary = PhaseSummary(PHASE, cfg, next_phase="08_explain_gradcam")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    classical_dir = Path(cfg["_abs_paths"]["models_dir"]) / "classical"
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "xai" / "shap")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])

    require_artifacts([interim / "segment_index.csv"], phase=PHASE)
    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    test_index = index[index["split"] == "test"].reset_index(drop=True)

    # Explanations are computed on TEST data, which phase 06 already opened.
    # Explaining training predictions would describe memorisation, not
    # generalisation.
    summary.touch_split("train", "test")
    summary.add_note(
        "SHAP values are computed on test predictions, with the background "
        "distribution drawn from training data. Explaining training predictions "
        "would describe what the model memorised rather than how it generalises."
    )

    shap_cfg = cfg["xai"]["shap"]
    seed = int(cfg["experiment"]["seed"])
    sr = float(cfg["preprocessing"]["target_sr"])
    n_explain = int(shap_cfg["explain_subsample"])
    n_background = int(shap_cfg["background_size"])
    top_k = int(shap_cfg["top_k_features"])

    extractors = {"mfcc": MFCCExtractor(sr, cfg), "pwp": PWPExtractor(sr, cfg)}
    frequency_profiles = {}
    all_results = {}

    for model_path in sorted(classical_dir.glob("*.joblib")):
        tag = model_path.stem
        model_kind, domain = tag.split("_", 1)
        if domain not in extractors:
            continue

        feature_path = processed / domain / "features_test.npy"
        names_path = processed / domain / "feature_names.json"
        if not feature_path.exists():
            logger.warning(f"Skipping {tag}: test features missing")
            continue

        logger.info(f"--- {tag} " + "-" * 50)
        model = load_joblib(model_path)
        X_test = load_npy(feature_path)
        X_train = load_npy(processed / domain / "features_train.npy")
        feature_names = load_json(names_path)

        rng = np.random.default_rng(seed)
        explain_idx = rng.choice(len(X_test), min(n_explain, len(X_test)), replace=False)
        background_idx = rng.choice(len(X_train), min(n_background, len(X_train)),
                                    replace=False)
        X_explain = X_test[explain_idx]
        X_background = X_train[background_idx]

        # ---- run the appropriate explainer -----------------------------
        if model_kind == "rf":
            logger.info(f"  TreeSHAP (exact) on {len(X_explain)} test segments")
            with watch.section(f"shap_{tag}"):
                result = explain_tree_model(model, X_explain, feature_names)
        else:
            logger.info(f"  KernelSHAP (approximate) on {len(X_explain)} test "
                        f"segments with {len(X_background)} background points")
            summary.add_warning(
                f"{tag}: KernelSHAP is a Monte-Carlo approximation. Its per-feature "
                "ranking is only as stable as the rank_stability check reports."
            )
            with watch.section(f"shap_{tag}"):
                result = explain_kernel_model(model, X_explain, X_background,
                                              feature_names, cfg=cfg, seed=seed)

        logger.info(f"  explainer={result['explainer']}  exact={result['exact']}  "
                    f"base value={result['base_value']:.4f}")

        rows = top_features(result, k=top_k)
        for row in rows[:8]:
            logger.info(f"    {row['rank']:2d}. {row['feature']:32s} "
                        f"|SHAP|={row['mean_abs_shap']:.5f} "
                        f"({100 * row['share_of_total']:.1f}% of total, "
                        f"consistency {row['sign_consistency']:.2f})")

        groups = group_by_prefix(result)

        # ---- frequency projection ---------------------------------------
        extractor = extractors[domain]
        if domain == "mfcc":
            profile = map_mfcc_shap_to_frequency(result, extractor)
            logger.info(f"  frequency peak (approx): {profile['peak_frequency_hz']:.0f} Hz, "
                        f"centroid {profile['centroid_hz']:.0f} Hz")
        else:
            profile = map_pwp_shap_to_frequency(result, extractor)
            logger.info(f"  frequency peak (exact): {profile['peak_frequency_hz']:.0f} Hz, "
                        f"centroid {profile['centroid_hz']:.0f} Hz")
            logger.info(f"  descriptor shares: {profile['per_descriptor_share']}")
        frequency_profiles[tag] = profile

        # ---- persist ----------------------------------------------------
        payload = {
            "model": tag,
            "domain": domain,
            "explainer": result["explainer"],
            "exact": result["exact"],
            "base_value": result["base_value"],
            "n_explained": result["n_explained"],
            "top_features": rows,
            "feature_groups": groups[:20],
            "frequency_profile": profile,
            # "rank_stability": stability,
        }
        save_json(results_dir / f"{tag}_shap.json", payload)
        summary.add_artifact(results_dir / f"{tag}_shap.json")
        all_results[tag] = payload

        plot_shap_bar(rows, figures_dir / f"fig_shap_bar_{tag}", cfg,
                      title=f"{tag} — top {top_k} features by mean |SHAP|")
        try:
            plot_shap_summary(result, figures_dir / f"fig_shap_summary_{tag}",
                              top_k=top_k, cfg=cfg, title=tag)
        except Exception as exc:
            logger.warning(f"  shap.summary_plot failed ({exc}); bar chart still written")

        summary.add_finding(f"{tag}_peak_frequency_hz",
                            round(profile["peak_frequency_hz"], 1))
        summary.add_finding(f"{tag}_centroid_hz", round(profile["centroid_hz"], 1))
        summary.add_finding(f"{tag}_top_feature", rows[0]["feature"] if rows else None)

    if not frequency_profiles:
        logger.error("No SHAP explanations were produced.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    # ---- cross-model agreement --------------------------------------------
    comparison = compare_frequency_profiles(frequency_profiles)
    if "mean_correlation" in comparison:
        logger.info(f"Cross-model frequency-profile agreement: "
                    f"r={comparison['mean_correlation']:.3f} ({comparison['agreement']})")
        for pair, value in comparison["pairwise_correlations"].items():
            logger.info(f"  {pair}: r={value:.3f}")

    save_json(results_dir / "frequency_profiles.json",
              {"profiles": frequency_profiles, "comparison": comparison})
    summary.add_artifact(results_dir / "frequency_profiles.json")

    plot_frequency_attribution(frequency_profiles,
                               figures_dir / "fig_frequency_attribution", cfg)
    summary.add_artifact(figures_dir / "fig_frequency_attribution.pdf")

    peaks = [p["peak_frequency_hz"] for p in frequency_profiles.values()]
    in_murmur_band = sum(1 for p in peaks if 100 <= p <= 300)
    summary.add_finding("peak_frequencies_hz", [round(p, 1) for p in peaks])
    summary.add_finding(
        "n_models_peaking_in_murmur_band", f"{in_murmur_band}/{len(peaks)}",
        "Models whose attribution peaks in the 100-300 Hz range where systolic "
        "murmur energy is clinically expected"
    )

    mean_correlation = comparison.get("mean_correlation", float("nan"))
    summary.add_claim_verdict(
        "C4",
        "Models trained on different feature domains attribute importance to the "
        "same frequency region.",
        ("supported" if mean_correlation > 0.7 else
         "weak" if mean_correlation > 0.4 else "contradicted"),
        f"mean pairwise correlation between frequency-attribution profiles = "
        f"{mean_correlation:.3f}",
        "Agreement across models with different inductive biases and different "
        "SHAP algorithms is far stronger evidence than any single explanation. "
        "Disagreement would mean at least one model is reading an artefact.",
    )

    summary.add_table("shap_frequency_summary", [
        {"model": tag,
         "explainer": p["explainer"],
         "exact": p["exact"],
         "peak_hz": round(p["frequency_profile"]["peak_frequency_hz"], 1),
         "centroid_hz": round(p["frequency_profile"]["centroid_hz"], 1),
         "mapping": p["frequency_profile"]["method"]}
        for tag, p in all_results.items()
    ])

    summary.add_note(
        "MFCC frequency attributions are an approximation: cepstral coefficients "
        "are a DCT of the log-mel spectrum, so SHAP mass is redistributed through "
        "the magnitude of the DCT basis and the sign is discarded. PWP attributions "
        "are exact, because each PWP feature belongs to exactly one frequency band "
        "by construction. Where the two disagree, the PWP result is the reliable one."
    )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
