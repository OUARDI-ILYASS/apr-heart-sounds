"""Phase 02 - Extract MFCC, log-Mel and PWP features from every segment.

Three feature domains, one loop. Scalers are fitted on TRAIN ONLY and saved so
that phases 04-08 apply exactly the same transformation.

Artifacts:
    data/processed/<domain>/features_<split>.npy
    data/processed/<domain>/feature_names.json
    models/scalers/<domain>_scaler.joblib

Run:  python scripts/02_extract_features.py [--domains mfcc logmel pwp]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup, build_parser

from src.features.mfcc import MFCCExtractor
from src.features.logmel import LogMelExtractor
from src.features.pwp import PWPExtractor
from src.features.scaling import (
    fit_scaler, apply_scaler, save_scaler, scaler_report,
    fit_logmel_norm, apply_logmel_norm,
)
from src.utils.io import save_npy, save_json, load_npy, require_artifacts, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "02_extract_features"

EXTRACTORS = {"mfcc": MFCCExtractor, "logmel": LogMelExtractor, "pwp": PWPExtractor}


def main() -> int:
    parser = build_parser(__doc__.split("\n")[0])
    parser.add_argument("--domains", nargs="*", default=["mfcc", "logmel", "pwp"],
                        choices=list(EXTRACTORS), help="Feature domains to extract.")
    cfg, logger, args = setup(PHASE, "Extract MFCC / log-Mel / PWP features", parser)

    summary = PhaseSummary(PHASE, cfg, next_phase="03_cluster_features")
    watch = Stopwatch()

    interim = Path(cfg["_abs_paths"]["interim_dir"])
    processed = ensure_dir(cfg["_abs_paths"]["processed_dir"])
    scalers_dir = ensure_dir(Path(cfg["_abs_paths"]["models_dir"]) / "scalers")

    require_artifacts(
        [interim / "segment_index.csv"] +
        [interim / f"segments_{s}.npy" for s in ["train", "val", "test"]],
        phase=PHASE,
    )

    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    summary.add_input(interim / "segment_index.csv", n_segments=len(index))
    summary.touch_split("train", "val", "test")

    sr = float(cfg["preprocessing"]["target_sr"])
    segments = {}
    for split in ["train", "val", "test"]:
        segments[split] = load_npy(interim / f"segments_{split}.npy")
        logger.info(f"{split}: {segments[split].shape}")

    for domain in args.domains:
        logger.info(f"--- {domain} " + "-" * 50)
        extractor = EXTRACTORS[domain](sr, cfg)
        description = extractor.describe()
        logger.info(f"  output shape per segment: {description['output_shape']} "
                    f"({description['n_features']} values)")

        domain_dir = ensure_dir(processed / domain)
        features = {}

        for split in ["train", "val", "test"]:
            with watch.section(f"{domain}_{split}"):
                features[split] = extractor.transform_batch(segments[split], progress=True)
            logger.info(f"  {split}: {features[split].shape}")

        # ---- scaling: fitted on train only ------------------------------
        if domain == "logmel":
            # Per-band statistics; the CNN input keeps its 2-D structure.
            stats = fit_logmel_norm(features["train"])
            for split in features:
                features[split] = apply_logmel_norm(features[split], stats)
            save_json(scalers_dir / "logmel_norm.json",
                      {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()})
            summary.add_artifact(scalers_dir / "logmel_norm.json")
            summary.add_finding(f"{domain}_n_degenerate_bands",
                                int(np.sum(stats["std"] < 1e-6)),
                                "Mel bands with (near) zero training variance")
        else:
            scaler = fit_scaler(features["train"],
                                method=str(cfg["features"]["scaling"]["method"]))
            for split in features:
                features[split] = apply_scaler(scaler, features[split])
            scaler_path = scalers_dir / f"{domain}_scaler.joblib"
            save_scaler(scaler_path, scaler,
                        {"fitted_on": "train", "n_train_samples": int(len(features["train"])),
                         "method": str(cfg["features"]["scaling"]["method"])})
            summary.add_artifact(scaler_path)

            report = scaler_report(scaler, extractor.feature_names)
            if report.get("n_near_constant", 0):
                summary.add_warning(
                    f"{domain}: {report['n_near_constant']} features have near-zero "
                    f"training variance ({report['near_constant_features'][:5]}...). "
                    "These carry no information and inflate the dimensionality."
                )
            summary.add_finding(f"{domain}_scaler_report", report)

        # ---- persist ----------------------------------------------------
        for split, array in features.items():
            path = save_npy(domain_dir / f"features_{split}.npy", array)
            summary.add_artifact(path, array=array)

        save_json(domain_dir / "feature_names.json", extractor.feature_names)
        save_json(domain_dir / "extractor_config.json", description)
        summary.add_artifact(domain_dir / "feature_names.json",
                             n_names=len(extractor.feature_names))

        # Domain-specific diagnostics worth having in the summary.
        summary.add_finding(f"{domain}_dim", int(description["n_features"]))
        summary.add_parameters({f"{domain}_config": description})

        if domain == "pwp":
            logger.info(f"  perceptual bands: {description['band_edges_hz']}")
            logger.info(f"  nodes per band  : {description['nodes_per_band']}")
            summary.add_assertion(
                "pwp_all_bands_populated",
                all(n > 0 for n in description["nodes_per_band"]),
                f"nodes per band: {description['nodes_per_band']}",
            )
        if domain == "mfcc":
            summary.add_finding("mfcc_window_ms", description["window_ms"])
        if domain == "logmel":
            summary.add_finding("logmel_shape", description["output_shape"],
                                "CNN input (n_mels, n_frames)")
            summary.add_finding("logmel_freq_resolution_hz",
                                description["freq_resolution_hz"])

        # Sanity: constant feature matrices mean the extractor is broken.
        train_std = float(np.std(features["train"]))
        summary.add_assertion(
            f"{domain}_features_non_degenerate", train_std > 1e-6,
            f"train feature std = {train_std:.6f}",
        )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
