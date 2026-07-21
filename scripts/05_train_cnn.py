#!/usr/bin/env python3
"""Phase 05 - Train the 2-D CNN on log-Mel spectrograms (PyTorch, local GPU).

Early stopping and checkpoint selection are driven by validation MAcc. The test
split is not opened in this phase.

Artifacts:
    models/cnn/cnn_logmel.pt      best-epoch weights + architecture spec
    results/evaluation/cnn_training.json
    figures/fig_training_curves.pdf

Run:  python scripts/05_train_cnn.py [--epochs N] [--device cuda]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _bootstrap import setup, build_parser

from src.models.cnn import build_cnn, resolve_device
from src.models.trainer import (
    make_dataloaders, compute_class_weights, train_model, diagnose_training,
)
from src.models.inference import predict_cnn, aggregate_to_recording, find_best_threshold
from src.evaluation.metrics import compute_metrics
from src.visualization.curves import plot_training_curves
from src.utils.io import load_npy, save_json, require_artifacts, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "05_train_cnn"


def main() -> int:
    parser = build_parser(__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--device", default=None, help="cuda | cuda:0 | cpu")
    cfg, logger, args = setup(PHASE, "Train the log-Mel CNN", parser)

    if args.epochs is not None:
        cfg["cnn"]["training"]["epochs"] = int(args.epochs)
    if args.device is not None:
        cfg["cnn"]["training"]["device"] = args.device

    summary = PhaseSummary(PHASE, cfg, next_phase="06_evaluate_models")
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    models_dir = ensure_dir(Path(cfg["_abs_paths"]["models_dir"]) / "cnn")
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "evaluation")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])

    require_artifacts([
        processed / "logmel" / "features_train.npy",
        processed / "logmel" / "features_val.npy",
        interim / "segment_index.csv",
    ], phase=PHASE)

    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})
    train_index = index[index["split"] == "train"].reset_index(drop=True)
    val_index = index[index["split"] == "val"].reset_index(drop=True)
    summary.touch_split("train", "val")

    X_train = load_npy(processed / "logmel" / "features_train.npy")
    X_val = load_npy(processed / "logmel" / "features_val.npy")
    y_train = train_index["label"].to_numpy()
    y_val = val_index["label"].to_numpy()
    recordings_val = val_index["recording_id"].to_numpy()

    logger.info(f"Train: {X_train.shape}  ({100 * y_train.mean():.1f}% abnormal)")
    logger.info(f"Val  : {X_val.shape}  ({100 * y_val.mean():.1f}% abnormal)")

    device = resolve_device(str(cfg["cnn"]["training"]["device"]))
    logger.info(f"Device: {device}")
    if device.type == "cpu" and str(cfg["cnn"]["training"]["device"]).startswith("cuda"):
        summary.add_warning(
            "CUDA was requested but is unavailable; training fell back to CPU. "
            "Expect a large slowdown."
        )

    seed = int(cfg["experiment"]["seed"])
    n_mels, n_frames = X_train.shape[1], X_train.shape[2]
    model = build_cnn(cfg, n_mels=n_mels, n_frames=n_frames)
    spec = model.architecture_spec()
    logger.info(f"Model: {spec['n_parameters']['total']:,} parameters "
                f"({spec['n_parameters']['conv']:,} conv / "
                f"{spec['n_parameters']['head']:,} head)")
    logger.info(f"Last conv feature map: {spec['feature_map_shape']} "
                "(this is the Grad-CAM resolution)")

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, cfg, seed)

    class_weights = None
    if bool(cfg["cnn"]["training"]["class_weighting"]):
        class_weights = compute_class_weights(y_train)
        logger.info(f"Class weights: {class_weights.tolist()}")

    checkpoint_path = models_dir / "cnn_logmel.pt"
    with watch.section("training"):
        result = train_model(model, train_loader, val_loader, cfg, device,
                             class_weights=class_weights,
                             checkpoint_path=checkpoint_path, logger=logger)

    logger.info(f"Best epoch {result['best_epoch']} — "
                f"val MAcc {result['best_val_metrics'].get('macc', 0):.4f}")

    # ---- validation, recording level ---------------------------------------
    predictions = predict_cnn(model, X_val, device)
    aggregated = aggregate_to_recording(
        recordings_val, predictions["y_prob"], y_val,
        method=str(cfg["evaluation"]["primary_aggregation"]),
    )
    recording_metrics = compute_metrics(aggregated["y_true"], aggregated["y_pred"],
                                        aggregated["y_prob"])
    threshold, threshold_score = find_best_threshold(
        aggregated["y_true"], aggregated["y_prob"], metric="macc"
    )
    logger.info(f"val MAcc (recording) : {recording_metrics['macc']:.4f}")
    logger.info(f"tuned threshold      : {threshold:.2f} -> val MAcc {threshold_score:.4f}")

    for message in diagnose_training(result["history"]):
        logger.warning(message)
        summary.add_warning(message)

    payload = {
        **{k: v for k, v in result.items() if k != "history"},
        "history": result["history"],
        "architecture": spec,
        "val_recording": recording_metrics,
        "tuned_threshold": threshold,
        "tuned_threshold_val_macc": threshold_score,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }
    save_json(results_dir / "cnn_training.json", payload)
    summary.add_artifact(checkpoint_path)
    summary.add_artifact(results_dir / "cnn_training.json")

    plot_training_curves(result["history"], figures_dir / "fig_training_curves",
                         cfg, best_epoch=result["best_epoch"])
    summary.add_artifact(figures_dir / "fig_training_curves.pdf")

    summary.add_parameters({
        "epochs_configured": int(cfg["cnn"]["training"]["epochs"]),
        "epochs_run": result["epochs_run"],
        "batch_size": int(cfg["cnn"]["training"]["batch_size"]),
        "learning_rate": float(cfg["cnn"]["training"]["learning_rate"]),
        "augmentation": cfg["cnn"]["augmentation"].to_dict(),
        "device": str(device),
        "amp": result["used_amp"],
    })
    summary.add_finding("n_parameters", spec["n_parameters"]["total"])
    summary.add_finding("best_epoch", result["best_epoch"])
    summary.add_finding("early_stopped", result["early_stopped"])
    summary.add_finding("val_macc_segment",
                        round(result["best_val_metrics"].get("macc", 0), 4))
    summary.add_finding("val_macc_recording", round(recording_metrics["macc"], 4))
    summary.add_finding("training_minutes", round(result["training_seconds"] / 60, 1))
    summary.add_finding("gradcam_feature_map_shape", spec["feature_map_shape"],
                        "Grad-CAM spatial resolution before upsampling")

    summary.add_assertion(
        "cnn_beats_chance", recording_metrics["macc"] > 0.55,
        f"val recording MAcc = {recording_metrics['macc']:.4f}",
    )
    final = result["history"][-1]
    summary.add_assertion(
        "train_val_gap_acceptable",
        (final.get("train_macc", 0) - final.get("val_macc", 0)) < 0.20,
        f"final gap = {final.get('train_macc', 0) - final.get('val_macc', 0):+.4f}",
    )
    summary.add_note(
        "Weights are restored from the best validation epoch, not the last. "
        "The test split has not been read in this phase."
    )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
