"""Segment-level prediction and aggregation to recording level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def aggregate_to_recording(recording_ids: np.ndarray, y_prob: np.ndarray,
                           y_true_segment: Optional[np.ndarray] = None,
                           method: str = "mean_probability",
                           threshold: float = 0.5
                           ) -> Dict[str, np.ndarray]:
    """Collapse segment predictions into one prediction per recording.

    Returns arrays aligned on the sorted unique recording IDs so that every
    downstream comparison (McNemar, bootstrap) indexes the same recordings in
    the same order.
    """
    recording_ids = np.asarray(recording_ids)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    unique_ids = np.unique(recording_ids)
    agg_prob = np.zeros(len(unique_ids), dtype=np.float64)
    agg_pred = np.zeros(len(unique_ids), dtype=np.int64)
    n_segments = np.zeros(len(unique_ids), dtype=np.int64)
    agg_true = np.full(len(unique_ids), -1, dtype=np.int64)

    for i, rec_id in enumerate(unique_ids):
        mask = recording_ids == rec_id
        probs = y_prob[mask]
        n_segments[i] = int(mask.sum())

        if method == "mean_probability":
            agg_prob[i] = float(probs.mean())
            agg_pred[i] = int(agg_prob[i] >= threshold)
        elif method == "majority_vote":
            votes = (probs >= threshold).astype(int)
            agg_prob[i] = float(votes.mean())
            # Ties -> abnormal: false negatives are the costly error in screening.
            agg_pred[i] = int(agg_prob[i] >= 0.5)
        elif method == "max_probability":
            # Any single confident segment flags the recording. Very sensitive,
            # very trigger-happy; reported only as a sensitivity analysis.
            agg_prob[i] = float(probs.max())
            agg_pred[i] = int(agg_prob[i] >= threshold)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        if y_true_segment is not None:
            labels = np.asarray(y_true_segment)[mask]
            # All segments of a recording share its label by construction; if
            # they ever disagree, the segment index is corrupt and we want to
            # know immediately rather than average the inconsistency away.
            if len(np.unique(labels)) > 1:
                raise ValueError(
                    f"Recording {rec_id} has inconsistent segment labels "
                    f"{np.unique(labels)} - the segment index is corrupt."
                )
            agg_true[i] = int(labels[0])

    out = {
        "recording_ids": unique_ids,
        "y_prob": agg_prob,
        "y_pred": agg_pred,
        "n_segments": n_segments,
    }
    if y_true_segment is not None:
        out["y_true"] = agg_true
    return out


def predict_classical(model, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Segment-level probabilities and hard predictions from an sklearn model."""
    from .classical import predict_proba_safe

    y_prob = predict_proba_safe(model, X)
    return {"y_prob": y_prob, "y_pred": (y_prob >= 0.5).astype(int)}


def predict_cnn(model, X: np.ndarray, device, batch_size: int = 128
                ) -> Dict[str, np.ndarray]:
    """Segment-level probabilities from the CNN, in eval mode without grad."""
    import torch

    model.eval()
    model = model.to(device)
    probs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                np.asarray(X[start:start + batch_size], dtype=np.float32)
            ).to(device)
            logits = model(batch)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    y_prob = np.concatenate(probs)
    return {"y_prob": y_prob, "y_pred": (y_prob >= 0.5).astype(int)}


def segment_agreement(recording_ids: np.ndarray, y_pred: np.ndarray
                      ) -> Dict[str, float]:
    """How consistently do segments within a recording agree?

    Reported in the paper as a stability diagnostic. Low agreement means the
    recording-level decision hinges on the aggregation rule, which is worth
    knowing before claiming a recording-level number.
    """
    recording_ids = np.asarray(recording_ids)
    y_pred = np.asarray(y_pred)
    agreements = []
    for rec_id in np.unique(recording_ids):
        preds = y_pred[recording_ids == rec_id]
        if len(preds) > 1:
            agreements.append(float(max(np.mean(preds == 0), np.mean(preds == 1))))
    if not agreements:
        return {"mean_agreement": 1.0, "n_multi_segment_recordings": 0}
    return {
        "mean_agreement": float(np.mean(agreements)),
        "median_agreement": float(np.median(agreements)),
        "frac_unanimous": float(np.mean(np.array(agreements) == 1.0)),
        "n_multi_segment_recordings": len(agreements),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        metric: str = "macc") -> Tuple[float, float]:
    """Choose a decision threshold on VALIDATION data.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold, best_score = 0.5, -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        if metric == "macc":
            score = 0.5 * (sensitivity + specificity)
        elif metric == "youden":
            score = sensitivity + specificity - 1.0
        elif metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            score = (2 * precision * sensitivity / (precision + sensitivity)
                     if (precision + sensitivity) else 0.0)
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")

        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)

    return best_threshold, best_score
