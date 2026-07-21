"""Confusion-matrix construction and error analysis."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = 2, normalize: Optional[str] = None
                     ) -> np.ndarray:
    """Confusion matrix with rows = true, columns = predicted.

    ``normalize='true'`` gives per-class recall on the diagonal, which is the
    version to show under class imbalance: a raw-count matrix makes the large
    Normal class visually dominate and hides how badly the minority class is
    handled.
    """
    from sklearn.metrics import confusion_matrix as sk_confusion

    matrix = sk_confusion(np.asarray(y_true).astype(int),
                          np.asarray(y_pred).astype(int),
                          labels=list(range(n_classes)))
    if normalize is None:
        return matrix.astype(int)

    matrix = matrix.astype(np.float64)
    if normalize == "true":
        denominator = matrix.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denominator = matrix.sum(axis=0, keepdims=True)
    elif normalize == "all":
        denominator = matrix.sum()
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")
    return np.divide(matrix, denominator, out=np.zeros_like(matrix),
                     where=denominator != 0)


def error_analysis(recording_ids: np.ndarray, y_true: np.ndarray,
                   y_pred: np.ndarray, y_prob: np.ndarray,
                   metadata: Optional[Dict[str, np.ndarray]] = None,
                   top_k: int = 20) -> Dict[str, object]:
    """Which recordings does the model get wrong, and are they systematic?

    Sorted by confidence, so the most instructive cases come first: a false
    negative at p=0.02 is a very different failure from one at p=0.48. The
    former means the model is confidently blind to that pathology; the latter
    is a borderline call that better thresholding might fix.
    """
    recording_ids = np.asarray(recording_ids)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    def _rows(mask: np.ndarray, ascending: bool) -> List[Dict[str, object]]:
        indices = np.where(mask)[0]
        if indices.size == 0:
            return []
        order = indices[np.argsort(y_prob[indices])]
        if not ascending:
            order = order[::-1]
        rows = []
        for index in order[:top_k]:
            row: Dict[str, object] = {
                "recording_id": str(recording_ids[index]),
                "true": int(y_true[index]),
                "pred": int(y_pred[index]),
                "prob_abnormal": round(float(y_prob[index]), 4),
            }
            if metadata:
                for key, values in metadata.items():
                    row[key] = values[index]
            rows.append(row)
        return rows

    return {
        "n_false_negatives": int(fn_mask.sum()),
        "n_false_positives": int(fp_mask.sum()),
        # Most confident misses first - the model was sure and wrong.
        "worst_false_negatives": _rows(fn_mask, ascending=True),
        "worst_false_positives": _rows(fp_mask, ascending=False),
        "mean_prob_correct": float(y_prob[y_true == y_pred].mean())
                             if np.any(y_true == y_pred) else float("nan"),
        "mean_prob_incorrect": float(y_prob[y_true != y_pred].mean())
                               if np.any(y_true != y_pred) else float("nan"),
    }


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> Dict[str, List[float]]:
    """Reliability diagram data plus the Expected Calibration Error.

    PROFESSOR Q: "Are your probabilities meaningful?"
    A: We measure it rather than assume it. ECE is the average gap between
       predicted confidence and observed accuracy. Random Forest probabilities
       are notoriously under-confident (they are vote fractions), and a CNN
       trained with a weighted loss is systematically shifted. If ECE is large
       we say so, because a poorly calibrated probability makes the
       mean-probability aggregation rule shakier than the majority vote.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    centres, accuracies, confidences, counts = [], [], [], []
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1] if i < n_bins - 1
                                       else y_prob <= edges[i + 1])
        n = int(mask.sum())
        counts.append(n)
        centres.append(float(0.5 * (edges[i] + edges[i + 1])))
        if n:
            accuracy = float(y_true[mask].mean())
            confidence = float(y_prob[mask].mean())
            accuracies.append(accuracy)
            confidences.append(confidence)
            ece += (n / len(y_true)) * abs(accuracy - confidence)
        else:
            accuracies.append(float("nan"))
            confidences.append(float("nan"))

    return {
        "bin_centers": centres,
        "accuracy": accuracies,
        "confidence": confidences,
        "counts": counts,
        "ece": float(ece),
    }
