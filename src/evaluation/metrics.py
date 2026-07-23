"""Figures of merit, with the clinical asymmetry made explicit.

The metric hierarchy for this task, in order of what actually matters:

1. **MAcc = (Se + Sp) / 2** - the official PhysioNet/CinC 2016 score, and the
   primary metric here. It is exactly ``balanced_accuracy`` in sklearn.
2. **Sensitivity (recall on Abnormal)** - the fraction of pathological
   recordings we catch. In screening this is the expensive one to get wrong.
3. **Specificity** - the fraction of normals correctly cleared. Low specificity
   means unnecessary referrals.
4. **ROC-AUC** - threshold-free ranking quality; useful because it separates
   "the model ranks well but the threshold is wrong" from "the model cannot
   rank".
5. **Accuracy** - reported for completeness and because the course asks for it,
   but flagged wherever it is misleading.

"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """TP/TN/FP/FN with Abnormal (class 1) as the positive class."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """All figures of merit for one set of predictions."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        cohen_kappa_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    counts = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),       # == recall for the positive class
        "specificity": float(specificity),
        "macc": float(0.5 * (sensitivity + specificity)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # MCC is the most honest single number under class imbalance: it uses
        # all four confusion-matrix cells and is 0 for any constant predictor.
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        **{k: float(v) for k, v in counts.items()},
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
    }

    # Negative/positive predictive value - what a clinician actually reads off
    # a report: "given this result, how likely is disease?"
    metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) else 0.0

    if y_prob is not None and len(np.unique(y_true)) > 1:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        # Average precision is more informative than AUC under imbalance
        # because it ignores the (huge) true-negative cell.
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


def baseline_metrics(y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Trivial baselines every reported model must beat.

    Including these in the results table is what stops a reader from having to
    do arithmetic to work out whether a 78% accuracy is impressive. It is not:
    it is the majority-class rate.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    rng = np.random.default_rng(0)

    return {
        "always_normal": compute_metrics(y_true, np.zeros(n, dtype=int)),
        "always_abnormal": compute_metrics(y_true, np.ones(n, dtype=int)),
        "random_stratified": compute_metrics(
            y_true, rng.binomial(1, float(np.mean(y_true)), n)
        ),
    }


def metrics_table(results: Dict[str, Dict[str, float]],
                  metric_names: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Reshape ``{model: metrics}`` into table rows sorted by MAcc."""
    metric_names = metric_names or [
        "accuracy", "sensitivity", "specificity", "macc",
        "precision", "f1", "roc_auc", "mcc",
    ]
    rows = []
    for model_name, metrics in results.items():
        row: Dict[str, object] = {"model": model_name}
        for name in metric_names:
            value = metrics.get(name, float("nan"))
            row[name] = round(float(value), 4) if isinstance(value, (int, float)) else value
        rows.append(row)
    return sorted(rows, key=lambda r: -(r.get("macc") or 0))


def per_group_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      groups: np.ndarray,
                      y_prob: Optional[np.ndarray] = None
                      ) -> List[Dict[str, object]]:
    """Metrics broken down by sub-database.

    This is the shortcut-learning check. If the model's MAcc is uniform across
    the six sites, it is detecting pathology. If it is excellent on the
    abnormal-heavy site and near chance elsewhere, it has learned site
    prevalence. Reporting only the pooled number would hide that completely.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    rows = []
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 5 or len(np.unique(y_true[mask])) < 2:
            # Too small or single-class - metrics would be undefined/unstable.
            rows.append({"group": str(group), "n": int(mask.sum()),
                         "note": "insufficient data or single class"})
            continue
        metrics = compute_metrics(
            y_true[mask], y_pred[mask],
            y_prob[mask] if y_prob is not None else None,
        )
        rows.append({
            "group": str(group), "n": int(mask.sum()),
            "prevalence": round(metrics["prevalence"], 3),
            "macc": round(metrics["macc"], 4),
            "sensitivity": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        })
    return rows


def compare_to_literature(our_metrics: Dict[str, float],
                          literature: Dict[str, Dict[str, float]]
                          ) -> List[Dict[str, object]]:
    """Rows placing our numbers next to published CinC 2016 entries.

    The ``comparable`` column is deliberately False for every literature row:
    those scores come from the official hidden test set, which was never
    released, so ours cannot be a like-for-like comparison. Making that a
    column rather than a footnote means the caveat travels with the table.
    """
    rows: List[Dict[str, object]] = [{
        "system": "This work (best)",
        "macc": round(float(our_metrics.get("macc", float("nan"))), 4),
        "sensitivity": round(float(our_metrics.get("sensitivity", float("nan"))), 4),
        "specificity": round(float(our_metrics.get("specificity", float("nan"))), 4),
        "test_set": "our held-out split of the public training data",
        "comparable": True,
    }]
    for name, metrics in literature.items():
        rows.append({
            "system": name,
            "macc": metrics.get("macc"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "test_set": "official CinC 2016 hidden test set",
            "comparable": False,
        })
    return rows
