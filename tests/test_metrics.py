"""Metric correctness, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    confusion_counts, compute_metrics, baseline_metrics, per_group_metrics,
)
from src.evaluation.statistical import (
    mcnemar_test, bootstrap_ci, holm_bonferroni, effect_size_cohens_h,
)
from src.evaluation.confusion import confusion_matrix, calibration_curve


def test_confusion_counts_by_hand():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0])
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tp": 2, "fn": 1, "fp": 1, "tn": 3}


def test_macc_is_balanced_accuracy():
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["macc"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 0, 1, 1, 0])
    metrics = compute_metrics(y, y, y.astype(float))
    assert metrics["accuracy"] == 1.0
    assert metrics["macc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_always_normal_baseline_exposes_accuracy_trap():
    """The central motivation for using MAcc: 80% accuracy at 0.5 MAcc."""
    y_true = np.array([0] * 80 + [1] * 20)
    baselines = baseline_metrics(y_true)
    always_normal = baselines["always_normal"]
    assert always_normal["accuracy"] == pytest.approx(0.80)
    assert always_normal["macc"] == pytest.approx(0.50)
    assert always_normal["sensitivity"] == 0.0
    # MCC is 0 for any constant predictor - the honest single number.
    assert always_normal["mcc"] == pytest.approx(0.0, abs=1e-9)


def test_metrics_survive_single_class_input():
    y_true = np.zeros(10, dtype=int)
    metrics = compute_metrics(y_true, np.zeros(10, dtype=int))
    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == 1.0


def test_mcnemar_identical_models_is_degenerate():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result["p_value"] == 1.0
    assert result["n_discordant"] == 0


def test_mcnemar_detects_a_clear_difference():
    y_true = np.array([1] * 50 + [0] * 50)
    good = y_true.copy()
    bad = np.zeros(100, dtype=int)
    result = mcnemar_test(y_true, good, bad)
    assert result["p_value"] < 0.001
    assert result["better_model"] == "A"


def test_mcnemar_uses_exact_test_when_discordant_count_is_small():
    y_true = np.array([1] * 10 + [0] * 10)
    a = y_true.copy()
    b = y_true.copy()
    b[0] = 0                       # one discordant case
    result = mcnemar_test(y_true, a, b)
    assert result["test"] == "exact_binomial"


def test_bootstrap_ci_contains_point_estimate():
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_pred = np.where(rng.random(300) < 0.8, y_true, 1 - y_true)
    ci = bootstrap_ci(y_true, y_pred, balanced_accuracy_score, n_resamples=200, seed=0)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert ci["n_valid_resamples"] > 150


def test_holm_bonferroni_is_stricter_than_uncorrected():
    p_values = {"a": 0.01, "b": 0.03, "c": 0.04, "d": 0.20}
    corrected = holm_bonferroni(p_values, alpha=0.05)
    # 0.01 < 0.05/4 = 0.0125 -> rejected; 0.03 > 0.05/3 = 0.0167 -> not.
    assert corrected["a"]["significant"]
    assert not corrected["b"]["significant"]
    assert not corrected["d"]["significant"]


def test_cohens_h_is_zero_for_equal_proportions():
    assert effect_size_cohens_h(0.6, 0.6) == pytest.approx(0.0)
    assert effect_size_cohens_h(0.5, 0.9) > 0.5


def test_confusion_matrix_row_normalisation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    assert matrix.sum(axis=1) == pytest.approx(np.ones(2))
    assert matrix[0, 0] == pytest.approx(0.75)


def test_calibration_of_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(2)
    probabilities = rng.uniform(0, 1, 5000)
    outcomes = (rng.random(5000) < probabilities).astype(int)
    calibration = calibration_curve(outcomes, probabilities, n_bins=10)
    assert calibration["ece"] < 0.05


def test_per_group_metrics_flags_small_groups():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = y_true.copy()
    groups = np.array(["a"] * 8 + ["b"] * 2)
    rows = per_group_metrics(y_true, y_pred, groups)
    by_group = {r["group"]: r for r in rows}
    assert "macc" in by_group["a"]
    assert "note" in by_group["b"]      # too small to score
