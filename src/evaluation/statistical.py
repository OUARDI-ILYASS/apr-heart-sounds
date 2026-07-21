"""Statistical tests: is the difference between two models real?

PROFESSOR Q: "Model A got 0.84 MAcc and model B got 0.82. Is A better?"
A: Not necessarily, and the whole point of this module is to answer that
   properly rather than by eyeballing decimals. With a test set of ~470
   recordings, a 2-point MAcc difference is well inside sampling noise. We
   report (a) bootstrap confidence intervals for each model's metric and (b)
   McNemar's test on the paired predictions. If the CIs overlap heavily and
   McNemar's p is large, the honest conclusion is that the models are
   statistically indistinguishable - which is itself a finding worth stating,
   especially when the simpler model is 100x cheaper to train.

PROFESSOR Q: "Why McNemar and not a t-test on accuracy?"
A: Because the two models are evaluated on the *same* recordings, so the
   samples are paired, not independent. McNemar's test uses exactly the right
   information: the two discordant cells (A right & B wrong, A wrong & B right)
   and ignores the cases where both agree, which carry no evidence about a
   difference.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 n_resamples: int = 1000, ci: float = 0.95,
                 seed: int = 42,
                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for any metric.

    Resampling is done over RECORDINGS with replacement, which is the correct
    unit of independence. Bootstrapping over segments would treat the 8
    segments of one recording as 8 independent observations and produce
    intervals roughly sqrt(8) times too narrow.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = float(metric_fn(y_true, y_pred))
    scores: List[float] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        # A resample containing a single class makes most metrics undefined.
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue

    if not scores:
        return {"point": point, "lower": float("nan"), "upper": float("nan"),
                "n_valid_resamples": 0}

    alpha = (1.0 - ci) / 2.0
    scores_array = np.asarray(scores)
    return {
        "point": point,
        "lower": float(np.percentile(scores_array, 100 * alpha)),
        "upper": float(np.percentile(scores_array, 100 * (1 - alpha))),
        "std": float(scores_array.std()),
        "ci_level": ci,
        "n_valid_resamples": len(scores),
    }


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                 correction: bool = True) -> Dict[str, float]:
    """McNemar's paired test on two models' predictions.

    Returns the discordant counts, the statistic, and a p-value. When the
    discordant total is small (<25) we use the exact binomial test instead of
    the chi-square approximation, which is the standard recommendation.
    """
    from scipy import stats

    y_true = np.asarray(y_true).astype(int)
    correct_a = np.asarray(y_pred_a).astype(int) == y_true
    correct_b = np.asarray(y_pred_b).astype(int) == y_true

    # b: A right, B wrong.  c: A wrong, B right.
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c

    result: Dict[str, float] = {
        "both_correct": int(np.sum(correct_a & correct_b)),
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
        "a_only_correct": b,
        "b_only_correct": c,
        "n_discordant": n_discordant,
    }

    if n_discordant == 0:
        result.update(statistic=0.0, p_value=1.0, test="degenerate",
                      note="The two models made identical predictions.")
        return result

    if n_discordant < 25:
        # Exact binomial: under H0 each discordant case is a fair coin flip.
        p_value = float(stats.binomtest(b, n_discordant, 0.5).pvalue)
        result.update(statistic=float(min(b, c)), p_value=p_value, test="exact_binomial")
    else:
        numerator = (abs(b - c) - (1.0 if correction else 0.0)) ** 2
        statistic = numerator / n_discordant
        result.update(
            statistic=float(statistic),
            p_value=float(stats.chi2.sf(statistic, df=1)),
            test="chi2_with_continuity_correction" if correction else "chi2",
        )

    result["significant_at_0.05"] = bool(result["p_value"] < 0.05)
    result["better_model"] = "A" if b > c else ("B" if c > b else "tie")
    return result


def permutation_test(values_a: np.ndarray, values_b: np.ndarray,
                     n_permutations: int = 10000, seed: int = 42
                     ) -> Dict[str, float]:
    """Two-sided permutation test on a difference of means.

    Used in the XAI section to compare an observed attribution-alignment score
    against a null distribution, where no parametric assumption is available.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    observed = float(a.mean() - b.mean())

    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= abs(observed):
            count += 1

    # +1 in numerator and denominator: the observed arrangement is itself one
    # of the possible permutations, so a p-value of exactly 0 is not attainable.
    p_value = (count + 1) / (n_permutations + 1)
    return {
        "observed_difference": observed,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05
                    ) -> Dict[str, Dict[str, object]]:
    """Holm-Bonferroni correction for a family of tests.

    PROFESSOR Q: "You ran several pairwise comparisons. Did you correct for
                  multiple testing?"
    A: Yes, with Holm-Bonferroni, which is uniformly more powerful than plain
       Bonferroni while controlling the same family-wise error rate. With six
       pairwise model comparisons an uncorrected 0.05 threshold gives roughly a
       26% chance of at least one spurious 'significant' result.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    out: Dict[str, Dict[str, object]] = {}
    previous_rejected = True

    for rank, (name, p) in enumerate(items):
        threshold = alpha / (n - rank)
        rejected = bool(p < threshold and previous_rejected)
        previous_rejected = rejected
        out[name] = {
            "p_value": float(p),
            "rank": rank + 1,
            "adjusted_threshold": float(threshold),
            "significant": rejected,
        }
    return out


def effect_size_cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions - the effect size to report next to p.

    A p-value tells you whether a difference is detectable; h tells you whether
    it is worth caring about. Conventional reading: 0.2 small, 0.5 medium,
    0.8 large.
    """
    phi1 = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
    phi2 = 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))
    return float(abs(phi1 - phi2))
