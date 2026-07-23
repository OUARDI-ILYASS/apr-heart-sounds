"""Cardiac-cycle alignment: turning a heatmap into a number you can test.

This module contains the study's main methodological contribution. Grad-CAM and
SHAP produce attributions; the usual practice in applied papers is to show three
pretty heatmaps and assert that the model "focuses on clinically relevant
regions". That assertion is unfalsifiable as stated. Here we make it falsifiable.

**The alignment score.** Given an attribution map normalised to unit mass and a
cardiac state map (S1 / systole / S2 / diastole) obtained independently of the
model, the alignment score for state *s* is

    A_s = sum of attribution mass falling inside state s

Because the map has unit mass, ``A_s`` is a fraction in [0, 1]. On its own it
means nothing: a model attending uniformly over time scores exactly ``T_s``, the
fraction of *time* occupied by state s. So the quantity we actually report is
the **enrichment**

    E_s = A_s / T_s

with ``E_s = 1`` meaning "no preference for this state", ``E_s > 1`` meaning
"attends to this state more than chance".

**Why systole.** Most clinically important murmurs in an adult screening
population are systolic - aortic stenosis, mitral regurgitation, the innocent
flow murmurs. If a model has learned pathology rather than an artefact, its
evidence should concentrate between S1 and S2. This gives us a *prediction*
made before looking at the results, which is what separates a hypothesis test
from a story told after the fact.

**Null baselines.** Three, all reported:
  * *uniform*  - attribution spread evenly over the segment; scores E = 1 by
    construction. The analytic null.
  * *shuffled* - the real attribution map with its time axis permuted. Preserves
    the map's marginal distribution (its sparsity, its dynamic range) while
    destroying any temporal relationship to the cardiac cycle. This is the
    strong null, and the one the permutation test uses.
  * *state-shuffled* - the real map against a cardiac state map taken from a
    *different* segment. Controls for the possibility that both the map and the
    state sequence are periodic at similar rates and would correlate by
    coincidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .segmenter import STATE_CODES, STATE_NAMES, states_to_frames


# --------------------------------------------------------------------------- #
# Core score
# --------------------------------------------------------------------------- #
def alignment_score(attribution: np.ndarray, frame_states: np.ndarray,
                    states: Sequence[str] = ("S1", "systole", "S2", "diastole")
                    ) -> Dict[str, object]:
    """Attribution mass and enrichment per cardiac state, for one segment.

    Parameters
    ----------
    attribution
        Either a 2-D Grad-CAM map ``(n_bands, n_frames)`` or a 1-D temporal
        profile ``(n_frames,)``. 2-D maps are marginalised over frequency,
        because the cardiac state is a purely temporal variable - assigning
        attribution at 300 Hz to "systole" is meaningful, assigning it to a
        frequency-specific state is not.
    frame_states
        Integer state code per frame, from ``states_to_frames``.
    """
    attribution = np.asarray(attribution, dtype=np.float64)
    profile = attribution.sum(axis=0) if attribution.ndim == 2 else attribution

    frame_states = np.asarray(frame_states)
    n = min(len(profile), len(frame_states))
    profile, frame_states = profile[:n], frame_states[:n]

    # Consider only frames the segmenter could label. Unknown frames are
    # excluded from BOTH the attribution mass and the time budget, so the two
    # fractions remain comparable.
    known = frame_states != STATE_CODES["unknown"]
    if not known.any():
        return {"valid": False, "reason": "no frames could be assigned a cardiac state"}

    profile = profile[known]
    frame_states = frame_states[known]
    total_mass = profile.sum()
    if total_mass <= 0:
        return {"valid": False, "reason": "attribution map has zero total mass"}
    profile = profile / total_mass

    out: Dict[str, object] = {"valid": True, "n_frames_used": int(known.sum())}
    for state in states:
        code = STATE_CODES[state]
        mask = frame_states == code
        time_fraction = float(mask.mean())
        mass_fraction = float(profile[mask].sum())
        out[f"mass_{state}"] = mass_fraction
        out[f"time_{state}"] = time_fraction
        # Enrichment is undefined when a state never occurs in this segment.
        out[f"enrichment_{state}"] = (
            float(mass_fraction / time_fraction) if time_fraction > 1e-6 else float("nan")
        )

    # Convenience roll-ups used repeatedly downstream.
    out["mass_sounds"] = out.get("mass_S1", 0.0) + out.get("mass_S2", 0.0)
    out["time_sounds"] = out.get("time_S1", 0.0) + out.get("time_S2", 0.0)
    out["enrichment_sounds"] = (
        float(out["mass_sounds"] / out["time_sounds"]) if out["time_sounds"] > 1e-6
        else float("nan")
    )
    return out


# --------------------------------------------------------------------------- #
# Null baselines
# --------------------------------------------------------------------------- #
def uniform_null(frame_states: np.ndarray,
                 states: Sequence[str] = ("S1", "systole", "S2", "diastole")
                 ) -> Dict[str, float]:
    """Analytic null: a flat attribution profile. Enrichment is 1 everywhere."""
    frame_states = np.asarray(frame_states)
    known = frame_states != STATE_CODES["unknown"]
    uniform = np.ones(int(known.sum()), dtype=np.float64)
    result = alignment_score(uniform, frame_states[known], states)
    return {k: v for k, v in result.items() if isinstance(v, float)}


def shuffled_null(attribution: np.ndarray, frame_states: np.ndarray,
                  n_permutations: int = 1000, seed: int = 42,
                  states: Sequence[str] = ("S1", "systole", "S2", "diastole")
                  ) -> Dict[str, Dict[str, float]]:
    """Empirical null from permuting the attribution profile in time.

    Preserving the profile's values and permuting only their order is the right
    null here: it keeps the map's sparsity and dynamic range - so a very peaky
    map is compared against equally peaky random maps - while destroying any
    alignment with the cardiac cycle.
    """
    attribution = np.asarray(attribution, dtype=np.float64)
    profile = attribution.sum(axis=0) if attribution.ndim == 2 else attribution

    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {state: [] for state in states}

    for _ in range(n_permutations):
        result = alignment_score(rng.permutation(profile), frame_states, states)
        if not result.get("valid"):
            continue
        for state in states:
            samples[state].append(float(result[f"mass_{state}"]))

    out: Dict[str, Dict[str, float]] = {}
    for state, values in samples.items():
        array = np.asarray(values)
        out[state] = {
            "mean": float(array.mean()) if array.size else float("nan"),
            "std": float(array.std()) if array.size else float("nan"),
            "p05": float(np.percentile(array, 5)) if array.size else float("nan"),
            "p95": float(np.percentile(array, 95)) if array.size else float("nan"),
            "n": int(array.size),
        }
    return out


def permutation_p_value(observed: float, null_samples: np.ndarray,
                        alternative: str = "greater") -> float:
    """One-sided permutation p-value with the +1 correction.

    The +1 in both numerator and denominator accounts for the observed
    arrangement itself being one of the possible permutations, which is why a
    permutation test can never legitimately report p = 0.
    """
    null_samples = np.asarray(null_samples, dtype=np.float64)
    if null_samples.size == 0:
        return float("nan")
    if alternative == "greater":
        count = int(np.sum(null_samples >= observed))
    elif alternative == "less":
        count = int(np.sum(null_samples <= observed))
    else:
        count = int(np.sum(np.abs(null_samples - null_samples.mean())
                           >= abs(observed - null_samples.mean())))
    return float((count + 1) / (null_samples.size + 1))


# --------------------------------------------------------------------------- #
# Population-level analysis
# --------------------------------------------------------------------------- #
def batch_alignment(attributions: np.ndarray, segmentations: List[Dict],
                    hop_length: int, n_frames: int, cfg: Dict,
                    labels: Optional[np.ndarray] = None,
                    predictions: Optional[np.ndarray] = None
                    ) -> Dict[str, object]:
    """Alignment across many segments, with exclusions accounted for.

    Only segments whose segmentation passed the confidence threshold are
    included. The exclusion rate is returned and belongs in the paper: an
    alignment score computed on 62% of segments is a different claim from one
    computed on all of them, and the reader is entitled to know which.
    """
    states = list(cfg["xai"]["alignment"]["states"])
    threshold = float(cfg["xai"]["segmentation"]["confidence_threshold"])

    rows: List[Dict[str, object]] = []
    n_excluded_confidence = 0
    n_excluded_invalid = 0

    for index, (attribution, segmentation) in enumerate(zip(attributions, segmentations)):
        if not segmentation.get("usable", False) or \
                segmentation.get("confidence", 0.0) < threshold:
            n_excluded_confidence += 1
            continue

        frame_states = states_to_frames(segmentation["states"], hop_length, n_frames)
        result = alignment_score(attribution, frame_states, states)
        if not result.get("valid"):
            n_excluded_invalid += 1
            continue

        row: Dict[str, object] = {"index": index, **result}
        row["segmenter_confidence"] = float(segmentation.get("confidence", 0.0))
        row["heart_rate_bpm"] = float(segmentation.get("heart_rate_bpm", float("nan")))
        if labels is not None:
            row["label"] = int(labels[index])
        if predictions is not None:
            row["prediction"] = int(predictions[index])
        rows.append(row)

    n_total = len(attributions)
    summary: Dict[str, object] = {
        "n_total": n_total,
        "n_included": len(rows),
        "n_excluded_low_confidence": n_excluded_confidence,
        "n_excluded_invalid": n_excluded_invalid,
        "inclusion_rate": round(len(rows) / max(1, n_total), 4),
        "per_segment": rows,
    }

    if not rows:
        summary["note"] = ("No segment survived the confidence threshold. The "
                           "alignment analysis cannot be reported for this model.")
        return summary

    for state in states:
        mass = np.array([r[f"mass_{state}"] for r in rows], dtype=float)
        time = np.array([r[f"time_{state}"] for r in rows], dtype=float)
        enrichment = np.array([r[f"enrichment_{state}"] for r in rows], dtype=float)
        enrichment = enrichment[np.isfinite(enrichment)]

        summary[f"mean_mass_{state}"] = float(mass.mean())
        summary[f"mean_time_{state}"] = float(time.mean())
        summary[f"mean_enrichment_{state}"] = float(enrichment.mean()) if enrichment.size else float("nan")
        summary[f"median_enrichment_{state}"] = float(np.median(enrichment)) if enrichment.size else float("nan")
        # Fraction of segments where attribution exceeds the time budget - a
        # per-segment, non-parametric restatement of the same claim.
        summary[f"frac_enriched_{state}"] = float(np.mean(enrichment > 1.0)) if enrichment.size else float("nan")

    return summary


def stratified_alignment(alignment: Dict[str, object], cfg: Dict
                         ) -> Dict[str, Dict[str, float]]:
    """Alignment split by outcome category (TP / TN / FP / FN).

    The most informative comparison in the whole XAI section. If the model
    attends to systole on true positives but not on false positives, its
    attention is tracking real evidence rather than a generic habit. If the
    pattern is identical for both, the "focus" is a property of the
    architecture, not of the diagnosis.
    """
    rows = alignment.get("per_segment", [])
    if not rows or "label" not in rows[0] or "prediction" not in rows[0]:
        return {}

    primary = str(cfg["xai"]["alignment"]["primary_state"])
    categories = {
        "true_positive": lambda r: r["label"] == 1 and r["prediction"] == 1,
        "true_negative": lambda r: r["label"] == 0 and r["prediction"] == 0,
        "false_positive": lambda r: r["label"] == 0 and r["prediction"] == 1,
        "false_negative": lambda r: r["label"] == 1 and r["prediction"] == 0,
    }

    out: Dict[str, Dict[str, float]] = {}
    for name, predicate in categories.items():
        subset = [r for r in rows if predicate(r)]
        if not subset:
            out[name] = {"n": 0}
            continue
        enrichment = np.array([r[f"enrichment_{primary}"] for r in subset], dtype=float)
        enrichment = enrichment[np.isfinite(enrichment)]
        mass = np.array([r[f"mass_{primary}"] for r in subset], dtype=float)
        out[name] = {
            "n": len(subset),
            f"mean_enrichment_{primary}": float(enrichment.mean()) if enrichment.size else float("nan"),
            f"mean_mass_{primary}": float(mass.mean()),
            f"frac_enriched_{primary}": float(np.mean(enrichment > 1.0)) if enrichment.size else float("nan"),
        }
    return out


def alignment_significance(alignment: Dict[str, object],
                           attributions: np.ndarray,
                           segmentations: List[Dict], hop_length: int,
                           n_frames: int, cfg: Dict, seed: int = 42
                           ) -> Dict[str, object]:
    """Test the observed enrichment against the shuffled null.

    Uses a subsample of segments because the permutation null is recomputed per
    segment; the subsample size is reported so the test's power is transparent.
    """
    primary = str(cfg["xai"]["alignment"]["primary_state"])
    n_permutations = int(cfg["xai"]["alignment"]["n_permutations"])
    rows = alignment.get("per_segment", [])
    if not rows:
        return {"note": "No usable segments; significance test not run."}

    rng = np.random.default_rng(seed)
    n_test = min(200, len(rows))
    chosen = rng.choice(len(rows), n_test, replace=False)

    observed_values: List[float] = []
    null_means: List[float] = []
    p_values: List[float] = []

    for position in chosen:
        row = rows[position]
        index = int(row["index"])
        segmentation = segmentations[index]
        frame_states = states_to_frames(segmentation["states"], hop_length, n_frames)

        attribution = np.asarray(attributions[index], dtype=np.float64)
        profile = attribution.sum(axis=0) if attribution.ndim == 2 else attribution

        observed = float(row[f"mass_{primary}"])
        observed_values.append(observed)

        permutation_rng = np.random.default_rng(seed + index)
        samples = []
        for _ in range(min(n_permutations, 200)):
            result = alignment_score(permutation_rng.permutation(profile),
                                     frame_states, [primary])
            if result.get("valid"):
                samples.append(float(result[f"mass_{primary}"]))
        if samples:
            samples_array = np.asarray(samples)
            null_means.append(float(samples_array.mean()))
            p_values.append(permutation_p_value(observed, samples_array, "greater"))

    if not p_values:
        return {"note": "Permutation null could not be constructed."}

    observed_array = np.asarray(observed_values)
    null_array = np.asarray(null_means)
    p_array = np.asarray(p_values)

    # Cohen's d on the paired difference between observed and null mass.
    difference = observed_array[: len(null_array)] - null_array
    effect_size = float(difference.mean() / (difference.std() + 1e-12))

    return {
        "state": primary,
        "n_segments_tested": len(p_values),
        "n_permutations_per_segment": min(n_permutations, 200),
        "mean_observed_mass": float(observed_array.mean()),
        "mean_null_mass": float(null_array.mean()),
        "mean_p_value": float(p_array.mean()),
        "median_p_value": float(np.median(p_array)),
        "frac_significant_at_0.05": float(np.mean(p_array < 0.05)),
        "effect_size_cohens_d": effect_size,
        # A single segment being significant proves little; the claim is that
        # the enrichment holds across the population.
        "population_conclusion": _conclude(float(np.mean(p_array < 0.05)), effect_size),
    }


def _conclude(fraction_significant: float, effect_size: float) -> str:
    if fraction_significant > 0.5 and effect_size > 0.5:
        return ("Attribution mass concentrates in the target state well beyond "
                "the shuffled null in a majority of segments, with a moderate "
                "or larger effect size. The alignment claim is supported.")
    if fraction_significant > 0.25:
        return ("Attribution mass exceeds the null in a substantial minority of "
                "segments. The alignment claim is weakly supported and should "
                "be stated as a tendency, not a property.")
    return ("Attribution mass is not distinguishable from the shuffled null. "
            "The alignment claim is NOT supported and must be reported as a "
            "negative result.")


# --------------------------------------------------------------------------- #
# Confound checks
# --------------------------------------------------------------------------- #
def energy_confound_check(attributions: np.ndarray, segments: np.ndarray,
                          hop_length: int, n_frames: int) -> Dict[str, float]:
    """Is the attribution map just following signal energy?

    The single most likely alternative explanation for any apparent "focus" in
    an audio saliency map. We correlate the temporal attribution profile with
    the frame-wise RMS envelope of the same segment. A high correlation means
    the model (or the explanation method) is largely an energy detector, and
    any claim about attending to a *specific cardiac phase* would be
    unwarranted, since S1 and S2 are the loudest events.
    """
    correlations: List[float] = []

    for attribution, segment in zip(attributions, segments):
        attribution = np.asarray(attribution, dtype=np.float64)
        profile = attribution.sum(axis=0) if attribution.ndim == 2 else attribution

        envelope = np.array([
            float(np.sqrt(np.mean(segment[f * hop_length:(f + 1) * hop_length] ** 2)))
            if (f + 1) * hop_length <= len(segment) else 0.0
            for f in range(n_frames)
        ])

        n = min(len(profile), len(envelope))
        if n < 3:
            continue
        a, b = profile[:n], envelope[:n]
        if a.std() > 1e-12 and b.std() > 1e-12:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))

    if not correlations:
        return {"note": "Correlation could not be computed."}

    array = np.asarray(correlations)
    mean_correlation = float(array.mean())
    return {
        "mean_correlation_with_envelope": mean_correlation,
        "median_correlation": float(np.median(array)),
        "frac_above_0.5": float(np.mean(array > 0.5)),
        "n_segments": len(correlations),
        "interpretation": (
            "Attribution tracks signal energy closely; phase-specific claims are "
            "confounded and must be withdrawn."
            if mean_correlation > 0.6 else
            "Attribution is partly energy-driven; report the correlation alongside "
            "the alignment score."
            if mean_correlation > 0.3 else
            "Attribution is largely independent of raw energy, supporting a "
            "phase-specific rather than loudness-driven interpretation."
        ),
    }


def build_alignment_table(results: Dict[str, Dict[str, object]], cfg: Dict
                          ) -> List[Dict[str, object]]:
    """Cross-model alignment table - the paper's headline XAI result."""
    states = list(cfg["xai"]["alignment"]["states"])
    rows: List[Dict[str, object]] = []

    for model_name, alignment in results.items():
        if not alignment or alignment.get("n_included", 0) == 0:
            continue
        row: Dict[str, object] = {
            "model": model_name,
            "n_segments": alignment["n_included"],
            "inclusion_rate": alignment["inclusion_rate"],
        }
        for state in states:
            row[f"mass_{state}"] = round(float(alignment.get(f"mean_mass_{state}", float("nan"))), 4)
            row[f"E_{state}"] = round(float(alignment.get(f"mean_enrichment_{state}", float("nan"))), 3)
        rows.append(row)

    return rows
