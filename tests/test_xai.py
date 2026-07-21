"""Segmenter and alignment-metric behaviour.

These use synthetic signals with a known cardiac structure, so the expected
answer is known in advance rather than being whatever the code happens to
produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.xai.segmenter import (
    shannon_energy_envelope, estimate_cycle_length, find_sound_peaks,
    segment_cardiac_cycle, states_to_frames, STATE_CODES,
)
from src.xai.alignment import (
    alignment_score, uniform_null, shuffled_null, permutation_p_value,
)


def test_envelope_is_non_negative_and_peaks_at_sounds(synthetic_pcg):
    x = synthetic_pcg(heart_rate=60)
    envelope = shannon_energy_envelope(x, 2000)
    assert len(envelope) == len(x)
    assert np.all(envelope >= 0)


def test_cycle_length_recovers_the_true_heart_rate(synthetic_pcg):
    for heart_rate in [60.0, 75.0, 90.0]:
        x = synthetic_pcg(duration_s=6.0, heart_rate=heart_rate)
        envelope = shannon_energy_envelope(x, 2000)
        cycle, confidence = estimate_cycle_length(envelope, 2000)
        estimated = 60.0 / cycle
        assert abs(estimated - heart_rate) < 10.0, \
            f"expected ~{heart_rate} bpm, got {estimated:.1f}"
        assert confidence > 0.1


def test_peak_detection_finds_roughly_two_sounds_per_cycle(synthetic_pcg):
    x = synthetic_pcg(duration_s=6.0, heart_rate=60)
    envelope = shannon_energy_envelope(x, 2000)
    cycle, _ = estimate_cycle_length(envelope, 2000)
    peaks = find_sound_peaks(envelope, 2000, cycle)
    # 6 s at 60 bpm = 6 cycles = up to 12 sounds; allow generous tolerance
    # because the envelope smoothing merges some.
    assert 5 <= len(peaks) <= 20


def test_segmentation_assigns_all_four_states(cfg, synthetic_pcg):
    x = synthetic_pcg(duration_s=3.0, heart_rate=70)
    result = segment_cardiac_cycle(x, 2000, cfg)
    present = {name for name, code in STATE_CODES.items()
               if name != "unknown" and np.any(result["states"] == code)}
    assert "systole" in present and "diastole" in present


def test_systole_is_shorter_than_diastole(cfg, synthetic_pcg):
    """The physiological prior the segmenter relies on."""
    x = synthetic_pcg(duration_s=6.0, heart_rate=65)
    result = segment_cardiac_cycle(x, 2000, cfg)
    if result["usable"]:
        fractions = result["state_fractions"]
        assert fractions["systole"] < fractions["diastole"]


def test_segmentation_of_noise_reports_low_confidence(cfg):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 6000).astype(np.float32)
    result = segment_cardiac_cycle(noise, 2000, cfg)
    # White noise has no cardiac cycle; the segmenter must say so rather than
    # inventing one.
    assert result["confidence"] < 0.7


def test_states_to_frames_uses_modal_assignment():
    states = np.array([0] * 32 + [1] * 32, dtype=np.int8)
    frames = states_to_frames(states, hop_length=32, n_frames=2)
    assert frames[0] == 0 and frames[1] == 1


# --------------------------------------------------------------------------- #
# Alignment metric
# --------------------------------------------------------------------------- #
def test_uniform_attribution_gives_enrichment_of_one():
    frame_states = np.array([STATE_CODES["systole"]] * 30 +
                            [STATE_CODES["diastole"]] * 70, dtype=np.int8)
    attribution = np.ones(100)
    result = alignment_score(attribution, frame_states)
    assert result["enrichment_systole"] == pytest.approx(1.0, abs=1e-6)
    assert result["mass_systole"] == pytest.approx(0.30, abs=1e-6)


def test_perfectly_targeted_attribution_gives_high_enrichment():
    frame_states = np.array([STATE_CODES["systole"]] * 30 +
                            [STATE_CODES["diastole"]] * 70, dtype=np.int8)
    attribution = np.zeros(100)
    attribution[:30] = 1.0                # all mass inside systole
    result = alignment_score(attribution, frame_states)
    assert result["mass_systole"] == pytest.approx(1.0)
    assert result["enrichment_systole"] == pytest.approx(1 / 0.30, rel=1e-3)


def test_attribution_avoiding_a_state_gives_enrichment_below_one():
    frame_states = np.array([STATE_CODES["systole"]] * 30 +
                            [STATE_CODES["diastole"]] * 70, dtype=np.int8)
    attribution = np.zeros(100)
    attribution[30:] = 1.0
    result = alignment_score(attribution, frame_states)
    assert result["enrichment_systole"] == pytest.approx(0.0)
    assert result["enrichment_diastole"] > 1.0


def test_2d_map_is_marginalised_over_frequency():
    frame_states = np.array([STATE_CODES["systole"]] * 50 +
                            [STATE_CODES["diastole"]] * 50, dtype=np.int8)
    cam = np.zeros((32, 100))
    cam[:, :50] = 1.0
    result = alignment_score(cam, frame_states)
    assert result["mass_systole"] == pytest.approx(1.0)


def test_unknown_frames_are_excluded_from_both_numerator_and_denominator():
    frame_states = np.array([STATE_CODES["unknown"]] * 50 +
                            [STATE_CODES["systole"]] * 50, dtype=np.int8)
    attribution = np.ones(100)
    result = alignment_score(attribution, frame_states)
    # Only the 50 labelled frames count, and they are all systole.
    assert result["n_frames_used"] == 50
    assert result["time_systole"] == pytest.approx(1.0)
    assert result["enrichment_systole"] == pytest.approx(1.0)


def test_invalid_input_is_reported_not_crashed():
    frame_states = np.full(100, STATE_CODES["unknown"], dtype=np.int8)
    result = alignment_score(np.ones(100), frame_states)
    assert result["valid"] is False


def test_uniform_null_matches_the_analytic_value():
    frame_states = np.array([STATE_CODES["systole"]] * 33 +
                            [STATE_CODES["diastole"]] * 67, dtype=np.int8)
    null = uniform_null(frame_states)
    assert null["enrichment_systole"] == pytest.approx(1.0, abs=1e-6)


def test_shuffled_null_centres_on_the_time_fraction():
    frame_states = np.array([STATE_CODES["systole"]] * 30 +
                            [STATE_CODES["diastole"]] * 70, dtype=np.int8)
    rng = np.random.default_rng(0)
    attribution = rng.random(100)
    null = shuffled_null(attribution, frame_states, n_permutations=200, seed=0,
                         states=("systole",))
    assert null["systole"]["mean"] == pytest.approx(0.30, abs=0.05)


def test_permutation_p_value_never_returns_zero():
    null = np.random.default_rng(0).normal(0, 1, 100)
    assert permutation_p_value(1e9, null) > 0.0


def test_permutation_p_value_is_large_for_a_typical_observation():
    null = np.random.default_rng(0).normal(0, 1, 1000)
    assert permutation_p_value(0.0, null) > 0.3
