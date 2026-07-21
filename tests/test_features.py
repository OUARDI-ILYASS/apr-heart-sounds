"""Feature-extraction contracts.

Tests that need librosa / pywt are marked `requires_optional` and skipped when
the library is absent, so the core suite still runs in a bare environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.base import (
    aggregate_over_time, aggregation_names, safe_log, sanitize,
    AGGREGATION_FUNCTIONS,
)



def _has(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


requires_librosa = pytest.mark.skipif(not _has("librosa"), reason="librosa not installed")
requires_pywt = pytest.mark.skipif(not _has("pywt"), reason="pywt not installed")


# --------------------------------------------------------------------------- #
# Aggregation (no optional dependencies)
# --------------------------------------------------------------------------- #
def test_aggregation_output_length():
    matrix = np.random.rand(13, 100)
    vector = aggregate_over_time(matrix, ["mean", "std", "min", "max"])
    assert vector.shape == (13 * 4,)


def test_aggregation_is_coefficient_major():
    """All statistics of coefficient 0 come first, then coefficient 1, etc."""
    matrix = np.zeros((3, 10))
    matrix[1, :] = 5.0
    vector = aggregate_over_time(matrix, ["mean", "std"])
    # coefficient 1's mean sits at index 1*2 + 0 = 2
    assert vector[2] == pytest.approx(5.0)
    assert vector[0] == pytest.approx(0.0)


def test_aggregation_names_match_ordering():
    names = aggregation_names(["mfcc0", "mfcc1"], ["mean", "std"])
    assert names == ["mfcc0_mean", "mfcc0_std", "mfcc1_mean", "mfcc1_std"]


def test_skew_and_kurtosis_of_constant_are_zero_not_nan():
    """scipy returns NaN for a constant sequence; we must not propagate that."""
    matrix = np.full((5, 50), 3.0)
    vector = aggregate_over_time(matrix, ["skew", "kurtosis"])
    assert np.all(np.isfinite(vector))
    assert np.allclose(vector, 0.0)


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_over_time(np.random.rand(3, 10), ["nonsense"])


def test_safe_log_handles_zero():
    assert np.isfinite(safe_log(np.array([0.0, 1.0, 1e-30]))).all()


def test_sanitize_reports_and_replaces():
    array = np.array([1.0, np.nan, np.inf, -np.inf])
    cleaned, report = sanitize(array)
    assert np.all(np.isfinite(cleaned))
    assert report["n_nan"] == 1 and report["n_inf"] == 2


def test_all_declared_aggregations_are_implemented(cfg):
    for name in cfg["features"]["mfcc"]["aggregations"]:
        assert name in AGGREGATION_FUNCTIONS


# --------------------------------------------------------------------------- #
# MFCC
# --------------------------------------------------------------------------- #
@requires_librosa
@pytest.mark.requires_optional
def test_mfcc_dimension_matches_config(cfg, synthetic_pcg):
    from src.features.mfcc import MFCCExtractor

    extractor = MFCCExtractor(float(cfg["preprocessing"]["target_sr"]), cfg)
    features = extractor.transform(synthetic_pcg())
    assert features.shape == extractor.output_shape
    assert features.shape[0] == 13 * 3 * 6
    assert len(extractor.feature_names) == features.shape[0]
    assert np.all(np.isfinite(features))


@requires_librosa
@pytest.mark.requires_optional
def test_dct_frequency_map_rows_sum_to_one(cfg):
    from src.features.mfcc import MFCCExtractor

    extractor = MFCCExtractor(2000.0, cfg)
    mapping = extractor.dct_frequency_map()
    assert mapping.shape == (13, cfg["features"]["n_mels"])
    np.testing.assert_allclose(mapping.sum(axis=1), 1.0, rtol=1e-6)


@requires_librosa
@pytest.mark.requires_optional
def test_mfcc_is_deterministic(cfg, synthetic_pcg):
    from src.features.mfcc import MFCCExtractor

    extractor = MFCCExtractor(2000.0, cfg)
    signal = synthetic_pcg()
    np.testing.assert_array_equal(extractor.transform(signal),
                                  extractor.transform(signal))


# --------------------------------------------------------------------------- #
# log-Mel
# --------------------------------------------------------------------------- #
@requires_librosa
@pytest.mark.requires_optional
def test_logmel_shape_matches_derived_shapes(cfg, synthetic_pcg):
    from src.features.logmel import LogMelExtractor
    from src.config.schema import derived_shapes

    extractor = LogMelExtractor(2000.0, cfg)
    features = extractor.transform(synthetic_pcg())
    expected = derived_shapes(cfg.to_dict())["logmel_shape"]
    assert list(features.shape) == expected


@requires_librosa
@pytest.mark.requires_optional
def test_logmel_band_frequencies_are_monotonic_and_in_band(cfg):
    from src.features.logmel import LogMelExtractor

    extractor = LogMelExtractor(2000.0, cfg)
    frequencies = extractor.band_frequencies()
    assert np.all(np.diff(frequencies) > 0)
    assert frequencies[0] >= cfg["features"]["fmin"] - 1
    assert frequencies[-1] <= cfg["features"]["fmax"] + 1


@requires_librosa
@pytest.mark.requires_optional
def test_linear_filterbank_ablation_changes_the_output(cfg, synthetic_pcg):
    from src.features.logmel import LogMelExtractor
    import copy

    signal = synthetic_pcg()
    mel = LogMelExtractor(2000.0, cfg).transform(signal)

    linear_cfg = copy.deepcopy(cfg.to_dict())
    linear_cfg["features"]["logmel"]["scale"] = "linear"
    from src.config.loader import ConfigDict
    linear = LogMelExtractor(2000.0, ConfigDict(linear_cfg)).transform(signal)

    assert mel.shape == linear.shape
    # Should differ, but only mildly - the mel scale is near-linear below 1 kHz,
    # which is exactly the point ablation A2 makes.
    assert not np.allclose(mel, linear)


# --------------------------------------------------------------------------- #
# PWP
# --------------------------------------------------------------------------- #
@requires_pywt
@pytest.mark.requires_optional
def test_pwp_dimension_and_band_coverage(cfg, synthetic_pcg):
    from src.features.pwp import PWPExtractor

    extractor = PWPExtractor(2000.0, cfg)
    features = extractor.transform(synthetic_pcg())
    assert features.shape == extractor.output_shape
    assert len(extractor.feature_names) == features.shape[0]
    assert np.all(np.isfinite(features))


@requires_pywt
@pytest.mark.requires_optional
def test_pwp_every_perceptual_band_receives_nodes(cfg):
    from src.features.pwp import PWPExtractor

    extractor = PWPExtractor(2000.0, cfg)
    counts = np.bincount(extractor.band_assignment, minlength=extractor.n_bands)
    assert np.all(counts > 0), f"empty perceptual bands: {counts}"


@requires_pywt
@pytest.mark.requires_optional
def test_pwp_band_edges_are_monotonic_and_within_range(cfg):
    from src.features.pwp import PWPExtractor

    extractor = PWPExtractor(2000.0, cfg)
    edges = extractor.band_edges
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(cfg["features"]["pwp"]["band_low"], abs=1)
    assert edges[-1] == pytest.approx(cfg["features"]["pwp"]["band_high"], abs=1)


@requires_pywt
@pytest.mark.requires_optional
def test_pwp_rejects_natural_node_ordering(cfg):
    from src.features.pwp import PWPExtractor
    from src.config.loader import ConfigDict
    import copy

    broken = copy.deepcopy(cfg.to_dict())
    broken["features"]["pwp"]["node_order"] = "natural"
    with pytest.raises(ValueError, match="scrambles"):
        PWPExtractor(2000.0, ConfigDict(broken))


@requires_pywt
@pytest.mark.requires_optional
def test_pwp_responds_to_a_murmur(cfg, synthetic_pcg):
    """A synthetic murmur must move the entropy descriptors.

    This is the behavioural test that the feature set measures something
    related to the thing we care about, rather than merely producing numbers.
    """
    from src.features.pwp import PWPExtractor

    extractor = PWPExtractor(2000.0, cfg)
    clean = extractor.transform(synthetic_pcg(murmur=False, seed=1))
    murmur = extractor.transform(synthetic_pcg(murmur=True, seed=1))
    assert not np.allclose(clean, murmur)
    # Broadband noise added in systole should raise total spectral entropy.
    entropy_idx = [i for i, n in enumerate(extractor.feature_names)
                   if n.endswith("shannon_entropy")]
    assert murmur[entropy_idx].sum() > clean[entropy_idx].sum()


# --------------------------------------------------------------------------- #
# Preprocessing and segmentation (scipy only)
# --------------------------------------------------------------------------- #
def test_bandpass_attenuates_out_of_band_tones(cfg):
    from src.data.preprocessing import bandpass_filter

    sr = 2000
    t = np.arange(sr * 2) / sr
    in_band = np.sin(2 * np.pi * 150 * t)
    out_of_band = np.sin(2 * np.pi * 800 * t)

    filtered_in = bandpass_filter(in_band, 25, 400, sr)
    filtered_out = bandpass_filter(out_of_band, 25, 400, sr)

    assert np.std(filtered_in) > 0.5 * np.std(in_band)
    assert np.std(filtered_out) < 0.1 * np.std(out_of_band)


def test_zero_phase_filtering_preserves_peak_position():
    """The reason we use filtfilt: a causal filter would shift the transient."""
    from src.data.preprocessing import bandpass_filter

    sr = 2000
    x = np.zeros(sr)
    x[500] = 1.0
    t = np.arange(sr) / sr
    x = x + 0.01 * np.sin(2 * np.pi * 100 * t)

    zero_phase = bandpass_filter(x, 25, 400, sr, zero_phase=True)
    causal = bandpass_filter(x, 25, 400, sr, zero_phase=False)

    assert abs(int(np.argmax(np.abs(zero_phase))) - 500) < 15
    assert abs(int(np.argmax(np.abs(causal))) - 500) > \
           abs(int(np.argmax(np.abs(zero_phase))) - 500)


def test_spike_removal_suppresses_an_artificial_transient():
    from src.data.preprocessing import remove_spikes

    sr = 2000
    rng = np.random.default_rng(0)
    x = 0.1 * rng.normal(0, 1, sr * 3).astype(np.float32)
    x[3000:3050] = 5.0
    cleaned, n_removed = remove_spikes(x, sr)
    assert n_removed >= 1
    assert np.max(np.abs(cleaned)) < np.max(np.abs(x))


def test_normalization_produces_unit_variance():
    from src.data.preprocessing import normalize_signal

    x = np.random.normal(5, 3, 1000).astype(np.float32)
    z = normalize_signal(x, "zscore")
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std()) - 1.0) < 1e-4


def test_segmentation_geometry(cfg):
    from src.data.segmentation import segment_signal

    sr = 2000
    x = np.random.rand(sr * 10).astype(np.float32)
    segments, meta = segment_signal(x, sr, 3.0, 1.5)
    assert segments.shape[1] == 6000
    assert len(meta) == len(segments)
    # Consecutive windows are one hop apart.
    assert meta[1]["start_sample"] - meta[0]["start_sample"] == 3000


def test_short_recording_is_dropped_not_silently_padded():
    from src.data.segmentation import segment_signal

    segments, meta = segment_signal(np.random.rand(1000).astype(np.float32),
                                    2000, 3.0, 1.5, pad_policy="drop")
    assert len(segments) == 0 and len(meta) == 0


def test_train_hop_differs_from_eval_hop(cfg):
    from src.data.segmentation import hop_for_split

    train_hop = hop_for_split("train", cfg["segmentation"])
    eval_hop = hop_for_split("test", cfg["segmentation"])
    assert train_hop < eval_hop, "training should overlap, evaluation should not"
