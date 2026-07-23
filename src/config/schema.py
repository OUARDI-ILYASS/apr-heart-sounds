"""Config validation - fail fast, fail loudly, fail *before* the GPU warms up.

Rather than a heavyweight schema library, this module encodes the handful of
constraints that are actually specific to this pipeline and that would
otherwise fail silently or many minutes into a run.


"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

REQUIRED_SECTIONS: List[str] = [
    "experiment", "paths", "dataset", "preprocessing", "segmentation",
    "splits", "features", "clustering", "classical", "cnn", "evaluation",
    "xai", "figures",
]


class ConfigValidationError(ValueError):
    """Raised when the configuration is internally inconsistent."""


def _require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def usable_fft_bins(sr: float, n_fft: int, fmin: float, fmax: float) -> int:
    """Number of rfft bins whose centre frequency falls inside [fmin, fmax]."""
    bin_width = sr / n_fft
    lo = int(round(fmin / bin_width))
    hi = int(round(fmax / bin_width))
    return max(0, min(hi, n_fft // 2) - lo + 1)


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate a fully-merged config. Raises ``ConfigValidationError``."""
    errors: List[str] = []

    # ---- structural -------------------------------------------------------
    for section in REQUIRED_SECTIONS:
        _require(section in cfg, f"Missing required config section: '{section}'", errors)
    if errors:  # No point continuing if the skeleton is wrong
        raise ConfigValidationError("Config validation failed:\n  - " + "\n  - ".join(errors))

    prep = cfg["preprocessing"]
    seg = cfg["segmentation"]
    feat = cfg["features"]
    sr = float(prep["target_sr"])

    # ---- sampling / filtering --------------------------------------------
    nyquist = sr / 2.0
    _require(
        0 < prep["bandpass_low"] < prep["bandpass_high"] < nyquist,
        f"Bandpass must satisfy 0 < low < high < Nyquist ({nyquist} Hz); got "
        f"[{prep['bandpass_low']}, {prep['bandpass_high']}]",
        errors,
    )
    _require(
        prep["filter_order"] >= 1,
        "preprocessing.filter_order must be >= 1",
        errors,
    )

    # ---- segmentation -----------------------------------------------------
    _require(seg["segment_seconds"] > 0, "segmentation.segment_seconds must be > 0", errors)
    _require(
        0 < seg["train_hop_seconds"] <= seg["segment_seconds"],
        "train_hop_seconds must be in (0, segment_seconds]",
        errors,
    )
    _require(
        0 < seg["eval_hop_seconds"] <= seg["segment_seconds"],
        "eval_hop_seconds must be in (0, segment_seconds]",
        errors,
    )
    _require(
        seg["pad_policy"] in {"drop", "zero_pad"},
        "segmentation.pad_policy must be 'drop' or 'zero_pad'",
        errors,
    )

    # ---- splits -----------------------------------------------------------
    ratios = [cfg["splits"]["train_ratio"], cfg["splits"]["val_ratio"], cfg["splits"]["test_ratio"]]
    _require(
        abs(sum(ratios) - 1.0) < 1e-6,
        f"Split ratios must sum to 1.0; got {sum(ratios):.4f}",
        errors,
    )
    _require(
        cfg["splits"]["level"] == "recording",
        "splits.level must be 'recording'. Splitting at segment level leaks "
        "the same patient into train and test and invalidates every result.",
        errors,
    )

    # ---- the mel-resolution trap -----------------------------------------
    n_fft = int(feat["n_fft"])
    n_mels = int(feat["n_mels"])
    fmin, fmax = float(feat["fmin"]), float(feat["fmax"])
    _require(fmax <= nyquist, f"features.fmax ({fmax}) exceeds Nyquist ({nyquist})", errors)

    n_bins = usable_fft_bins(sr, n_fft, fmin, fmax)
    if n_mels > n_bins:
        # Smallest power-of-two n_fft that would give enough non-degenerate
        # filters: n_bins ~= (fmax - fmin) * n_fft / sr, so we need
        # n_fft >= n_mels * sr / (fmax - fmin), rounded up to a power of 2.
        needed = int(np.ceil(n_mels * sr / max(fmax - fmin, 1e-9)))
        suggested = 1 << max(0, int(needed - 1)).bit_length()
        errors.append(
            f"features.n_mels={n_mels} exceeds the {n_bins} usable FFT bins between "
            f"{fmin}-{fmax} Hz at sr={sr:g}, n_fft={n_fft} (bin width {sr / n_fft:.2f} Hz). "
            f"Filters beyond that are all-zero and inject structural zeros into every "
            f"feature vector. Either lower n_mels to <= {n_bins} or raise n_fft to "
            f">= {suggested}."
        )

    # ---- MFCC -------------------------------------------------------------
    n_mfcc = int(feat["mfcc"]["n_mfcc"])
    _require(
        n_mfcc <= n_mels,
        f"features.mfcc.n_mfcc={n_mfcc} must be <= features.n_mels={n_mels}: "
        "the DCT cannot produce more cepstral coefficients than mel bands.",
        errors,
    )
    _require(
        len(feat["mfcc"]["aggregations"]) > 0,
        "features.mfcc.aggregations must not be empty",
        errors,
    )
    if feat["mfcc"]["use_delta2"] and not feat["mfcc"]["use_delta"]:
        errors.append(
            "features.mfcc.use_delta2=true requires use_delta=true "
            "(delta-delta is computed from delta)."
        )

    # ---- frame count sanity ----------------------------------------------
    seg_samples = int(seg["segment_seconds"] * sr)
    _require(
        seg_samples >= n_fft,
        f"A {seg['segment_seconds']} s segment is {seg_samples} samples, shorter "
        f"than n_fft={n_fft}. Reduce n_fft or lengthen the segment.",
        errors,
    )

    # ---- PWP --------------------------------------------------------------
    pwp = feat["pwp"]
    level = int(pwp["level"])
    _require(1 <= level <= 8, "features.pwp.level should be in [1, 8]", errors)
    subband_width = nyquist / (2 ** level)
    _require(
        subband_width <= (pwp["band_high"] - pwp["band_low"]),
        f"Wavelet packet level {level} gives {subband_width:.1f} Hz subbands, wider "
        f"than the whole band of interest. Increase the level.",
        errors,
    )
    _require(
        pwp["node_order"] == "freq",
        "features.pwp.node_order must be 'freq'. pywt's natural (Paley) ordering "
        "is NOT monotonic in frequency, so using it silently scrambles the "
        "frequency axis of every PWP feature.",
        errors,
    )
    if pwp["perceptual_grouping"]:
        n_nodes_in_band = int((pwp["band_high"] - pwp["band_low"]) / subband_width) + 1
        _require(
            pwp["n_perceptual_bands"] <= n_nodes_in_band,
            f"features.pwp.n_perceptual_bands={pwp['n_perceptual_bands']} exceeds the "
            f"~{n_nodes_in_band} wavelet-packet nodes available in "
            f"[{pwp['band_low']}, {pwp['band_high']}] Hz at level {level}.",
            errors,
        )

    # ---- CNN --------------------------------------------------------------
    arch = cfg["cnn"]["architecture"]
    n_blocks = len(arch["conv_channels"])
    pool = int(arch["pool_size"])
    freq_after = n_mels // (pool ** n_blocks)
    _require(
        freq_after >= 1,
        f"{n_blocks} pooling stages of size {pool} reduce the {n_mels}-band frequency "
        f"axis to {n_mels / pool ** n_blocks:.2f} < 1. Reduce the number of conv "
        f"blocks or the pool size.",
        errors,
    )
    n_frames = 1 + seg_samples // int(feat["hop_length"])
    time_after = n_frames // (pool ** n_blocks)
    _require(
        time_after >= 1,
        f"{n_blocks} pooling stages reduce the {n_frames}-frame time axis below 1.",
        errors,
    )
    _require(0.0 <= arch["dropout"] < 1.0, "cnn.architecture.dropout must be in [0, 1)", errors)

    # ---- evaluation -------------------------------------------------------
    _require(
        cfg["evaluation"]["primary_aggregation"] in cfg["evaluation"]["aggregation"],
        "evaluation.primary_aggregation must be one of evaluation.aggregation",
        errors,
    )

    # ---- XAI --------------------------------------------------------------
    xseg = cfg["xai"]["segmentation"]
    _require(
        0 < xseg["min_hr_bpm"] < xseg["max_hr_bpm"] < 300,
        "xai.segmentation heart-rate bounds must satisfy 0 < min < max < 300 bpm",
        errors,
    )
    _require(
        0.0 < xseg["systole_fraction_prior"] < 1.0,
        "xai.segmentation.systole_fraction_prior must be in (0, 1)",
        errors,
    )
    _require(
        cfg["xai"]["alignment"]["primary_state"] in cfg["xai"]["alignment"]["states"],
        "xai.alignment.primary_state must appear in xai.alignment.states",
        errors,
    )

    if errors:
        raise ConfigValidationError(
            "Config validation failed:\n  - " + "\n  - ".join(errors)
        )


def derived_shapes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the tensor shapes implied by the config, without touching data.

    Printed at the start of every run so shape mismatches are caught in seconds
    rather than after a 25-minute feature-extraction pass.
    """
    sr = float(cfg["preprocessing"]["target_sr"])
    feat = cfg["features"]
    seg_samples = int(cfg["segmentation"]["segment_seconds"] * sr)
    n_frames = 1 + seg_samples // int(feat["hop_length"])   # librosa center=True

    n_mfcc = int(feat["mfcc"]["n_mfcc"])
    n_streams = 1 + int(feat["mfcc"]["use_delta"]) + int(feat["mfcc"]["use_delta2"])
    n_aggs = len(feat["mfcc"]["aggregations"])

    pwp = feat["pwp"]
    n_bands = pwp["n_perceptual_bands"] if pwp["perceptual_grouping"] else \
        int((pwp["band_high"] - pwp["band_low"]) / ((sr / 2) / 2 ** pwp["level"])) + 1

    return {
        "segment_samples": seg_samples,
        "n_frames": n_frames,
        "mfcc_dim": n_mfcc * n_streams * n_aggs,
        "logmel_shape": [int(feat["n_mels"]), n_frames],
        "pwp_n_bands": n_bands,
        "pwp_dim": n_bands * len(pwp["descriptors"]),
        "fft_bin_width_hz": sr / int(feat["n_fft"]),
        "usable_fft_bins_in_band": usable_fft_bins(
            sr, int(feat["n_fft"]), float(feat["fmin"]), float(feat["fmax"])
        ),
        "wp_subband_width_hz": (sr / 2) / 2 ** int(pwp["level"]),
    }
