"""PCG preprocessing: resampling, band-limiting, spike removal, normalisation.

The order of operations matters and is not arbitrary:

    resample -> bandpass -> spike removal -> normalise

* Resample first so all subsequent parameters are expressed in one sample rate.
* Bandpass before spike removal so the spike detector is not triggered by
  low-frequency baseline wander.
* Normalise last so the amplitude statistics reflect the signal we actually
  feed the model, not the raw recording.

PROFESSOR Q: "Why zero-phase filtering?"
A: A causal IIR filter introduces a frequency-dependent group delay. Our entire
   explainability argument rests on *when* in the cardiac cycle the model
   attends. If the filter shifted 200 Hz murmur energy by a few milliseconds
   relative to the 40 Hz S1 fundamental, the attribution maps would be
   comparing evidence that had been silently misaligned in time. filtfilt runs
   the filter forwards then backwards, cancelling the phase response exactly
   (at the cost of doubling the effective filter order, which we account for).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sps


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def resample_signal(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling. No-op when rates already match (the common case)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    from math import gcd

    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out // g), int(sr_in // g)
    return sps.resample_poly(x, up, down).astype(np.float32)


# --------------------------------------------------------------------------- #
# Band-limiting
# --------------------------------------------------------------------------- #
def butter_bandpass(low: float, high: float, sr: float, order: int = 4):
    """Design a Butterworth bandpass as second-order sections.

    SOS rather than transfer-function coefficients: a 4th-order bandpass is an
    8th-order filter, and direct-form ``b, a`` coefficients become numerically
    unstable at that order, especially with the very low normalised cutoffs we
    use here (25 Hz at sr=2000 is 0.025 of Nyquist). SOS is the numerically
    robust factorisation.
    """
    nyquist = 0.5 * sr
    low_n, high_n = low / nyquist, min(high / nyquist, 0.999)
    if not 0 < low_n < high_n < 1:
        raise ValueError(f"Invalid band [{low}, {high}] Hz for sr={sr} Hz")
    return sps.butter(order, [low_n, high_n], btype="band", output="sos")


def bandpass_filter(x: np.ndarray, low: float, high: float, sr: float,
                    order: int = 4, zero_phase: bool = True) -> np.ndarray:
    sos = butter_bandpass(low, high, sr, order)
    if zero_phase:
        # padlen guards against short segments; sosfiltfilt needs 3*(2*order)
        padlen = min(len(x) - 1, 3 * (2 * order + 1))
        return sps.sosfiltfilt(sos, x, padlen=max(0, padlen)).astype(np.float32)
    return sps.sosfilt(sos, x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Spike removal (Schmidt et al. 2010)
# --------------------------------------------------------------------------- #
def remove_spikes(x: np.ndarray, sr: float, window_ms: float = 500.0,
                  threshold_factor: float = 3.0,
                  max_iterations: int = 10) -> Tuple[np.ndarray, int]:
    """Suppress friction/handling transients.

    Algorithm (following the widely used Schmidt implementation):

    1. Split the signal into ``window_ms`` frames.
    2. Take each frame's maximum absolute amplitude (MAA).
    3. While any MAA exceeds ``threshold_factor`` x median(MAA):
         a. locate the offending frame,
         b. find the spike peak inside it,
         c. find the zero crossings bracketing the peak,
         d. replace that span with zeros.

    Returns the cleaned signal and the number of spikes removed.

    PROFESSOR Q: "Doesn't zeroing a span destroy information?"
    A: The removed spans are typically 10-50 ms of clipped, non-cardiac
       transient - the diaphragm being brushed. Leaving them in is worse: they
       are the loudest events in the recording, so they dominate every
       energy-based descriptor and the per-recording normalisation constant.
       We count removals and report them, so the cost is visible rather than
       hidden.
    """
    x = x.astype(np.float32, copy=True)
    window = max(1, int(round(window_ms * sr / 1000.0)))
    n_removed = 0

    for _ in range(max_iterations):
        n_windows = len(x) // window
        if n_windows < 3:
            break
        frames = x[: n_windows * window].reshape(n_windows, window)
        maa = np.abs(frames).max(axis=1)
        median_maa = float(np.median(maa))
        if median_maa <= 0 or not np.any(maa > threshold_factor * median_maa):
            break

        idx = int(np.argmax(maa))
        frame = frames[idx]
        peak = int(np.argmax(np.abs(frame)))

        sign_changes = np.where(np.diff(np.signbit(frame)))[0]
        before = sign_changes[sign_changes < peak]
        after = sign_changes[sign_changes > peak]
        start = int(before[-1]) if before.size else 0
        end = int(after[0]) if after.size else window - 1

        x[idx * window + start: idx * window + end + 1] = 0.0
        n_removed += 1

    return x, n_removed


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize_signal(x: np.ndarray, method: str = "zscore",
                     eps: float = 1e-8) -> np.ndarray:
    """Per-recording amplitude normalisation.

    PROFESSOR Q: "Why normalise per recording and not globally?"
    A: Recording gain in this dataset is a site artefact, not a physiological
       variable. Different clinics used different digital stethoscopes with
       different preamp gains. A global normalisation would preserve those
       gain differences, and since gain correlates with site, and site
       correlates with class prevalence, the model could exploit loudness as a
       proxy for the label. Per-recording normalisation removes that channel
       entirely. The cost is that we discard absolute intensity - but murmur
       *grade* is judged relative to S1/S2 in the same recording anyway, so
       the clinically meaningful information is relative, not absolute.
    """
    x = np.asarray(x, dtype=np.float32)
    if method == "none":
        return x
    if method == "zscore":
        return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)
    if method == "peak":
        return (x / (np.abs(x).max() + eps)).astype(np.float32)
    if method == "rms":
        return (x / (np.sqrt(np.mean(x ** 2)) + eps)).astype(np.float32)
    raise ValueError(f"Unknown normalisation method: {method}")


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess_recording(x: np.ndarray, sr_in: int,
                         cfg: Dict) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the complete preprocessing chain to one recording.

    Returns the processed signal and a metadata dict recorded in the phase
    summary (so that e.g. an unusually high spike count is visible later).
    """
    meta: Dict[str, object] = {"sr_in": int(sr_in), "n_samples_in": int(len(x))}

    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size == 0:
        raise ValueError("Empty recording")

    # Remove DC before anything else - a large offset upsets the filter design.
    x = x - float(np.mean(x))

    target_sr = int(cfg["target_sr"])
    x = resample_signal(x, sr_in, target_sr)
    meta["sr_out"] = target_sr
    meta["resampled"] = sr_in != target_sr

    x = bandpass_filter(
        x, low=float(cfg["bandpass_low"]), high=float(cfg["bandpass_high"]),
        sr=target_sr, order=int(cfg["filter_order"]),
        zero_phase=bool(cfg.get("zero_phase", True)),
    )

    if cfg.get("spike_removal", True):
        x, n_spikes = remove_spikes(
            x, target_sr,
            window_ms=float(cfg.get("spike_window_ms", 500)),
            threshold_factor=float(cfg.get("spike_threshold_factor", 3.0)),
        )
        meta["n_spikes_removed"] = int(n_spikes)
    else:
        meta["n_spikes_removed"] = 0

    x = normalize_signal(x, method=str(cfg.get("normalize", "zscore")))

    meta["n_samples_out"] = int(len(x))
    meta["duration_s"] = round(len(x) / target_sr, 3)
    meta["rms"] = float(np.sqrt(np.mean(x ** 2)))
    meta["is_finite"] = bool(np.all(np.isfinite(x)))
    return x, meta


def quality_flags(x: np.ndarray, sr: float,
                  min_duration_s: float = 3.0) -> Dict[str, bool]:
    """Cheap quality checks used to exclude unusable recordings.

    Deliberately conservative: we only drop recordings that are *structurally*
    unusable (too short, silent, constant, non-finite). Aggressive quality
    filtering would improve headline numbers while quietly removing the hard
    cases, which is a form of cherry-picking. The exclusion count is reported.
    """
    duration = len(x) / sr
    return {
        "too_short": bool(duration < min_duration_s),
        "silent": bool(np.allclose(x, 0.0)),
        "constant": bool(float(np.std(x)) < 1e-8),
        "non_finite": bool(not np.all(np.isfinite(x))),
        "clipped": bool(np.mean(np.abs(x) >= 0.999 * np.abs(x).max()) > 0.01)
                   if np.abs(x).max() > 0 else False,
    }
