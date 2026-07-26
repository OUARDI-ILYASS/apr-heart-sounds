"""Lightweight cardiac-cycle segmenter, used ONLY to evaluate explanations.

What this is, precisely: an envelope-based S1/systole/S2/diastole labeller.
What this is NOT: the Springer et al. (2016) logistic-regression HSMM, which is
the reference segmenter for PhysioNet 2016 and is distributed as MATLAB code.

Algorithm
---------
1. Shannon-energy envelope, low-pass smoothed.
2. Autocorrelation of the envelope -> dominant cycle length (constrained to
   the configured heart-rate range).
3. Peak picking on the envelope with a minimum distance derived from the
   cycle length -> candidate S1/S2 locations.
4. Interval classification: within each cycle the two inter-peak gaps are the
   systolic and diastolic intervals. At normal heart rates diastole is the
   longer one; that physiological asymmetry is what lets us assign the labels
   without supervision.
5. State map over the whole segment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

# Integer codes for the state map. Order matters for plotting only.
STATE_CODES = {"S1": 0, "systole": 1, "S2": 2, "diastole": 3, "unknown": -1}
STATE_NAMES = {v: k for k, v in STATE_CODES.items()}


# --------------------------------------------------------------------------- #
# Envelopes
# --------------------------------------------------------------------------- #
def shannon_energy_envelope(x: np.ndarray, sr: float,
                            smoothing_ms: float = 20.0) -> np.ndarray:
    """Normalised average Shannon energy envelope.

    E = -x^2 * log(x^2). Compared with a plain squared-amplitude envelope this
    emphasises medium-amplitude components relative to the very largest ones,
    which in a PCG means S1 and S2 both survive even when one is much louder
    than the other - the usual failure mode of naive energy envelopes.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.max(np.abs(x)) + 1e-12)

    squared = x ** 2
    energy = -squared * np.log(squared + 1e-12)

    window = max(3, int(round(smoothing_ms * sr / 1000.0)))
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    smoothed = np.convolve(energy, kernel, mode="same")

    # Z-score then shift to non-negative: peak picking works on relative
    # prominence, and this keeps thresholds comparable across recordings.
    smoothed = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-12)
    return smoothed - smoothed.min()


def homomorphic_envelope(x: np.ndarray, sr: float,
                         cutoff_hz: float = 8.0) -> np.ndarray:
    """Homomorphic envelope: low-pass the log of the analytic-signal magnitude.

    This is the envelope Springer's segmenter uses as its first feature. It
    separates the slowly varying amplitude modulation (which carries the
    cardiac cycle) from the fast carrier.
    """
    x = np.asarray(x, dtype=np.float64)
    analytic = np.abs(sps.hilbert(x))
    log_envelope = np.log(analytic + 1e-12)

    sos = sps.butter(2, cutoff_hz / (0.5 * sr), btype="low", output="sos")
    smoothed = sps.sosfiltfilt(sos, log_envelope)

    envelope = np.exp(smoothed)
    return (envelope - envelope.min()) / (envelope.ptp() + 1e-12)


def hilbert_envelope(x: np.ndarray, sr: float, smoothing_ms: float = 20.0) -> np.ndarray:
    """Plain Hilbert magnitude envelope, smoothed."""
    envelope = np.abs(sps.hilbert(np.asarray(x, dtype=np.float64)))
    window = max(3, int(round(smoothing_ms * sr / 1000.0)))
    kernel = np.ones(window) / window
    envelope = np.convolve(envelope, kernel, mode="same")
    return (envelope - envelope.min()) / (envelope.ptp() + 1e-12)


ENVELOPE_FUNCTIONS = {
    "shannon": shannon_energy_envelope,
    "homomorphic": homomorphic_envelope,
    "hilbert": hilbert_envelope,
}


# --------------------------------------------------------------------------- #
# Cycle length
# --------------------------------------------------------------------------- #
def estimate_cycle_length(envelope: np.ndarray, sr: float,
                          min_hr_bpm: float = 40.0,
                          max_hr_bpm: float = 180.0) -> Tuple[float, float]:
    """Dominant cardiac cycle length (s) via envelope autocorrelation.

    Returns ``(cycle_seconds, confidence)`` where confidence is the normalised
    autocorrelation peak height - a direct measure of how periodic the segment
    actually is. Near zero means the recording is too noisy for the cycle to be
    recovered, and the caller should exclude it.
    """
    envelope = np.asarray(envelope, dtype=np.float64)
    envelope = envelope - envelope.mean()

    autocorr = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1:]
    if autocorr[0] <= 0:
        return float("nan"), 0.0
    autocorr = autocorr / autocorr[0]

    # Search only over physiologically possible lags.
    min_lag = int(round(60.0 / max_hr_bpm * sr))
    max_lag = int(round(60.0 / min_hr_bpm * sr))
    max_lag = min(max_lag, len(autocorr) - 1)
    if min_lag >= max_lag:
        return float("nan"), 0.0

    window = autocorr[min_lag:max_lag]
    peak_index = int(np.argmax(window))
    lag = min_lag + peak_index
    return float(lag / sr), float(np.clip(window[peak_index], 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Peak picking
# --------------------------------------------------------------------------- #
def find_sound_peaks(envelope: np.ndarray, sr: float, cycle_seconds: float
                     ) -> np.ndarray:
    """Locate candidate S1/S2 peaks in the envelope.

    Minimum spacing is set to 20% of the cycle. Systole is roughly 30-35% of
    the cycle, so 20% is comfortably below the true S1-S2 gap while still
    rejecting the double peaks that a split S2 produces.
    """
    envelope = np.asarray(envelope, dtype=np.float64)
    if not np.isfinite(cycle_seconds) or cycle_seconds <= 0:
        return np.array([], dtype=int)

    min_distance = max(1, int(round(0.20 * cycle_seconds * sr)))
    peaks, _ = sps.find_peaks(
        envelope,
        distance=min_distance,
        # Prominence relative to the envelope's own spread: an absolute
        # threshold would not transfer between recordings.
        prominence=0.3 * float(np.std(envelope)),
    )
    return peaks


def classify_intervals(peaks: np.ndarray, sr: float,
                       systole_fraction_prior: float = 0.35
                       ) -> Tuple[List[Dict], float]:
    """Decide which inter-peak gaps are systole and which are diastole.

    The physiological rule: at rest, systole occupies roughly a third of the
    cycle and diastole the remaining two thirds, so consecutive intervals
    alternate short-long-short-long. We determine the global phase (does the
    sequence start with a short or a long interval?) by comparing the mean
    length of the even-indexed intervals with the odd-indexed ones, then label
    accordingly. Deciding the phase globally rather than per interval makes the
    labelling robust to a single mis-detected peak.

    NOTE: this rule degrades at tachycardia (>120 bpm), where diastole shortens
    much faster than systole and the two intervals converge. The returned
    confidence encodes how well separated they were, and the alignment analysis
    excludes low-confidence segments.
    """
    if len(peaks) < 3:
        return [], 0.0

    intervals = np.diff(peaks) / sr
    even, odd = intervals[0::2], intervals[1::2]
    if even.size == 0 or odd.size == 0:
        return [], 0.0

    # The shorter mean interval is systole.
    if even.mean() < odd.mean():
        systole_positions, diastole_positions = set(range(0, len(intervals), 2)), set(range(1, len(intervals), 2))
        systole_lengths, diastole_lengths = even, odd
    else:
        systole_positions, diastole_positions = set(range(1, len(intervals), 2)), set(range(0, len(intervals), 2))
        systole_lengths, diastole_lengths = odd, even

    labelled: List[Dict] = []
    for i, length in enumerate(intervals):
        labelled.append({
            "start_sample": int(peaks[i]),
            "end_sample": int(peaks[i + 1]),
            "duration_s": float(length),
            "state": "systole" if i in systole_positions else "diastole",
        })

    # Confidence: how clearly separated are the two interval populations, and
    # how close is the observed systolic fraction to the physiological prior?
    separation = float(
        (diastole_lengths.mean() - systole_lengths.mean())
        / (intervals.std() + 1e-12)
    )
    observed_fraction = float(
        systole_lengths.mean() / (systole_lengths.mean() + diastole_lengths.mean() + 1e-12)
    )
    prior_agreement = float(1.0 - min(1.0, abs(observed_fraction - systole_fraction_prior) / 0.25))
    regularity = float(1.0 - min(1.0, intervals.std() / (intervals.mean() + 1e-12)))

    confidence = float(np.clip(
        0.4 * np.clip(separation, 0, 1) + 0.3 * prior_agreement + 0.3 * regularity,
        0.0, 1.0,
    ))
    return labelled, confidence


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def segment_cardiac_cycle(x: np.ndarray, sr: float, cfg: Dict
                          ) -> Dict[str, object]:
    """Produce a per-sample cardiac-state map for one segment.

    Returns a dict with the state array, the detected peaks, per-state time
    fractions, and a confidence score. The time fractions are essential: they
    are the null hypothesis against which attribution concentration is judged.
    A model attending uniformly over time scores exactly the time fraction of
    each state, so "40% of attribution in systole" is only meaningful once you
    know systole occupies 33% of the time.
    """
    seg_cfg = cfg["xai"]["segmentation"]
    envelope_fn = ENVELOPE_FUNCTIONS[str(seg_cfg.get("envelope", "shannon"))]

    envelope = envelope_fn(x, sr, float(seg_cfg.get("smoothing_ms", 20.0)))
    cycle_seconds, cycle_confidence = estimate_cycle_length(
        envelope, sr,
        min_hr_bpm=float(seg_cfg.get("min_hr_bpm", 40)),
        max_hr_bpm=float(seg_cfg.get("max_hr_bpm", 180)),
    )

    peaks = find_sound_peaks(envelope, sr, cycle_seconds)
    intervals, interval_confidence = classify_intervals(
        peaks, sr, float(seg_cfg.get("systole_fraction_prior", 0.35))
    )

    states = np.full(len(x), STATE_CODES["unknown"], dtype=np.int8)

    # Sound windows around each detected peak: S1 and S2 last ~100-120 ms and
    # ~80-100 ms respectively. We use a symmetric 60 ms half-width, which is a
    # deliberate simplification - the alignment metric is not sensitive to a
    # few milliseconds at the edges.
    half_width = int(round(0.06 * sr))
    for interval in intervals:
        start, end = interval["start_sample"], interval["end_sample"]
        code = STATE_CODES[interval["state"]]
        states[start:end] = code

    # The peak that opens a systolic interval is S1; the one that opens a
    # diastolic interval is S2.
    for interval in intervals:
        peak = interval["start_sample"]
        lo, hi = max(0, peak - half_width), min(len(x), peak + half_width)
        states[lo:hi] = STATE_CODES["S1"] if interval["state"] == "systole" else STATE_CODES["S2"]

    total_confidence = float(np.clip(0.5 * cycle_confidence + 0.5 * interval_confidence, 0, 1))

    fractions: Dict[str, float] = {}
    for name, code in STATE_CODES.items():
        fractions[name] = float(np.mean(states == code))

    heart_rate = 60.0 / cycle_seconds if np.isfinite(cycle_seconds) and cycle_seconds > 0 else float("nan")

    return {
        "states": states,
        "envelope": envelope,
        "peaks": peaks,
        "intervals": intervals,
        "cycle_seconds": cycle_seconds,
        "heart_rate_bpm": heart_rate,
        "cycle_confidence": cycle_confidence,
        "interval_confidence": interval_confidence,
        "confidence": total_confidence,
        "state_fractions": fractions,
        "n_cycles_detected": len([i for i in intervals if i["state"] == "systole"]),
        "usable": bool(total_confidence >= float(seg_cfg.get("confidence_threshold", 0.5))),
    }


def states_to_frames(states: np.ndarray, hop_length: int, n_frames: int
                     ) -> np.ndarray:
    """Downsample a per-sample state map onto the spectrogram's frame grid.

    Grad-CAM lives on the (band, frame) grid, so the cardiac states must be
    expressed there too. Each frame takes the *modal* state of the samples it
    covers, which is the right reduction for a categorical variable (averaging
    state codes would be meaningless).
    """
    frame_states = np.full(n_frames, STATE_CODES["unknown"], dtype=np.int8)
    for frame in range(n_frames):
        start = frame * hop_length
        end = min(start + hop_length, len(states))
        if start >= len(states):
            break
        window = states[start:end]
        window = window[window != STATE_CODES["unknown"]]
        if window.size:
            values, counts = np.unique(window, return_counts=True)
            frame_states[frame] = values[int(np.argmax(counts))]
    return frame_states


def batch_segment(segments: np.ndarray, sr: float, cfg: Dict,
                  progress: bool = False) -> List[Dict[str, object]]:
    """Segment a stack of audio windows, reporting how many were usable."""
    results = []
    for i, segment in enumerate(segments):
        try:
            results.append(segment_cardiac_cycle(segment, sr, cfg))
        except Exception as exc:
            results.append({
                "states": np.full(len(segment), STATE_CODES["unknown"], dtype=np.int8),
                "confidence": 0.0, "usable": False, "error": str(exc),
                "state_fractions": {k: 0.0 for k in STATE_CODES},
            })
        if progress and (i + 1) % 200 == 0:
            print(f"\r  segmenting: {i + 1}/{len(segments)}", end="", flush=True)
    if progress:
        print()
    return results


def segmentation_quality_report(results: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate segmenter diagnostics - reported as a limitation in the paper."""
    confidences = np.array([r.get("confidence", 0.0) for r in results])
    usable = np.array([bool(r.get("usable", False)) for r in results])
    heart_rates = np.array([r.get("heart_rate_bpm", np.nan) for r in results], dtype=float)
    valid_hr = heart_rates[np.isfinite(heart_rates)]

    return {
        "n_segments": len(results),
        "n_usable": int(usable.sum()),
        "usable_fraction": float(usable.mean()) if len(results) else 0.0,
        "mean_confidence": float(confidences.mean()) if len(results) else 0.0,
        "median_confidence": float(np.median(confidences)) if len(results) else 0.0,
        "mean_heart_rate_bpm": float(valid_hr.mean()) if valid_hr.size else float("nan"),
        "heart_rate_range": [float(valid_hr.min()), float(valid_hr.max())]
                            if valid_hr.size else [float("nan"), float("nan")],
        "n_errors": int(sum(1 for r in results if "error" in r)),
    }
