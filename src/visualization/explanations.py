"""Explanation figures: SHAP summaries, Grad-CAM overlays, alignment plots.

The Grad-CAM overlay figure is the paper's centrepiece and it is built to be
*checkable* rather than merely attractive. Each panel shows three aligned rows:
the waveform with the detected cardiac states shaded, the log-Mel spectrogram
with the attribution contour on top, and the temporal attribution profile with
the systolic windows marked. A reader can therefore verify the alignment claim
by eye against the same segmentation the numbers were computed from, instead of
being asked to trust a lone heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .style import ieee_figure, save_figure, figure_size, apply_ieee_style
from ..xai.segmenter import STATE_CODES

# Shading colours for the cardiac states. S1/S2 are darker because they are
# short; systole/diastole are pale so the attribution curve stays readable.
STATE_COLORS = {
    "S1": "#4A4A4A", "systole": "#F4A261",
    "S2": "#4A4A4A", "diastole": "#A8DADC",
}


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def plot_shap_summary(result: Dict, path: str | Path, top_k: int = 20,
                      cfg: Optional[Dict] = None, title: str = ""):
    """Beeswarm-style SHAP summary using shap's own plotting, resized for IEEE."""
    import shap

    apply_ieee_style(cfg)
    fig = plt.figure(figsize=figure_size(cfg, columns=1, height=3.2))
    shap.summary_plot(
        result["shap_values"], result["X"],
        feature_names=result["feature_names"],
        max_display=top_k, show=False, plot_size=None,
    )
    if title:
        plt.title(title, fontsize=8)
    plt.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)


def plot_shap_bar(rows: List[Dict], path: str | Path,
                  cfg: Optional[Dict] = None, title: str = ""):
    """Horizontal bar chart of mean |SHAP|, coloured by effect direction.

    Colour encodes the sign of the mean SHAP value: red means the feature
    pushes toward Abnormal on average, blue toward Normal. Mean |SHAP| alone
    tells you a feature matters but not which way it pushes, which is usually
    the more interesting half of the answer.
    """
    with ieee_figure(cfg, columns=1, height=3.2) as (fig, ax):
        names = [r["feature"] for r in rows][::-1]
        values = [r["mean_abs_shap"] for r in rows][::-1]
        directions = [r.get("mean_shap", 0) for r in rows][::-1]
        colors = ["#D7263D" if d > 0 else "#2E86AB" for d in directions]

        ax.barh(np.arange(len(names)), values, color=colors)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=5.5)
        ax.set_xlabel("Mean |SHAP value|")
        if title:
            ax.set_title(title)
        ax.legend(handles=[
            Patch(color="#D7263D", label="pushes toward Abnormal"),
            Patch(color="#2E86AB", label="pushes toward Normal"),
        ], fontsize=6, loc="lower right")
    return save_figure(fig, path, cfg)


def plot_frequency_attribution(profiles: Dict[str, Dict], path: str | Path,
                               cfg: Optional[Dict] = None):
    """Attribution against frequency for every model, on one axis.

    The cross-model agreement figure. Curves that peak in the same region
    despite coming from different models, different feature domains and
    different SHAP algorithms are the strongest evidence the XAI section can
    offer. We shade the 100-300 Hz region where systolic murmur energy is
    clinically expected, so agreement or disagreement with prior knowledge is
    immediately visible.
    """
    with ieee_figure(cfg, columns=1, height=2.6) as (fig, ax):
        ax.axvspan(100, 300, color="#F4A261", alpha=0.18,
                   label="expected murmur band")
        for name, profile in profiles.items():
            frequencies = np.asarray(profile["band_frequencies_hz"])
            attribution = np.asarray(profile["attribution"])
            style = "--" if "approx" in str(profile.get("method", "")).lower() \
                or "DCT" in str(profile.get("method", "")) else "-"
            ax.plot(frequencies, attribution, style, marker="o", ms=2, label=name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalised attribution")
        ax.set_xlim(20, 410)
        ax.legend(fontsize=6)
    return save_figure(fig, path, cfg)


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def plot_gradcam_example(segment: np.ndarray, logmel: np.ndarray, cam: np.ndarray,
                         segmentation: Dict, sr: float, hop_length: int,
                         path: str | Path, cfg: Optional[Dict] = None,
                         title: str = "", band_frequencies: Optional[np.ndarray] = None):
    """Three-row diagnostic panel for a single segment."""
    apply_ieee_style(cfg)
    fig, axes = plt.subplots(3, 1, figsize=figure_size(cfg, columns=1, height=4.2),
                             sharex=True, height_ratios=[1, 1.6, 1])

    times = np.arange(len(segment)) / sr
    frame_times = np.arange(logmel.shape[1]) * hop_length / sr

    # --- row 1: waveform with cardiac states shaded ----------------------
    axes[0].plot(times, segment, lw=0.35, color="#333333")
    states = segmentation.get("states")
    if states is not None:
        _shade_states(axes[0], np.asarray(states), sr)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title, fontsize=7)
    axes[0].grid(False)

    # --- row 2: log-Mel with the attribution contoured on top ------------
    extent = [frame_times[0], frame_times[-1],
              (band_frequencies[0] if band_frequencies is not None else 0),
              (band_frequencies[-1] if band_frequencies is not None else logmel.shape[0])]
    axes[1].imshow(logmel, aspect="auto", origin="lower", cmap="viridis",
                   extent=extent)
    # Contours rather than an alpha-blended overlay: blending makes it
    # impossible to read the underlying spectrogram, and the reader needs to
    # see what the model was looking at, not just where.
    cam_resized = _match_shape(cam, logmel.shape)
    axes[1].contour(np.linspace(extent[0], extent[1], cam_resized.shape[1]),
                    np.linspace(extent[2], extent[3], cam_resized.shape[0]),
                    cam_resized, levels=4, colors="white", linewidths=0.6, alpha=0.85)
    axes[1].set_ylabel("Frequency (Hz)" if band_frequencies is not None else "Mel band")
    axes[1].grid(False)

    # --- row 3: temporal attribution profile ------------------------------
    profile = np.asarray(cam).sum(axis=0)
    profile = profile / (profile.sum() + 1e-12)
    axes[2].fill_between(frame_times[:len(profile)], profile, color="#D7263D", alpha=0.6)
    if states is not None:
        _shade_states(axes[2], np.asarray(states), sr)
    axes[2].axhline(1.0 / len(profile), color="k", ls=":", lw=0.6)
    axes[2].annotate("uniform", xy=(frame_times[-1], 1.0 / len(profile)),
                     fontsize=5.5, ha="right", va="bottom")
    axes[2].set_ylabel("Attribution")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(False)

    handles = [Patch(color=STATE_COLORS[s], alpha=0.35, label=s)
               for s in ["S1", "systole", "S2", "diastole"]]
    axes[0].legend(handles=handles, fontsize=5.5, ncol=4, loc="upper right")

    fig.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)


def _shade_states(ax, states: np.ndarray, sr: float) -> None:
    """Shade contiguous runs of each cardiac state behind the plotted curve."""
    for name, code in STATE_CODES.items():
        if name == "unknown":
            continue
        mask = states == code
        if not mask.any():
            continue
        # Find run boundaries by differencing the boolean mask.
        edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for start, end in zip(starts, ends):
            ax.axvspan(start / sr, end / sr, color=STATE_COLORS[name],
                       alpha=0.28, lw=0)


def _match_shape(cam: np.ndarray, shape: tuple) -> np.ndarray:
    """Nearest-neighbour resize a CAM onto the spectrogram grid."""
    cam = np.asarray(cam, dtype=float)
    if cam.shape == tuple(shape):
        return cam
    rows = np.linspace(0, cam.shape[0] - 1, shape[0]).astype(int)
    cols = np.linspace(0, cam.shape[1] - 1, shape[1]).astype(int)
    return cam[np.ix_(rows, cols)]


def plot_average_cams(averages: Dict[str, np.ndarray], path: str | Path,
                      band_frequencies: Optional[np.ndarray] = None,
                      frame_times: Optional[np.ndarray] = None,
                      cfg: Optional[Dict] = None):
    """Mean attribution maps per outcome category (TP/TN/FP/FN).

    Averaging over hundreds of segments is what turns an anecdote into a
    population-level statement. The counts are printed in each title so the
    reader can discount panels built from a handful of examples.
    """
    apply_ieee_style(cfg)
    categories = ["true_positive", "true_negative", "false_positive", "false_negative"]
    categories = [c for c in categories if c in averages]

    fig, axes = plt.subplots(1, len(categories),
                             figsize=figure_size(cfg, columns=2, height=2.0),
                             squeeze=False)
    vmax = max(float(np.asarray(averages[c]).max()) for c in categories) or 1.0

    for index, category in enumerate(categories):
        ax = axes[0][index]
        cam = np.asarray(averages[category])
        extent = None
        if band_frequencies is not None and frame_times is not None:
            extent = [frame_times[0], frame_times[-1],
                      band_frequencies[0], band_frequencies[-1]]
        ax.imshow(cam, aspect="auto", origin="lower", cmap="magma",
                  vmin=0, vmax=vmax, extent=extent)
        count = averages.get(f"{category}_count", "?")
        ax.set_title(f"{category.replace('_', ' ')} (n={count})", fontsize=6.5)
        ax.set_xlabel("Time (s)", fontsize=6)
        if index == 0:
            ax.set_ylabel("Frequency (Hz)" if band_frequencies is not None else "Band",
                          fontsize=6)
        ax.grid(False)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def plot_alignment(alignment: Dict, path: str | Path, cfg: Optional[Dict] = None,
                   states: Optional[List[str]] = None, title: str = ""):
    """Attribution mass against time budget, per cardiac state.

    Paired bars: for each state, the fraction of attribution mass next to the
    fraction of time that state occupies. Equal heights mean "no preference".
    This framing is the whole point - a bare "38% of attention was in systole"
    is uninterpretable until you know systole is 33% of the recording.
    """
    states = states or ["S1", "systole", "S2", "diastole"]

    with ieee_figure(cfg, columns=1, height=2.4) as (fig, ax):
        positions = np.arange(len(states))
        width = 0.38
        mass = [alignment.get(f"mean_mass_{s}", 0) for s in states]
        time = [alignment.get(f"mean_time_{s}", 0) for s in states]

        ax.bar(positions - width / 2, mass, width, label="attribution mass",
               color="#D7263D")
        ax.bar(positions + width / 2, time, width, label="time fraction",
               color="#B0B0B0")

        for index, (m, t) in enumerate(zip(mass, time)):
            if t > 1e-6:
                ax.annotate(f"E={m / t:.2f}", xy=(index, max(m, t) + 0.015),
                            ha="center", fontsize=6)

        ax.set_xticks(positions)
        ax.set_xticklabels(states)
        ax.set_ylabel("Fraction")
        ax.set_title(title or "Attribution mass vs. time budget", fontsize=7)
        ax.legend(fontsize=6)
    return save_figure(fig, path, cfg)


def plot_alignment_comparison(table: List[Dict], path: str | Path,
                              state: str = "systole", cfg: Optional[Dict] = None):
    """Enrichment in one state, compared across models, with the null at 1.0."""
    with ieee_figure(cfg, columns=1, height=2.4) as (fig, ax):
        names = [r["model"] for r in table]
        values = [r.get(f"E_{state}", np.nan) for r in table]
        colors = ["#D7263D" if (v == v and v > 1.0) else "#999999" for v in values]

        ax.barh(np.arange(len(names)), values, color=colors)
        ax.axvline(1.0, color="k", ls="--", lw=0.8)
        ax.annotate("no preference", xy=(1.0, len(names) - 0.4), fontsize=6,
                    rotation=90, va="top", ha="right")
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(f"Enrichment in {state} (mass / time)")
        ax.invert_yaxis()
    return save_figure(fig, path, cfg)


def plot_segmentation_example(segment: np.ndarray, segmentation: Dict,
                              sr: float, path: str | Path,
                              cfg: Optional[Dict] = None):
    """Waveform, envelope, detected peaks and state assignment.

    Included in the supplementary material so the reader can judge the
    segmenter's quality directly. Since we substituted a simplified segmenter
    for the reference Springer HSMM, showing its output is not optional - it is
    what makes the substitution auditable.
    """
    apply_ieee_style(cfg)
    fig, axes = plt.subplots(2, 1, figsize=figure_size(cfg, columns=1, height=3.0),
                             sharex=True)
    times = np.arange(len(segment)) / sr

    axes[0].plot(times, segment, lw=0.35, color="#333333")
    _shade_states(axes[0], np.asarray(segmentation["states"]), sr)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(False)

    envelope = np.asarray(segmentation["envelope"])
    axes[1].plot(np.arange(len(envelope)) / sr, envelope, lw=0.7, color="#2E86AB")
    peaks = np.asarray(segmentation.get("peaks", []), dtype=int)
    if peaks.size:
        axes[1].plot(peaks / sr, envelope[peaks], "rv", ms=3, label="detected peaks")
        axes[1].legend(fontsize=6)
    axes[1].set_ylabel("Envelope")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(False)

    heart_rate = segmentation.get("heart_rate_bpm", float("nan"))
    confidence = segmentation.get("confidence", 0.0)
    axes[0].set_title(f"HR ≈ {heart_rate:.0f} bpm, segmenter confidence {confidence:.2f}",
                      fontsize=7)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, path, cfg)
