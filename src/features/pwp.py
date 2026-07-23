"""Perceptual Wavelet Packet (PWP) features.

Motivation. MFCC and log-Mel both sit on top of the STFT, which imposes one
fixed time-frequency resolution on the whole signal. A PCG is the worst case
for that assumption: S1 and S2 are short, sharp transients (10-50 ms), while a
murmur is a sustained, noise-like texture lasting a few hundred milliseconds.
The wavelet packet transform tiles the time-frequency plane adaptively, giving
short bases at high frequency and long bases at low frequency, which is the
classical argument for using it on heart sounds (Safara et al. 2013).

Pipeline per segment::

    full wavelet packet decomposition to `level`
      -> take the terminal nodes IN FREQUENCY ORDER
      -> keep only nodes overlapping the diagnostic band [25, 400] Hz
      -> merge them into perceptually spaced bands (mel-spaced edges)
      -> 7 descriptors per band
      -> one fixed-length vector

Default dimensionality: 12 perceptual bands x 7 descriptors = **84**.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps_stats

from .base import BaseFeatureExtractor, safe_log, sanitize

# Descriptors computed per perceptual band. Each is a scalar summary of the
# wavelet coefficients falling in that band.
DESCRIPTOR_DOC = {
    "log_energy": "log of total squared coefficient energy in the band",
    "rel_energy": "band energy / total energy across retained bands",
    "shannon_entropy": "Coifman-Wickerhauser entropy of the normalised energy "
                       "distribution - low for tonal/impulsive content, high "
                       "for noise-like content such as a murmur",
    "std": "coefficient standard deviation",
    "skew": "coefficient skewness",
    "kurtosis": "coefficient kurtosis - high for impulsive events (S1/S2 "
                "clicks), low for sustained noise (murmur)",
    "max_abs": "peak absolute coefficient",
}


class PWPExtractor(BaseFeatureExtractor):
    """Perceptual wavelet packet descriptors."""

    domain = "pwp"

    def __init__(self, sr: float, cfg: Dict):
        super().__init__(sr, cfg)
        pwp_cfg = cfg["features"]["pwp"]

        self.wavelet = str(pwp_cfg["wavelet"])
        self.level = int(pwp_cfg["level"])
        self.node_order = str(pwp_cfg.get("node_order", "freq"))
        self.band_low = float(pwp_cfg["band_low"])
        self.band_high = float(pwp_cfg["band_high"])
        self.perceptual_grouping = bool(pwp_cfg.get("perceptual_grouping", True))
        self.n_perceptual_bands = int(pwp_cfg.get("n_perceptual_bands", 12))
        self.best_basis = bool(pwp_cfg.get("best_basis", False))
        self.descriptors: List[str] = list(pwp_cfg["descriptors"])

        if self.node_order != "freq":
            raise ValueError(
                "PWP node_order must be 'freq'. Natural (Paley) ordering is not "
                "monotonic in frequency and silently scrambles the spectrum."
            )

        # --- geometry of the decomposition, computed once ------------------
        self.n_nodes = 2 ** self.level
        self.node_width_hz = (self.sr / 2.0) / self.n_nodes
        node_lo = np.arange(self.n_nodes) * self.node_width_hz
        node_hi = node_lo + self.node_width_hz
        self.node_edges = np.stack([node_lo, node_hi], axis=1)

        # Nodes that overlap the diagnostic band at all.
        self.kept_nodes = np.where(
            (node_hi > self.band_low) & (node_lo < self.band_high)
        )[0]
        if self.kept_nodes.size == 0:
            raise ValueError(
                f"No wavelet packet node at level {self.level} overlaps "
                f"[{self.band_low}, {self.band_high}] Hz."
            )

        self.band_assignment, self.band_edges = self._build_perceptual_bands()
        self.n_bands = len(self.band_edges) - 1
        self._dim = self.n_bands * len(self.descriptors)

    # ------------------------------------------------------------------ #
    def _build_perceptual_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """Assign each retained WP node to a perceptual band.

        Returns ``(assignment, edges)`` where ``assignment[i]`` is the band
        index of ``self.kept_nodes[i]``.
        """
        if not self.perceptual_grouping:
            # One band per node - a plain (non-perceptual) wavelet packet
            # feature set. Kept as a code path so the "perceptual" part is
            # itself ablatable.
            edges = np.append(
                self.node_edges[self.kept_nodes, 0],
                self.node_edges[self.kept_nodes[-1], 1],
            )
            return np.arange(len(self.kept_nodes)), edges

        edges = self._mel_spaced_edges(self.band_low, self.band_high,
                                       self.n_perceptual_bands)
        centres = self.node_edges[self.kept_nodes].mean(axis=1)
        # np.digitize returns 1-based bin indices; clip so nodes below the
        # first edge or above the last land in the extreme bands rather than
        # being silently dropped.
        assignment = np.clip(np.digitize(centres, edges) - 1,
                             0, self.n_perceptual_bands - 1)
        return assignment, edges

    @staticmethod
    def _mel_spaced_edges(fmin: float, fmax: float, n_bands: int) -> np.ndarray:
        """Band edges equally spaced on the mel scale (O'Shaughnessy formula)."""
        def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
            return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)

        def mel_to_hz(m: np.ndarray) -> np.ndarray:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_bands + 1)
        return mel_to_hz(mels)

    # ------------------------------------------------------------------ #
    def _node_coefficients(self, segment: np.ndarray) -> List[np.ndarray]:
        """Wavelet packet coefficients of the retained nodes, frequency-ordered."""
        import pywt

        packet = pywt.WaveletPacket(
            data=np.asarray(segment, dtype=np.float64),
            wavelet=self.wavelet,
            mode="symmetric",     # symmetric padding avoids the artificial
                                  # discontinuity that zero padding creates at
                                  # the segment boundary
            maxlevel=self.level,
        )
        # order='freq' is the critical argument - see the module docstring.
        nodes = packet.get_level(self.level, order="freq")
        return [np.asarray(nodes[i].data, dtype=np.float64) for i in self.kept_nodes]

    # ------------------------------------------------------------------ #
    def transform(self, segment: np.ndarray) -> np.ndarray:
        node_coeffs = self._node_coefficients(segment)

        # Merge node coefficients into perceptual bands.
        band_coeffs: List[np.ndarray] = []
        for band in range(self.n_bands):
            members = [c for c, b in zip(node_coeffs, self.band_assignment) if b == band]
            band_coeffs.append(
                np.concatenate(members) if members else np.zeros(1, dtype=np.float64)
            )

        energies = np.array([float(np.sum(c ** 2)) for c in band_coeffs])
        total_energy = float(energies.sum())

        rows = [self._descriptors_for_band(c, e, total_energy)
                for c, e in zip(band_coeffs, energies)]

        # Band-major ordering: all descriptors of band 0, then band 1, ...
        # Keeps a band's descriptors adjacent in SHAP plots.
        features, _ = sanitize(np.concatenate(rows), "pwp")
        return features

    def _descriptors_for_band(self, coeffs: np.ndarray, energy: float,
                              total_energy: float) -> np.ndarray:
        values: List[float] = []
        n = max(1, coeffs.size)

        for name in self.descriptors:
            if name == "log_energy":
                values.append(float(safe_log(energy, 1e-12)))
            elif name == "rel_energy":
                values.append(float(energy / total_energy) if total_energy > 0 else 0.0)
            elif name == "shannon_entropy":
                values.append(self._shannon_entropy(coeffs, energy))
            elif name == "std":
                values.append(float(np.std(coeffs)))
            elif name == "skew":
                # Degenerate (constant / all-zero) bands give NaN; 0 is the
                # correct limiting value and keeps the matrix finite.
                values.append(0.0 if np.std(coeffs) < 1e-12
                              else float(sps_stats.skew(coeffs, bias=False)))
            elif name == "kurtosis":
                values.append(0.0 if np.std(coeffs) < 1e-12
                              else float(sps_stats.kurtosis(coeffs, bias=False)))
            elif name == "max_abs":
                values.append(float(np.max(np.abs(coeffs))))
            elif name == "mean_abs":
                values.append(float(np.mean(np.abs(coeffs))))
            elif name == "zero_crossing_rate":
                values.append(float(np.mean(np.diff(np.signbit(coeffs)) != 0)) if n > 1 else 0.0)
            else:
                raise ValueError(f"Unknown PWP descriptor: {name}")

        return np.nan_to_num(np.asarray(values, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _shannon_entropy(coeffs: np.ndarray, energy: float) -> float:
        """Coifman-Wickerhauser entropy of the energy distribution.

        Interpretation for PCG: energy concentrated in a few coefficients
        (a click, a transient) gives low entropy; energy spread evenly across
        coefficients (noise-like murmur turbulence) gives high entropy. This is
        the descriptor we expect to carry most of the murmur signal.
        """
        if energy <= 0:
            return 0.0
        p = (coeffs ** 2) / energy
        p = p[p > 0]
        return float(-np.sum(p * np.log(p))) if p.size else 0.0

    # ------------------------------------------------------------------ #
    def best_basis_entropy(self, segment: np.ndarray) -> Dict[str, object]:
        """Coifman-Wickerhauser best-basis cost, reported but not used by default.

        Walks the packet tree bottom-up and, at each parent, compares the
        parent's entropy against the sum of its children's. The resulting cost
        tells you whether an adaptive basis would be materially better than the
        fixed full decomposition.

        We do not use an adaptive basis for the features themselves: the chosen
        basis would differ from segment to segment, so feature index *k* would
        mean different things in different samples, which breaks both the
        classifier's input contract and SHAP's interpretation. Reporting the
        cost lets us say in the paper that we considered it and why we did not
        adopt it.
        """
        import pywt

        packet = pywt.WaveletPacket(
            data=np.asarray(segment, dtype=np.float64),
            wavelet=self.wavelet, mode="symmetric", maxlevel=self.level,
        )
        costs: Dict[int, float] = {}
        for level in range(self.level, 0, -1):
            nodes = packet.get_level(level, order="natural")
            total = 0.0
            for node in nodes:
                data = np.asarray(node.data, dtype=np.float64)
                energy = float(np.sum(data ** 2))
                total += self._shannon_entropy(data, energy)
            costs[level] = total
        best_level = int(min(costs, key=lambda k: costs[k]))
        return {"cost_per_level": costs, "best_level": best_level,
                "used_level": self.level}

    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            names: List[str] = []
            for band in range(self.n_bands):
                lo, hi = self.band_edges[band], self.band_edges[band + 1]
                tag = f"pwp_b{band}_{lo:.0f}-{hi:.0f}Hz"
                names.extend(f"{tag}_{d}" for d in self.descriptors)
            self._feature_names = names
        return self._feature_names

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self._dim,)

    def band_centre_frequencies(self) -> np.ndarray:
        """Centre frequency of each perceptual band.

        Unlike MFCC, PWP features map *directly* onto frequency - no DCT sits
        in between. This is why the PWP SHAP analysis is the more trustworthy
        of the two frequency attributions, and we say so in the paper.
        """
        return 0.5 * (self.band_edges[:-1] + self.band_edges[1:])

    def describe(self) -> Dict[str, object]:
        base = super().describe()
        base.update(
            wavelet=self.wavelet, level=self.level, node_order=self.node_order,
            n_nodes_total=self.n_nodes,
            n_nodes_kept=int(self.kept_nodes.size),
            node_width_hz=round(self.node_width_hz, 3),
            n_perceptual_bands=self.n_bands,
            perceptual_grouping=self.perceptual_grouping,
            band_edges_hz=[round(float(e), 1) for e in self.band_edges],
            descriptors=self.descriptors,
            nodes_per_band=np.bincount(
                self.band_assignment, minlength=self.n_bands
            ).tolist(),
        )
        return base
