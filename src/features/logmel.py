"""Log-Mel spectrogram: the 2-D time-frequency "image" fed to the CNN.

This is the only feature domain that preserves the time axis. That is the whole
point: the classical branch aggregates time away, the CNN branch keeps it, and
the comparison between them is one of the study's findings.

Output shape at default settings: ``(32 mel bands, 188 frames)`` for a 3 s
segment at 2 kHz with ``hop_length=32``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import BaseFeatureExtractor, safe_log, sanitize


class LogMelExtractor(BaseFeatureExtractor):
    """2-D log-Mel (or log-linear) spectrogram."""

    domain = "logmel"

    def __init__(self, sr: float, cfg: Dict):
        super().__init__(sr, cfg)
        feat = cfg["features"]
        logmel_cfg = feat.get("logmel", {})

        self.n_fft = int(feat["n_fft"])
        self.hop_length = int(feat["hop_length"])
        self.window = str(feat["window"])
        self.fmin = float(feat["fmin"])
        self.fmax = float(feat["fmax"])
        self.n_mels = int(feat["n_mels"])

        # "mel" (default) or "linear" - the latter drives ablation A2.
        self.scale = str(logmel_cfg.get("scale", "mel"))
        self.log_offset = float(logmel_cfg.get("log_offset", 1e-6))
        self.normalize_per_segment = bool(logmel_cfg.get("normalize_per_segment", False))

        seg_samples = int(round(cfg["segmentation"]["segment_seconds"] * self.sr))
        self.n_frames = 1 + seg_samples // self.hop_length   # librosa center=True
        self._filterbank: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _get_filterbank(self) -> np.ndarray:
        """Cache the filterbank - rebuilding it per segment is pure waste."""
        if self._filterbank is not None:
            return self._filterbank

        import librosa

        if self.scale == "mel":
            self._filterbank = librosa.filters.mel(
                sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax,
            )
        elif self.scale == "linear":
            # Triangular filters on a LINEAR frequency grid, otherwise
            # identical in construction to the mel bank - so the ablation
            # isolates the warping and nothing else.
            self._filterbank = self._linear_filterbank()
        else:
            raise ValueError(f"Unknown filterbank scale: {self.scale}")
        return self._filterbank

    def _linear_filterbank(self) -> np.ndarray:
        """Triangular filterbank with linearly spaced edges."""
        n_bins = 1 + self.n_fft // 2
        fft_freqs = np.linspace(0, self.sr / 2, n_bins)
        edges = np.linspace(self.fmin, self.fmax, self.n_mels + 2)

        bank = np.zeros((self.n_mels, n_bins), dtype=np.float32)
        for m in range(self.n_mels):
            left, centre, right = edges[m], edges[m + 1], edges[m + 2]
            rising = (fft_freqs - left) / max(centre - left, 1e-9)
            falling = (right - fft_freqs) / max(right - centre, 1e-9)
            bank[m] = np.maximum(0.0, np.minimum(rising, falling))
            # Slaney-style area normalisation, matching librosa's default norm
            area = right - left
            if area > 0:
                bank[m] *= 2.0 / area
        return bank

    # ------------------------------------------------------------------ #
    def transform(self, segment: np.ndarray) -> np.ndarray:
        import librosa

        spectrum = np.abs(
            librosa.stft(
                np.asarray(segment, dtype=np.float32),
                n_fft=self.n_fft, hop_length=self.hop_length,
                window=self.window, center=True,
            )
        ) ** 2.0                                   # power spectrogram

        banded = self._get_filterbank() @ spectrum  # (n_mels, n_frames)
        log_spec = safe_log(banded, self.log_offset).astype(np.float32)

        if self.normalize_per_segment:
            # Off by default: per-segment normalisation would remove absolute
            # level differences that the CNN can legitimately use, and it
            # breaks comparability between segments in the Grad-CAM analysis.
            log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-8)

        # Guard the frame count: librosa's centre padding can give n_frames+-1
        # depending on segment length parity. Downstream tensors are fixed-size,
        # so we pad or crop to the shape declared by output_shape.
        log_spec = self._fit_frames(log_spec)
        features, _ = sanitize(log_spec, "logmel")
        return features

    def _fit_frames(self, spec: np.ndarray) -> np.ndarray:
        current = spec.shape[1]
        if current == self.n_frames:
            return spec
        if current > self.n_frames:
            return spec[:, : self.n_frames]
        pad = np.full((spec.shape[0], self.n_frames - current),
                      spec.min(), dtype=spec.dtype)
        return np.concatenate([spec, pad], axis=1)

    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> List[str]:
        """Flattened (band, frame) names - used only for diagnostics.

        We never run SHAP on the flattened log-Mel: 6016 dimensions with
        KernelSHAP would be intractable and the result uninterpretable. The CNN
        is explained with Grad-CAM instead, which respects the 2-D structure.
        """
        if self._feature_names is None:
            freqs = self.band_frequencies()
            self._feature_names = [
                f"band{b}_{freqs[b]:.0f}Hz_t{t}"
                for b in range(self.n_mels) for t in range(self.n_frames)
            ]
        return self._feature_names

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.n_mels, self.n_frames)

    def band_frequencies(self) -> np.ndarray:
        """Centre frequency (Hz) of each band - the y-axis of every plot."""
        if self.scale == "linear":
            return np.linspace(self.fmin, self.fmax, self.n_mels + 2)[1:-1]
        import librosa
        return librosa.mel_frequencies(n_mels=self.n_mels + 2,
                                       fmin=self.fmin, fmax=self.fmax)[1:-1]

    def frame_times(self) -> np.ndarray:
        """Centre time (s) of each frame - the x-axis of every plot."""
        return np.arange(self.n_frames) * self.hop_length / self.sr

    def describe(self) -> Dict[str, object]:
        base = super().describe()
        base.update(
            scale=self.scale, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, n_frames=self.n_frames,
            fmin=self.fmin, fmax=self.fmax,
            window_ms=round(1000 * self.n_fft / self.sr, 1),
            hop_ms=round(1000 * self.hop_length / self.sr, 1),
            freq_resolution_hz=round(self.sr / self.n_fft, 2),
        )
        return base
