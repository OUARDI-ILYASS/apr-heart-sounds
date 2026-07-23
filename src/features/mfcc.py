"""MFCC features: 13 cepstral coefficients + velocity + acceleration, aggregated.

Pipeline per segment::

    STFT -> mel filterbank -> log -> DCT-II -> 13 coefficients
         -> [+ delta, + delta-delta]
         -> statistical aggregation over time
         -> one fixed-length vector

Default dimensionality: 13 coefficients x 3 streams x 6 statistics = **234**.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import BaseFeatureExtractor, aggregate_over_time, aggregation_names, sanitize


class MFCCExtractor(BaseFeatureExtractor):
    """MFCC + Δ + ΔΔ with statistical pooling over time."""

    domain = "mfcc"

    def __init__(self, sr: float, cfg: Dict):
        super().__init__(sr, cfg)
        feat = cfg["features"]
        mfcc_cfg = feat["mfcc"]

        self.n_fft = int(feat["n_fft"])
        self.hop_length = int(feat["hop_length"])
        self.window = str(feat["window"])
        self.fmin = float(feat["fmin"])
        self.fmax = float(feat["fmax"])
        self.n_mels = int(feat["n_mels"])

        self.n_mfcc = int(mfcc_cfg["n_mfcc"])
        self.use_delta = bool(mfcc_cfg["use_delta"])
        self.use_delta2 = bool(mfcc_cfg["use_delta2"])
        self.delta_width = int(mfcc_cfg.get("delta_width", 5))
        self.aggregations: List[str] = list(mfcc_cfg["aggregations"])

        self.n_streams = 1 + int(self.use_delta) + int(self.use_delta2)
        self._dim = self.n_mfcc * self.n_streams * len(self.aggregations)

    # ------------------------------------------------------------------ #
    def _mfcc_matrix(self, segment: np.ndarray) -> np.ndarray:
        """Return the (n_mfcc, n_frames) cepstral matrix for one segment."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=np.asarray(segment, dtype=np.float32),
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        # power_to_db uses a floor (top_db) internally, so no explicit epsilon
        # is needed - and it is the reference implementation everyone compares
        # against, which matters for reproducing published baselines.
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
        return librosa.feature.mfcc(S=log_mel, n_mfcc=self.n_mfcc)

    def transform(self, segment: np.ndarray) -> np.ndarray:
        import librosa

        mfcc = self._mfcc_matrix(segment)

        streams = [mfcc]
        if self.use_delta:
            # width must be odd and <= n_frames; on very short segments librosa
            # would raise, so we clamp defensively.
            width = min(self.delta_width, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))
            width = max(3, width if width % 2 == 1 else width - 1)
            delta = librosa.feature.delta(mfcc, width=width, order=1)
            streams.append(delta)
            if self.use_delta2:
                streams.append(librosa.feature.delta(mfcc, width=width, order=2))

        vectors = [aggregate_over_time(s, self.aggregations) for s in streams]
        features, _ = sanitize(np.concatenate(vectors), "mfcc")
        return features

    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            names: List[str] = []
            stream_labels = ["mfcc"]
            if self.use_delta:
                stream_labels.append("dmfcc")
            if self.use_delta2:
                stream_labels.append("ddmfcc")
            for stream in stream_labels:
                coefficients = [f"{stream}{i}" for i in range(self.n_mfcc)]
                names.extend(aggregation_names(coefficients, self.aggregations))
            self._feature_names = names
        return self._feature_names

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self._dim,)

    def describe(self) -> Dict[str, object]:
        base = super().describe()
        base.update(
            n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels,
            n_mfcc=self.n_mfcc, fmin=self.fmin, fmax=self.fmax,
            use_delta=self.use_delta, use_delta2=self.use_delta2,
            aggregations=self.aggregations, n_streams=self.n_streams,
            window_ms=round(1000 * self.n_fft / self.sr, 1),
            hop_ms=round(1000 * self.hop_length / self.sr, 1),
        )
        return base

    # ------------------------------------------------------------------ #
    def dct_frequency_map(self) -> np.ndarray:
        """|DCT-II basis| mapping cepstral coefficients to mel bands.

        Shape ``(n_mfcc, n_mels)``, each row L1-normalised.

        This is what makes MFCC SHAP values interpretable in the frequency
        domain. MFCC coefficient *i* is
        ``sum_m log_mel[m] * cos(pi * i * (m + 0.5) / n_mels)``, so the
        magnitude of that cosine tells us how much each mel band contributes to
        that coefficient. Projecting SHAP mass through this matrix gives an
        approximate per-mel-band attribution.

        Caveat we state in the paper: this ignores the *sign* of the basis, so
        it answers "which bands does this coefficient read from?" rather than
        "in which direction". It is an approximation, and we label it as one.
        """
        i = np.arange(self.n_mfcc)[:, None]
        m = np.arange(self.n_mels)[None, :]
        basis = np.cos(np.pi * i * (m + 0.5) / self.n_mels)
        magnitude = np.abs(basis)
        magnitude[0, :] = 1.0 / self.n_mels     # c0 is the mean: uniform read-out
        return magnitude / magnitude.sum(axis=1, keepdims=True)

    def mel_band_frequencies(self) -> np.ndarray:
        """Centre frequency (Hz) of each mel band - the axis for the map above."""
        import librosa

        edges = librosa.mel_frequencies(n_mels=self.n_mels + 2,
                                        fmin=self.fmin, fmax=self.fmax)
        return edges[1:-1]
