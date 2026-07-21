"""Shared feature-extraction machinery.

Every feature domain implements the same tiny interface so that scripts can
loop over them without special-casing:

    extractor.transform(segment) -> np.ndarray
    extractor.feature_names      -> list[str]
    extractor.output_shape       -> tuple

The ``feature_names`` requirement is not cosmetic. SHAP produces one
attribution per input dimension; without names, a SHAP plot is 234 anonymous
bars and the explainability section of the paper says nothing. Named features
turn "dimension 147 matters" into "the standard deviation of the 4th delta-MFCC
matters", which is a statement you can actually interpret.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps_stats


# --------------------------------------------------------------------------- #
# Statistical aggregation over the time axis
# --------------------------------------------------------------------------- #
AGGREGATION_FUNCTIONS = {
    "mean": lambda a, axis: np.mean(a, axis=axis),
    "std": lambda a, axis: np.std(a, axis=axis),
    "min": lambda a, axis: np.min(a, axis=axis),
    "max": lambda a, axis: np.max(a, axis=axis),
    "median": lambda a, axis: np.median(a, axis=axis),
    "range": lambda a, axis: np.ptp(a, axis=axis),
    # skew/kurtosis are undefined for a constant sequence (zero variance):
    # scipy returns NaN. We substitute 0, which is the correct limiting value
    # for a degenerate distribution and avoids poisoning the feature matrix.
    "skew": lambda a, axis: np.nan_to_num(
        sps_stats.skew(a, axis=axis, bias=False), nan=0.0, posinf=0.0, neginf=0.0),
    "kurtosis": lambda a, axis: np.nan_to_num(
        sps_stats.kurtosis(a, axis=axis, bias=False), nan=0.0, posinf=0.0, neginf=0.0),
    "q25": lambda a, axis: np.percentile(a, 25, axis=axis),
    "q75": lambda a, axis: np.percentile(a, 75, axis=axis),
    "iqr": lambda a, axis: np.percentile(a, 75, axis=axis) - np.percentile(a, 25, axis=axis),
}


def aggregate_over_time(matrix: np.ndarray, aggregations: Sequence[str],
                        axis: int = 1) -> np.ndarray:
    """Summarise a (n_coefficients, n_frames) matrix into one vector.

    Output ordering is *coefficient-major*: all aggregations of coefficient 0,
    then all aggregations of coefficient 1, and so on. This keeps related
    features adjacent, which makes SHAP summary plots readable and makes it
    trivial to slice out "everything about coefficient k".

    PROFESSOR Q: "Why summarise instead of feeding the sequence to the model?"
    A: Because SVM and Random Forest require a fixed-length vector - they have
       no notion of a time axis. The aggregation is exactly where the classical
       branch loses temporal structure, and that loss is precisely what the CNN
       branch is designed to avoid. The contrast between the two is one of the
       findings of the study, not an accident of implementation.
    """
    unknown = [a for a in aggregations if a not in AGGREGATION_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown aggregation(s): {unknown}. "
            f"Available: {sorted(AGGREGATION_FUNCTIONS)}"
        )
    stacked = np.stack(
        [AGGREGATION_FUNCTIONS[name](matrix, 1 if axis == 1 else axis)
         for name in aggregations],
        axis=-1,
    )                                   # (n_coefficients, n_aggregations)
    return stacked.reshape(-1).astype(np.float32)


def aggregation_names(coefficient_names: Sequence[str],
                      aggregations: Sequence[str]) -> List[str]:
    """Names matching the ordering produced by :func:`aggregate_over_time`."""
    return [f"{coef}_{agg}" for coef in coefficient_names for agg in aggregations]


def safe_log(x: np.ndarray, offset: float = 1e-10) -> np.ndarray:
    """log(x + offset) with the offset chosen to keep the result finite.

    Every log in this codebase goes through here. Bare ``np.log`` on an energy
    that happens to be exactly zero - which occurs after spike removal zeroes a
    span - yields -inf, which then propagates into every statistic computed
    from it.
    """
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), 0.0) + offset)


def sanitize(features: np.ndarray, name: str = "features") -> Tuple[np.ndarray, Dict[str, int]]:
    """Replace any residual non-finite values and report how many there were.

    This is a *net*, not a strategy: each extractor handles its own known
    degenerate cases explicitly. If this function ever reports a non-zero
    count, that is a bug worth investigating, and the count appears in the
    phase summary so it cannot be ignored.
    """
    features = np.asarray(features, dtype=np.float32)
    n_nan = int(np.isnan(features).sum())
    n_inf = int(np.isinf(features).sum())
    if n_nan or n_inf:
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, {"n_nan": n_nan, "n_inf": n_inf}


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class BaseFeatureExtractor(ABC):
    """Common interface for MFCC / log-Mel / PWP extractors."""

    domain: str = "base"

    def __init__(self, sr: float, cfg: Dict):
        self.sr = float(sr)
        self.cfg = cfg
        self._feature_names: List[str] | None = None

    @abstractmethod
    def transform(self, segment: np.ndarray) -> np.ndarray:
        """Extract features from one segment."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """One human-readable name per output dimension (flattened order)."""

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """Shape of a single segment's feature array."""

    def transform_batch(self, segments: np.ndarray,
                        progress: bool = False) -> np.ndarray:
        """Extract features for a stack of segments, shape (n, seg_len)."""
        out = np.empty((len(segments), *self.output_shape), dtype=np.float32)
        for i, segment in enumerate(segments):
            out[i] = self.transform(segment)
            if progress and (i + 1) % 500 == 0:
                print(f"\r  {self.domain}: {i + 1}/{len(segments)}", end="", flush=True)
        if progress:
            print()
        return out

    def describe(self) -> Dict[str, object]:
        """Parameter record for the phase summary."""
        return {
            "domain": self.domain,
            "sr": self.sr,
            "output_shape": list(self.output_shape),
            "n_features": int(np.prod(self.output_shape)),
        }
