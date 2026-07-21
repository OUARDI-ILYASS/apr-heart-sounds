"""Feature scaling - fit on TRAIN only, applied unchanged to val and test.

PROFESSOR Q: "Where do you fit the scaler and why does it matter?"
A: On the training partition only. Fitting a StandardScaler on the full dataset
   leaks the mean and variance of the test set into training. It sounds
   harmless - it is only two numbers per feature - but it is still information
   the model should not have, it is the single most common subtle leak in
   applied ML papers, and reviewers look for it. Our loader physically cannot
   do it wrong: `fit_scaler` takes only the training matrix, and `apply_scaler`
   takes a fitted object.

For the CNN, the same logic applies but per mel band: we compute the mean and
standard deviation of each of the 32 log-Mel bands over the training segments
and standardise every split with those statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..utils.io import save_joblib, load_joblib, save_json


def fit_scaler(X_train: np.ndarray, method: str = "standard"):
    """Fit a scaler on training features only."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        # RobustScaler uses median/IQR - worth trying if outlier segments
        # (residual spikes) dominate the variance.
        "robust": RobustScaler,
        "none": None,
    }
    if method not in scalers:
        raise ValueError(f"Unknown scaling method '{method}'. Options: {sorted(scalers)}")
    if method == "none":
        return None
    return scalers[method]().fit(np.asarray(X_train, dtype=np.float64))


def apply_scaler(scaler, X: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(X, dtype=np.float32)
    return scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)


def fit_logmel_norm(X_train: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-band mean/std for the CNN input, computed over training segments.

    ``X_train`` has shape (n_segments, n_mels, n_frames). We reduce over
    segments AND time, giving one mean and one std per mel band. Normalising
    per band rather than globally stops the loudest bands from dominating the
    first convolution's effective learning rate.
    """
    X = np.asarray(X_train, dtype=np.float64)
    mean = X.mean(axis=(0, 2), keepdims=False)
    std = X.std(axis=(0, 2), keepdims=False)
    std = np.where(std < 1e-8, 1.0, std)     # guard silent bands
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_logmel_norm(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(stats["mean"])[None, :, None]
    std = np.asarray(stats["std"])[None, :, None]
    return ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32)


def save_scaler(path: str | Path, scaler, metadata: Dict | None = None) -> Path:
    """Persist a scaler plus a human-readable sidecar describing what it saw."""
    path = Path(path)
    save_joblib(path, scaler)
    if metadata is not None:
        save_json(path.with_suffix(".meta.json"), metadata)
    return path


def load_scaler(path: str | Path):
    return load_joblib(path)


def scaler_report(scaler, feature_names) -> Dict[str, object]:
    """Diagnostics: which features were near-constant in training?

    A feature with near-zero training variance is uninformative and its scaled
    values explode, so it is worth flagging. It usually indicates a degenerate
    band (e.g. a mel filter with no energy) rather than a real signal.
    """
    if scaler is None or not hasattr(scaler, "scale_"):
        return {}
    scale = np.asarray(scaler.scale_)
    degenerate = np.where(scale < 1e-6)[0]
    return {
        "n_features": int(scale.size),
        "n_near_constant": int(degenerate.size),
        "near_constant_features": [str(feature_names[i]) for i in degenerate[:20]],
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
    }
