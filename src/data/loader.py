"""Audio loading with a graceful backend fallback.

We prefer soundfile (fast, no resampling surprises) and fall back to
scipy.io.wavfile, then librosa. PhysioNet 2016 wavs are 16-bit PCM mono at
2000 Hz, so the simple path almost always works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio(path: str | Path, sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Load a wav file as float32 mono in [-1, 1]. Returns (signal, sample_rate).

    If ``sr`` is given and differs from the file's rate, the caller is
    responsible for resampling (we do it in preprocessing so the whole chain
    is in one place).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # --- backend 1: soundfile -------------------------------------------
    try:
        import soundfile as sf
        x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), int(file_sr)
    except Exception:
        pass

    # --- backend 2: scipy -------------------------------------------------
    try:
        from scipy.io import wavfile
        file_sr, x = wavfile.read(str(path))
        x = np.asarray(x)
        if x.ndim > 1:
            x = x.mean(axis=1)
        # Integer PCM -> float in [-1, 1]
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        return np.ascontiguousarray(x, dtype=np.float32), int(file_sr)
    except Exception:
        pass

    # --- backend 3: librosa ----------------------------------------------
    import librosa
    x, file_sr = librosa.load(str(path), sr=sr, mono=True)
    return np.ascontiguousarray(x, dtype=np.float32), int(file_sr)


def load_reference(folder: str | Path) -> dict:
    """Read a REFERENCE.csv into {recording_id: raw_label}."""
    import pandas as pd

    path = Path(folder) / "REFERENCE.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, header=None, names=["recording_id", "label"])
    return {str(r).strip(): int(l) for r, l in zip(df["recording_id"], df["label"])}
