"""Typed, defensive I/O helpers.

Every artifact in this project goes through here. Centralising I/O buys three
things:

* directories are always created before writing (no half-failed phases),
* saved arrays are always checked for NaN/Inf (silent NaNs are the single most
  common cause of "my model won't train"),
* loads fail with a message that names the *phase* that should have produced
  the missing file, rather than a bare ``FileNotFoundError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Which phase produces what - used to write helpful error messages
# --------------------------------------------------------------------------- #
PHASE_PRODUCERS: Dict[str, str] = {
    "data/raw": "00_download_data.py",
    "data/interim": "01_preprocess_audio.py",
    "data/processed": "02_extract_features.py",
    "results/clustering": "03_cluster_features.py",
    "models/classical": "04_train_classical.py",
    "models/cnn": "05_train_cnn.py",
    "results/evaluation": "06_evaluate_models.py",
    "results/xai/shap": "07_explain_shap.py",
    "results/xai/gradcam": "08_explain_gradcam.py",
    "results/segmentation": "09_cycle_alignment.py",
    "results/ablations": "10_run_ablations.py",
}


def _producer_hint(path: Path) -> str:
    text = str(path).replace("\\", "/")
    for marker, script in PHASE_PRODUCERS.items():
        if marker in text:
            return f" This artifact is produced by scripts/{script} - run that phase first."
    return ""


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of a file path; return the file path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# NumPy
# --------------------------------------------------------------------------- #
def save_npy(path: str | Path, array: np.ndarray, check_finite: bool = True) -> Path:
    """Save an array, refusing to persist NaN/Inf unless explicitly allowed.

    PROFESSOR Q: "How do you handle NaNs from feature extraction?"
    A: We refuse to write them. A NaN in a feature file propagates silently -
       sklearn raises much later with an unrelated message, and PyTorch just
       produces NaN loss. Failing at the point of creation makes the cause
       obvious. Feature extractors sanitise known-degenerate cases (log of
       zero energy, kurtosis of a constant signal) explicitly rather than
       relying on this net.
    """
    path = ensure_parent(path)
    array = np.asarray(array)
    if check_finite and array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        n_nan = int(np.isnan(array).sum())
        n_inf = int(np.isinf(array).sum())
        raise ValueError(
            f"Refusing to save non-finite array to {path}: "
            f"{n_nan} NaN, {n_inf} Inf out of {array.size} values."
        )
    np.save(path, array)
    return path


def load_npy(path: str | Path, mmap: bool = False) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing array: {path}.{_producer_hint(path)}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    path = ensure_parent(path)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing archive: {path}.{_producer_hint(path)}")
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars, arrays and Paths.

    Without this, ``json.dump`` blows up on ``np.float32`` - which is what
    every sklearn metric returns.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            value = float(o)
            return None if not np.isfinite(value) else value
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def save_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, cls=NumpyJSONEncoder, sort_keys=False)
    return path


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}.{_producer_hint(path)}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# joblib (sklearn models, scalers)
# --------------------------------------------------------------------------- #
def save_joblib(path: str | Path, obj: Any) -> Path:
    import joblib

    path = ensure_parent(path)
    joblib.dump(obj, path, compress=3)
    return path


def load_joblib(path: str | Path) -> Any:
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}.{_producer_hint(path)}")
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Torch
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Save a torch checkpoint.

    We always store the architecture spec and the config hash *inside* the
    checkpoint, not just the weights. A bare ``state_dict`` is useless six
    weeks later when you no longer remember how many channels the third block
    had.
    """
    import torch

    path = ensure_parent(path)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}.{_producer_hint(path)}")
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def require_artifacts(paths: Iterable[str | Path], phase: Optional[str] = None) -> None:
    """Assert that all upstream artifacts exist before a phase starts.

    Called at the top of every script. The point is to fail in the first second
    with a clear message instead of twenty minutes in with a cryptic one.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        header = f"Phase '{phase}' cannot start: missing upstream artifacts." if phase \
            else "Missing required artifacts."
        hints = {_producer_hint(Path(m)).strip() for m in missing}
        hints.discard("")
        raise FileNotFoundError(
            header
            + "\n\nMissing:\n  - "
            + "\n  - ".join(missing)
            + ("\n\n" + "\n".join(sorted(hints)) if hints else "")
        )


def describe_array(array: np.ndarray) -> Dict[str, Any]:
    """Compact statistics used in phase summaries."""
    array = np.asarray(array)
    out: Dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "n_elements": int(array.size),
        "mb": round(array.nbytes / 1024 ** 2, 2),
    }
    if array.dtype.kind in "fiu" and array.size:
        finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
        if finite.size:
            out.update(
                min=float(finite.min()),
                max=float(finite.max()),
                mean=float(finite.mean()),
                std=float(finite.std()),
            )
        if array.dtype.kind == "f":
            out["n_nan"] = int(np.isnan(array).sum())
            out["n_inf"] = int(np.isinf(array).sum())
    return out
