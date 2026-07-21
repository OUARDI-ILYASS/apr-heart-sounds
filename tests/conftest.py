"""Shared pytest fixtures.

Tests here are fast and dependency-light. Anything requiring torch, librosa or
pywt is marked so it can be deselected with `-m "not requires_optional"`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_optional: needs torch / librosa / pywt / shap"
    )


@pytest.fixture(scope="session")
def cfg():
    """The real project config, loaded and validated."""
    from src.config.loader import load_config

    return load_config(PROJECT_ROOT / "configs" / "config.yaml",
                       project_root=PROJECT_ROOT)


@pytest.fixture
def synthetic_pcg():
    """A synthetic heart sound: S1, systole, S2, diastole, repeated.

    Deterministic, so assertions about the segmenter are stable. The systolic
    interval optionally contains band-limited noise standing in for a murmur.
    """
    def _make(duration_s=3.0, sr=2000, heart_rate=75.0, murmur=False, seed=0):
        rng = np.random.default_rng(seed)
        n = int(duration_s * sr)
        x = np.zeros(n, dtype=np.float32)
        cycle = 60.0 / heart_rate
        systole_fraction = 0.35

        t_local = np.arange(int(0.08 * sr)) / sr
        s1 = np.sin(2 * np.pi * 40 * t_local) * np.exp(-25 * t_local)
        s2 = 0.7 * np.sin(2 * np.pi * 65 * t_local) * np.exp(-35 * t_local)

        start = 0.0
        while start + cycle < duration_s:
            i1 = int(start * sr)
            i2 = int((start + systole_fraction * cycle) * sr)
            x[i1:i1 + len(s1)] += s1[:max(0, n - i1)]
            x[i2:i2 + len(s2)] += s2[:max(0, n - i2)]
            if murmur:
                span = slice(i1 + len(s1), i2)
                width = max(0, span.stop - span.start)
                if width > 0:
                    x[span] += 0.25 * rng.normal(0, 1, width).astype(np.float32)
            start += cycle
        return x

    return _make


@pytest.fixture
def toy_census():
    """A small census DataFrame for split testing."""
    import pandas as pd

    rows = []
    for subdb, n, abnormal_rate in [("training-a", 40, 0.7), ("training-b", 60, 0.2),
                                    ("training-c", 20, 0.5)]:
        for i in range(n):
            rows.append({
                "recording_id": f"{subdb[-1]}{i:04d}",
                "subdb": subdb,
                "label": int(i < n * abnormal_rate),
                "duration_s": 20.0,
                "sr": 2000,
            })
    return pd.DataFrame(rows)
