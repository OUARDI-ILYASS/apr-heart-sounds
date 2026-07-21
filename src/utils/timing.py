"""Timing helpers used by the phase-summary machinery."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


class Stopwatch:
    """Accumulates named durations so a phase summary can report a breakdown."""

    def __init__(self) -> None:
        self._marks: Dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._marks[name] = self._marks.get(name, 0.0) + (time.perf_counter() - start)

    def record(self, name: str, seconds: float) -> None:
        self._marks[name] = self._marks.get(name, 0.0) + seconds

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> Dict[str, float]:
        out = {k: round(v, 3) for k, v in self._marks.items()}
        out["_total"] = round(self.total, 3)
        return out


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


@contextmanager
def timed(label: str, logger: Optional[object] = None) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    message = f"{label} finished in {human_duration(elapsed)}"
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)
    else:
        print(message)
