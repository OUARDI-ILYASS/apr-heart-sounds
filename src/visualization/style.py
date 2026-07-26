"""Matplotlib styling for IEEE two-column figures.

Getting figure sizing right matters more than it sounds. A figure drawn at
default size and then shrunk by LaTeX to column width has 6 pt axis labels that
nobody can read - the single most common reason a reviewer complains about
figure quality. We draw at the *final* physical size (3.5 in for one column,
7.16 in for two) with fonts already at the target point size, so LaTeX includes
them at scale 1.0 and nothing is resampled.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")   # No display in a headless run; must precede pyplot.
import matplotlib.pyplot as plt


IEEE_RC: Dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Type 42 = TrueType. Many publishers reject Type 3 fonts, which is
    # matplotlib's PDF default.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_ieee_style(cfg: Optional[Dict] = None) -> None:
    """Install the IEEE rcParams globally. Call once per script."""
    rc = dict(IEEE_RC)
    if cfg is not None:
        figures = cfg.get("figures", {})
        rc["font.size"] = figures.get("font_size", rc["font.size"])
        rc["figure.dpi"] = figures.get("dpi", rc["figure.dpi"])
        rc["savefig.dpi"] = figures.get("dpi", rc["savefig.dpi"])
        rc["font.family"] = figures.get("font_family", rc["font.family"])
    plt.rcParams.update(rc)


def figure_size(cfg: Optional[Dict], columns: int = 1,
                height: Optional[float] = None) -> Tuple[float, float]:
    """Physical figure size in inches for a 1- or 2-column IEEE figure."""
    figures = (cfg or {}).get("figures", {})
    width = (figures.get("single_col_width", 3.5) if columns == 1
             else figures.get("double_col_width", 7.16))
    return (float(width), float(height if height is not None
                                 else figures.get("default_height", 2.6)))


def class_colors(cfg: Optional[Dict] = None) -> List[str]:
    """Colour-blind-safe class colours. Blue = Normal, red = Abnormal."""
    return list((cfg or {}).get("figures", {}).get(
        "class_colors", ["#2E86AB", "#D7263D"]
    ))


def save_figure(fig, path: str | Path, cfg: Optional[Dict] = None,
                formats: Optional[List[str]] = None, close: bool = True
                ) -> List[Path]:
    """Save a figure in every configured format (PNG for previews, PDF for LaTeX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formats = formats or (cfg or {}).get("figures", {}).get("formats", ["png", "pdf"])

    written: List[Path] = []
    for extension in formats:
        target = path.with_suffix(f".{extension}")
        fig.savefig(target, format=extension)
        written.append(target)

    if close:
        plt.close(fig)
    return written


@contextmanager
def ieee_figure(cfg: Optional[Dict] = None, columns: int = 1,
                height: Optional[float] = None, **subplot_kwargs) -> Iterator:
    """Context manager yielding a correctly sized ``(fig, ax)``."""
    apply_ieee_style(cfg)
    fig, ax = plt.subplots(figsize=figure_size(cfg, columns, height), **subplot_kwargs)
    try:
        yield fig, ax
    finally:
        fig.tight_layout(pad=0.3)
