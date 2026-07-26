"""Assemble per-model results into the comparison tables the paper needs."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def build_comparison_table(results: Dict[str, Dict], level: str = "recording",
                           metrics: Optional[List[str]] = None
                           ) -> List[Dict[str, object]]:
    """One row per model, with the bootstrap CI folded into the MAcc cell."""
    metrics = metrics or ["accuracy", "sensitivity", "specificity", "macc",
                          "f1", "roc_auc"]
    rows: List[Dict[str, object]] = []

    for model_name, payload in results.items():
        entry = payload.get(level, payload)
        row: Dict[str, object] = {"model": model_name}
        for name in metrics:
            value = entry.get(name)
            row[name] = round(float(value), 4) if isinstance(value, (int, float)) else None

        ci = (payload.get("bootstrap") or {}).get("macc")
        if ci:
            row["macc_ci"] = f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
        row["n"] = entry.get("n")
        rows.append(row)

    return sorted(rows, key=lambda r: -(r.get("macc") or 0))


def build_pairwise_table(mcnemar_results: Dict[str, Dict]) -> List[Dict[str, object]]:
    """Pairwise significance table from McNemar results."""
    rows = []
    for pair, result in mcnemar_results.items():
        rows.append({
            "comparison": pair,
            "a_only_correct": result.get("a_only_correct"),
            "b_only_correct": result.get("b_only_correct"),
            "statistic": round(float(result.get("statistic", 0)), 3),
            "p_value": round(float(result.get("p_value", 1)), 4),
            "significant": result.get("significant_at_0.05"),
            "test": result.get("test"),
        })
    return sorted(rows, key=lambda r: r["p_value"])


def rank_models(results: Dict[str, Dict], metric: str = "macc",
                level: str = "recording") -> List[Dict[str, object]]:
    """Rank models and flag which differences survive the CI overlap check.

    ``distinguishable_from_best`` is False when a model's CI overlaps the best
    model's CI. That is a deliberately conservative reading (overlapping CIs do
    not strictly imply non-significance), and it is the reading we use in the
    paper before falling back to McNemar for the formal test.
    """
    entries = []
    for name, payload in results.items():
        level_metrics = payload.get(level, payload)
        ci = (payload.get("bootstrap") or {}).get(metric, {})
        entries.append({
            "model": name,
            metric: float(level_metrics.get(metric, float("nan"))),
            "ci_lower": ci.get("lower"),
            "ci_upper": ci.get("upper"),
        })

    entries.sort(key=lambda e: -(e[metric] if e[metric] == e[metric] else -np.inf))
    if entries and entries[0].get("ci_lower") is not None:
        best_lower = entries[0]["ci_lower"]
        for entry in entries:
            upper = entry.get("ci_upper")
            entry["distinguishable_from_best"] = (
                bool(upper is not None and upper < best_lower)
            )

    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def summarize_ablations(base_result: Dict, ablation_results: Dict[str, Dict],
                        metric: str = "macc") -> List[Dict[str, object]]:
    """Delta table: how much does each ablation cost?

    The sign convention is "delta = ablated - full", so a negative delta means
    the removed component was helping. Reporting the delta rather than two
    absolute numbers is what makes an ablation table readable at a glance.
    """
    base_value = float(base_result.get(metric, float("nan")))
    rows = [{
        "variant": "Full model",
        metric: round(base_value, 4),
        "delta": 0.0,
        "delta_pct": 0.0,
        "changed": "-",
    }]

    for name, result in ablation_results.items():
        value = float(result.get(metric, float("nan")))
        delta = value - base_value
        rows.append({
            "variant": name,
            metric: round(value, 4),
            "delta": round(delta, 4),
            "delta_pct": round(100.0 * delta / base_value, 2) if base_value else None,
            "changed": result.get("changed_params", "-"),
        })
    return rows
