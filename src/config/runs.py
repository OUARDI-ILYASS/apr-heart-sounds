"""Run-scoped artifact paths.

Every run writes into `runs/<run_id>/`, where run_id is "baseline" for the full
model and the ablation name (e.g. "A3_no_classweight") for a variant. Reads fall
back to the baseline run, which is what lets an ablation that skips a phase
consume the baseline's artifacts without copying them.

PROFESSOR Q: "How do you keep ablation runs from destroying the baseline?"
A: Outputs are namespaced per run, so they physically cannot collide. Reads are
   layered: a phase looks in its own run directory first and falls back to
   baseline. So A1 (which does not re-run feature extraction) reads baseline
   features, while A4 (which does) writes and then reads its own -- with no
   flag, no copying, and no way to get it wrong by forgetting a step.

Two directories are deliberately NOT scoped:
  raw_dir   -- the corpus is ~3 GB and immutable; duplicating it per run would
               be absurd and no phase ever writes to it.
  paper_dir -- there is one paper. Phase 11 chooses which run to report on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .loader import ConfigDict

BASELINE_RUN = "baseline"

# Config key -> subdirectory name inside runs/<run_id>/
RUN_SCOPED: Dict[str, str] = {
    "interim_dir": "interim",
    "processed_dir": "processed",
    "models_dir": "models",
    "results_dir": "results",
    "figures_dir": "figures",
    "reports_dir": "reports",
}
SHARED = {"raw_dir", "paper_dir"}


def apply_run_scope(cfg: ConfigDict, run_id: str = BASELINE_RUN) -> ConfigDict:
    """Redirect output directories into runs/<run_id>/ and record the fallback."""
    root = Path(cfg["_project_root"]) / "runs"

    run_paths: Dict[str, str] = {}
    baseline_paths: Dict[str, str] = {}

    for key, original in dict(cfg["_abs_paths"]).items():
        if key in RUN_SCOPED:
            sub = RUN_SCOPED[key]
            run_paths[key] = str(root / run_id / sub)
            baseline_paths[key] = str(root / BASELINE_RUN / sub)
        else:
            run_paths[key] = original
            baseline_paths[key] = original

    cfg["_run_id"] = run_id
    cfg["_is_baseline"] = run_id == BASELINE_RUN
    cfg["_abs_paths"] = ConfigDict(run_paths)
    cfg["_baseline_paths"] = ConfigDict(baseline_paths)
    return cfg


def output_path(cfg: ConfigDict, key: str, *parts: str) -> Path:
    """Where THIS run writes. Always inside runs/<run_id>/, parents created.

    Never falls back to baseline. A write that resolved to the baseline
    directory is exactly the bug this module exists to prevent.
    """
    path = Path(cfg["_abs_paths"][key]).joinpath(*parts)
    (path.parent if path.suffix else path).mkdir(parents=True, exist_ok=True)
    return path


def input_path(cfg: ConfigDict, key: str, *parts: str,
               required: bool = True) -> Path:
    """Where to READ an upstream artifact: own run first, then baseline."""
    own = Path(cfg["_abs_paths"][key]).joinpath(*parts)
    if own.exists():
        return own

    fallback = Path(cfg["_baseline_paths"][key]).joinpath(*parts)
    if fallback.exists():
        return fallback

    if required:
        raise FileNotFoundError(
            f"Artifact not found in run '{cfg['_run_id']}' or in baseline.\n"
            f"  looked in : {own}\n"
            f"  fallback  : {fallback}\n"
            f"Run the producing phase for this run, or run the baseline first."
        )
    return own


def run_provenance(cfg: ConfigDict) -> Dict[str, Any]:
    """Recorded in every phase summary so a result names its own run."""
    return {
        "run_id": cfg["_run_id"],
        "is_baseline": cfg["_is_baseline"],
        "config_hash": cfg.get("_config_hash"),
        "experiment_name": cfg["experiment"]["name"],
        "output_root": cfg["_abs_paths"]["results_dir"],
        "baseline_fallback": cfg["_baseline_paths"]["results_dir"],
    }