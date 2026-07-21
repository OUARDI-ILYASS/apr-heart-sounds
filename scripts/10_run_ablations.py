#!/usr/bin/env python3
"""Phase 10 - Run the ablation study.

Each ablation is a *partial* config merged onto the base, so the only thing
that differs is the parameter under test. For every variant we re-run only the
phases that parameter can affect, then report the delta in test MAcc.

The ablations and what each one is for:
    A1 mfcc_only      - does the PWP branch contribute anything?
    A2 linear_spec    - does the mel warping matter at 25-400 Hz?
    A3 no_classweight - how much of the performance is imbalance handling?
    A4 no_delta       - do the dynamic MFCC coefficients earn their place?
    A5 shallow_cnn    - are four conv blocks needed?
    A6 seglen_1s      - is the 3 s window justified?
    A7 wavelet_haar   - does the choice of mother wavelet matter?

Run:  python scripts/10_run_ablations.py [--only A1_mfcc_only A3_no_classweight]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from _bootstrap import setup, build_parser, PROJECT_ROOT

from src.config.loader import load_config, config_diff
from src.evaluation.aggregation import summarize_ablations
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "10_run_ablations"

# Which phases must be re-run for each ablation. Re-running everything would be
# correct but wasteful; re-running too little would silently reuse stale
# artifacts. This mapping is the contract.
PHASE_DEPENDENCIES = {
    "A1_mfcc_only":      ["04", "06"],
    "A2_linear_spec":    ["02", "05", "06"],
    "A3_no_classweight": ["04", "05", "06"],
    "A4_no_delta":       ["02", "04", "06"],
    "A5_shallow_cnn":    ["05", "06"],
    "A6_seglen_1s":      ["01", "02", "04", "05", "06"],
    "A7_wavelet_haar":   ["02", "04", "06"],
}

SCRIPT_NAMES = {
    "01": "01_preprocess_audio.py", "02": "02_extract_features.py",
    "04": "04_train_classical.py", "05": "05_train_cnn.py",
    "06": "06_evaluate_models.py",
}


def main() -> int:
    parser = build_parser(__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these ablations by name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without executing them.")
    cfg, logger, args = setup(PHASE, "Run the ablation study", parser)

    summary = PhaseSummary(PHASE, cfg, next_phase="11_build_report_assets")
    watch = Stopwatch()

    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "ablations")
    base_results_path = Path(cfg["_abs_paths"]["results_dir"]) / "evaluation" / "test_results.json"

    if not base_results_path.exists():
        logger.error("Baseline test results not found. Run phases 00-06 first.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    base_results = load_json(base_results_path)
    base_best = base_results["ranking"][0]
    logger.info(f"Baseline: {base_best['model']} with MAcc {base_best['macc']:.4f}")

    enabled = list(args.only) if args.only else list(cfg["ablations"]["enabled"])
    ablation_configs = dict(cfg["ablations"]["configs"])
    base_config_path = PROJECT_ROOT / "configs" / "config.yaml"

    ablation_results = {}

    for name in enabled:
        if name not in ablation_configs:
            logger.warning(f"Unknown ablation '{name}', skipping")
            continue

        override_path = PROJECT_ROOT / ablation_configs[name]
        if not override_path.exists():
            logger.warning(f"Config not found for {name}: {override_path}")
            continue

        logger.info("=" * 70)
        logger.info(f"ABLATION {name}")
        logger.info("=" * 70)

        # Prove which parameters actually differ from the base.
        base_cfg = load_config(base_config_path, project_root=PROJECT_ROOT)
        ablated_cfg = load_config(override_path, base_path=base_config_path,
                                  project_root=PROJECT_ROOT)
        differences = config_diff(
            {k: v for k, v in base_cfg.to_dict().items() if not k.startswith("_")},
            {k: v for k, v in ablated_cfg.to_dict().items() if not k.startswith("_")},
        )
        # `experiment.*` differences are bookkeeping, not the parameter under test.
        substantive = {k: v for k, v in differences.items()
                       if not k.startswith("experiment.")}
        logger.info(f"  changed parameters: {list(substantive)}")

        phases = PHASE_DEPENDENCIES.get(name, ["06"])
        logger.info(f"  re-running phases: {phases}")

        succeeded = True
        with watch.section(name):
            for phase_id in phases:
                script = SCRIPT_NAMES[phase_id]
                command = [
                    sys.executable, str(PROJECT_ROOT / "scripts" / script),
                    "--config", str(override_path.relative_to(PROJECT_ROOT)),
                    "--base-config", "configs/config.yaml",
                    "--tag", name,
                ]
                logger.info(f"    $ {' '.join(command[1:])}")
                if args.dry_run:
                    continue
                completed = subprocess.run(command, cwd=PROJECT_ROOT,
                                           capture_output=True, text=True)
                if completed.returncode != 0:
                    logger.error(f"    phase {phase_id} failed:\n"
                                 f"{completed.stderr[-2000:]}")
                    succeeded = False
                    break

        if args.dry_run:
            continue

        if not succeeded:
            summary.add_warning(f"Ablation {name} failed and is excluded from the table.")
            ablation_results[name] = {"macc": float("nan"), "error": "phase failed",
                                      "changed_params": ", ".join(substantive)}
            continue

        # Phase 06 overwrites test_results.json, so read it immediately and
        # archive it under the ablation's own name.
        if base_results_path.exists():
            ablated = load_json(base_results_path)
            best = ablated["ranking"][0]
            save_json(results_dir / f"{name}_results.json", ablated)
            ablation_results[name] = {
                "macc": float(best["macc"]),
                "best_model": best["model"],
                "changed_params": ", ".join(substantive),
                "config_diff": substantive,
            }
            delta = best["macc"] - base_best["macc"]
            logger.info(f"  -> MAcc {best['macc']:.4f} ({delta:+.4f} vs baseline)")

    if args.dry_run:
        logger.info("Dry run complete; no phases were executed.")
        return 0

    if not ablation_results:
        logger.error("No ablation produced a result.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    # ---- delta table -------------------------------------------------------
    table = summarize_ablations({"macc": base_best["macc"]}, ablation_results, "macc")
    save_json(results_dir / "ablation_summary.json", {
        "baseline": base_best,
        "ablations": ablation_results,
        "table": table,
    })
    summary.add_artifact(results_dir / "ablation_summary.json")
    summary.add_table("ablation_deltas", table)

    logger.info("")
    logger.info("Ablation summary (delta = ablated - full):")
    for row in table:
        logger.info(f"  {row['variant']:22s} MAcc {row['macc']:.4f}  "
                    f"delta {row.get('delta', 0):+.4f}")

    # A negative delta means the removed component was contributing.
    harmful = [r for r in table if (r.get("delta") or 0) < -0.02]
    beneficial = [r for r in table if (r.get("delta") or 0) > 0.02]
    summary.add_finding(
        "components_that_matter", [r["variant"] for r in harmful],
        "Ablations that cost more than 2 MAcc points; these components are earning "
        "their place in the pipeline."
    )
    if beneficial:
        summary.add_finding(
            "ablations_that_helped", [r["variant"] for r in beneficial],
            "Ablations that IMPROVED performance. Report these honestly - they mean "
            "the corresponding design choice in the full model was not justified."
        )
        summary.add_warning(
            f"{len(beneficial)} ablation(s) outperformed the full model. The full "
            "model's design should be revised or the result explicitly discussed."
        )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
