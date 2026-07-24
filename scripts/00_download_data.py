"""Phase 00 - Acquire the PhysioNet/CinC 2016 dataset and build a raw census.

Produces a single table describing every recording (id, sub-database, label,
duration, sample rate) plus dataset-level statistics. Everything downstream
reads that table rather than walking the filesystem, so the set of recordings
used is fixed and auditable from one artifact.

Run:  python scripts/00_download_data.py [--skip-download]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from _bootstrap import setup, build_parser, PROJECT_ROOT

from src.data.download import (
    download_dataset, verify_dataset, build_raw_census, census_statistics,
)
from src.utils.io import save_json, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "00_download_data"


def main() -> int:
    parser = build_parser(__doc__.split("\n")[0])
    parser.add_argument("--skip-download", action="store_true",
                        help="Assume the data is already on disk.")
    cfg, logger, args = setup(PHASE, "Download data and build the raw census", parser)

    summary = PhaseSummary(PHASE, cfg, next_phase="01_preprocess_audio")
    watch = Stopwatch()

    raw_dir = Path(cfg["_abs_paths"]["raw_dir"])
    subdatabases = list(cfg["dataset"]["subdatabases"])

    # ---- download ------------------------------------------------------
    if not args.skip_download:
        logger.info(f"Downloading into {raw_dir} ...")
        with watch.section("download"):
            status = download_dataset(raw_dir, cfg["dataset"]["base_url"],
                                      subdatabases, force=args.force)
        for name, state in status.items():
            (logger.warning if state.startswith("failed") else logger.info)(
                f"  {name}: {state}"
            )
        summary.add_parameters({"download_status": status})
        if any(str(s).startswith("failed") for s in status.values()):
            summary.add_warning(
                "One or more sub-databases failed to download. If your network "
                "blocks physionet.org, fetch the archives manually into "
                f"{raw_dir} and re-run with --skip-download."
            )
    else:
        logger.info("Skipping download (--skip-download).")

    # ---- verify --------------------------------------------------------
    with watch.section("verify"):
        verification = verify_dataset(raw_dir, subdatabases)

    missing = [name for name, info in verification.items() if not info.get("present")]
    if missing:
        logger.error(f"Missing sub-databases: {missing}")
        summary.add_assertion("all_subdatabases_present", False, f"missing: {missing}")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1
    summary.add_assertion("all_subdatabases_present", True,
                          f"{len(subdatabases)} sub-databases found")

    mismatched = [n for n, i in verification.items() if not i.get("wav_hea_match", True)]
    summary.add_assertion(
        "wav_header_counts_match", not mismatched,
        "every .wav has a matching .hea" if not mismatched else f"mismatch in {mismatched}",
    )

    # ---- census --------------------------------------------------------
    logger.info("Building the raw census ...")
    with watch.section("census"):
        census = build_raw_census(raw_dir, subdatabases,
                                  dict(cfg["dataset"]["label_map"]))
    if census.empty:
        logger.error("Census is empty - no recordings were indexed.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    statistics = census_statistics(census)

    census_path = ensure_dir(cfg["_abs_paths"]["interim_dir"]) / "raw_census.csv"
    census.to_csv(census_path, index=False)
    statistics_path = save_json(
        Path(cfg["_abs_paths"]["interim_dir"]) / "raw_census_stats.json", statistics
    )

    # ---- report --------------------------------------------------------
    logger.info(f"Recordings   : {statistics['n_recordings']}")
    logger.info(f"Normal       : {statistics['n_normal']}")
    logger.info(f"Abnormal     : {statistics['n_abnormal']} "
                f"({statistics['pct_abnormal']:.1f}%)")
    logger.info(f"Imbalance    : {statistics['imbalance_ratio']}:1 (normal:abnormal)")
    logger.info(f"Total audio  : {statistics['duration_total_hours']:.2f} h")
    logger.info(f"Duration     : {statistics['duration_min_s']:.1f}-"
                f"{statistics['duration_max_s']:.1f} s "
                f"(median {statistics['duration_median_s']:.1f} s)")

    for row in statistics["per_subdatabase"]:
        logger.info(f"  {row['subdb']}: n={row['n']:4d}  "
                    f"abnormal={row['pct_abnormal']:5.1f}%  "
                    f"mean {row['mean_duration_s']:.1f} s")

    summary.add_artifact(census_path, n_rows=len(census))
    summary.add_artifact(statistics_path)
    summary.add_parameters({"subdatabases": subdatabases, "raw_dir": str(raw_dir)})
    summary.add_finding("n_recordings", statistics["n_recordings"])
    summary.add_finding("pct_abnormal", statistics["pct_abnormal"],
                        "Class imbalance drives every design decision downstream")
    summary.add_finding("imbalance_ratio", statistics["imbalance_ratio"])
    summary.add_finding("duration_total_hours", statistics["duration_total_hours"])
    summary.add_table("per_subdatabase", statistics["per_subdatabase"])

    # Sample-rate consistency: the whole pipeline assumes a single rate.
    rates = statistics["sample_rates"]
    summary.add_assertion(
        "single_sample_rate", len(rates) == 1,
        f"sample rates present: {rates}",
    )
    if len(rates) > 1:
        summary.add_warning(
            f"Multiple sample rates found ({rates}). Phase 01 resamples "
            f"everything to {cfg['preprocessing']['target_sr']} Hz, but verify "
            "this is intended."
        )

    # The prevalence spread across sites is the confound the splits must handle.
    prevalences = [r["pct_abnormal"] for r in statistics["per_subdatabase"]]
    spread = max(prevalences) - min(prevalences)
    summary.add_finding(
        "prevalence_spread_pp", round(spread, 1),
        "Difference in abnormal rate between the most and least pathological "
        "sub-database. A large spread means site identity is predictive of the "
        "label, which is exactly the shortcut the stratified splits prevent."
    )
    if spread > 30:
        summary.add_note(
            f"Abnormal prevalence ranges over {spread:.0f} percentage points across "
            "sub-databases. Splits are stratified jointly on (label, subdb) in "
            "phase 01, and per-site metrics are reported in phase 06 so any "
            "site shortcut would be visible."
        )

    summary.touch_split()   # no split exists yet
    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
