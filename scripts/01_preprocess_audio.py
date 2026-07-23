
"""Phase 01 - Preprocess audio, build splits, and segment into fixed windows.

Chain per recording: resample -> bandpass -> spike removal -> normalise ->
segment. Splits are created at RECORDING level and asserted disjoint before
any segment is written.

Artifacts:
    data/interim/splits.json          recording IDs per split
    data/interim/segments_<split>.npy segment waveforms
    data/interim/segment_index.csv    one row per segment with its provenance

Run:  python scripts/01_preprocess_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from src.data.loader import load_audio
from src.data.preprocessing import preprocess_recording, quality_flags
from src.data.segmentation import segment_signal, hop_for_split
from src.data.splits import (
    make_splits, assert_no_leakage, split_statistics, stratification_quality,
)
from src.utils.io import save_npy, save_json, require_artifacts, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "01_preprocess_audio"


def main() -> int:
    cfg, logger, args = setup(PHASE, "Preprocess, split and segment the audio")
    summary = PhaseSummary(PHASE, cfg, next_phase="02_extract_features")
    watch = Stopwatch()

    interim = ensure_dir(cfg["_abs_paths"]["interim_dir"])
    raw_dir = Path(cfg["_abs_paths"]["raw_dir"])
    census_path = interim / "raw_census.csv"
    require_artifacts([census_path], phase=PHASE)

    census = pd.read_csv(census_path, dtype={"recording_id": str})
    summary.add_input(census_path, n_recordings=len(census))
    logger.info(f"Loaded census: {len(census)} recordings")

    # ---- splits --------------------------------------------------------
    logger.info("Building subject-independent stratified splits ...")
    with watch.section("splits"):
        splits = make_splits(
            census,
            train_ratio=float(cfg["splits"]["train_ratio"]),
            val_ratio=float(cfg["splits"]["val_ratio"]),
            test_ratio=float(cfg["splits"]["test_ratio"]),
            stratify_on=list(cfg["splits"]["stratify_on"]),
            seed=int(cfg["experiment"]["seed"]),
        )
        n_tiny = splits.pop("_n_tiny_strata", 0)
        leakage = assert_no_leakage(splits)   # raises on violation

    summary.add_assertion(
        "no_patient_leakage", leakage["passed"],
        f"train={leakage['n_train']}, val={leakage['n_val']}, test={leakage['n_test']}, "
        f"0 recording IDs shared between any pair of splits",
    )
    logger.info(f"Splits: train={leakage['n_train']}, val={leakage['n_val']}, "
                f"test={leakage['n_test']} recordings")

    quality = stratification_quality(census, splits)
    summary.add_finding("max_class_deviation_pp", quality["max_class_deviation_pp"],
                        "Largest gap between a split's abnormal rate and the global rate")
    summary.add_finding("max_site_deviation_pp", quality["max_site_deviation_pp"])
    summary.add_assertion(
        "stratification_within_2pp", quality["max_class_deviation_pp"] < 2.0,
        f"max class deviation {quality['max_class_deviation_pp']:.2f} pp",
    )
    if n_tiny:
        summary.add_warning(
            f"{n_tiny} (label, subdb) strata contained fewer than 3 recordings and "
            "were assigned entirely to train. This is reported as a minor "
            "deviation from exact joint stratification."
        )

    splits_path = save_json(interim / "splits.json", splits)
    save_json(interim / "split_statistics.json", split_statistics(census, splits))
    summary.add_artifact(splits_path)

    # ---- preprocess + segment -------------------------------------------
    target_sr = int(cfg["preprocessing"]["target_sr"])
    label_by_id = dict(zip(census["recording_id"].astype(str), census["label"]))
    subdb_by_id = dict(zip(census["recording_id"].astype(str), census["subdb"]))
    path_by_id = {}
    for _, row in census.iterrows():
        path_by_id[str(row["recording_id"])] = raw_dir / str(row["subdb"]) / f"{row['recording_id']}.wav"

    index_rows = []
    excluded = {"load_error": 0, "quality": 0, "no_segments": 0}
    spike_counts = []

    for split in ["train", "val", "test"]:
        ids = splits[split]
        hop = hop_for_split(split, cfg["segmentation"])
        logger.info(f"Processing {split}: {len(ids)} recordings (hop {hop} s) ...")

        buffer = []
        with watch.section(f"process_{split}"):
            for position, rec_id in enumerate(ids):
                wav_path = path_by_id.get(rec_id)
                if wav_path is None or not wav_path.exists():
                    excluded["load_error"] += 1
                    continue
                try:
                    signal, file_sr = load_audio(wav_path)
                    processed, meta = preprocess_recording(
                        signal, file_sr, cfg["preprocessing"].to_dict()
                    )
                except Exception as exc:
                    logger.warning(f"  {rec_id}: load/preprocess failed ({exc})")
                    excluded["load_error"] += 1
                    continue

                spike_counts.append(meta["n_spikes_removed"])

                flags = quality_flags(processed, target_sr,
                                      float(cfg["segmentation"]["segment_seconds"]))
                if any(flags.values()):
                    excluded["quality"] += 1
                    continue

                segments, seg_meta = segment_signal(
                    processed, target_sr,
                    segment_seconds=float(cfg["segmentation"]["segment_seconds"]),
                    hop_seconds=hop,
                    pad_policy=str(cfg["segmentation"]["pad_policy"]),
                )
                if len(segments) < int(cfg["segmentation"]["min_segments_per_recording"]):
                    excluded["no_segments"] += 1
                    continue

                for local_index, entry in enumerate(seg_meta):
                    index_rows.append({
                        "global_index": len(buffer) + local_index,
                        "split": split,
                        "recording_id": rec_id,
                        "subdb": subdb_by_id[rec_id],
                        "label": int(label_by_id[rec_id]),
                        "segment_index": entry["index"],
                        "start_s": entry["start_s"],
                        "end_s": entry["end_s"],
                        "n_spikes_removed": meta["n_spikes_removed"],
                    })
                buffer.extend(segments)

                if (position + 1) % 200 == 0:
                    logger.info(f"  {position + 1}/{len(ids)} recordings")

        if not buffer:
            logger.error(f"No segments produced for split '{split}'.")
            summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
            return 1

        array = np.stack(buffer).astype(np.float32)
        path = save_npy(interim / f"segments_{split}.npy", array)
        summary.add_artifact(path, array=array)
        logger.info(f"  {split}: {len(array)} segments of {array.shape[1]} samples")

    # Fix up global indices so they are per-split contiguous.
    index = pd.DataFrame(index_rows)
    index["global_index"] = index.groupby("split").cumcount()
    index_path = interim / "segment_index.csv"
    index.to_csv(index_path, index=False)
    summary.add_artifact(index_path, n_rows=len(index))

    # ---- report --------------------------------------------------------
    total_excluded = sum(excluded.values())
    logger.info(f"Segments total: {len(index)}  (excluded {total_excluded} recordings)")

    summary.touch_split("train", "val", "test")
    summary.add_parameters({
        "target_sr": target_sr,
        "bandpass": [cfg["preprocessing"]["bandpass_low"], cfg["preprocessing"]["bandpass_high"]],
        "segment_seconds": cfg["segmentation"]["segment_seconds"],
        "train_hop_seconds": cfg["segmentation"]["train_hop_seconds"],
        "eval_hop_seconds": cfg["segmentation"]["eval_hop_seconds"],
        "normalize": cfg["preprocessing"]["normalize"],
    })
    summary.add_finding("n_segments_total", int(len(index)))
    for split in ["train", "val", "test"]:
        subset = index[index["split"] == split]
        summary.add_finding(f"n_segments_{split}", int(len(subset)))
        summary.add_finding(f"pct_abnormal_segments_{split}",
                            round(100.0 * float(subset["label"].mean()), 2))
    summary.add_finding("n_recordings_excluded", total_excluded)
    summary.add_finding("exclusion_breakdown", excluded)
    summary.add_finding("mean_spikes_removed", round(float(np.mean(spike_counts)), 2)
                        if spike_counts else 0.0,
                        "Mean friction transients suppressed per recording")

    if total_excluded > 0.05 * len(census):
        summary.add_warning(
            f"{total_excluded} recordings ({100 * total_excluded / len(census):.1f}%) "
            "were excluded. Report this exclusion rate in the paper; a high rate "
            "silently changes the population being evaluated."
        )

    summary.add_table("split_composition", [
        {"split": s,
         "n_recordings": len(splits[s]),
         "n_segments": int((index["split"] == s).sum()),
         "pct_abnormal": round(100.0 * float(index[index["split"] == s]["label"].mean()), 2)}
        for s in ["train", "val", "test"]
    ])

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
