"""Subject-independent, doubly-stratified train/val/test splitting.

This module is short but it is the single most consequential piece of code in
the project. Everything downstream is only as trustworthy as the splits.

Three properties are enforced and then *asserted*:

1. **Recording-level disjointness.** No recording contributes segments to more
   than one split. Violating this is the classic way to publish a 99% accurate
   heart-sound classifier that does nothing.
2. **Class stratification.** Each split has the same Normal/Abnormal ratio as
   the full dataset.
3. **Sub-database stratification.** Each split has the same site mix. Without
   this, the test set can end up dominated by one clinic, and you are measuring
   cross-site transfer while believing you are measuring pathology detection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def make_splits(census: pd.DataFrame, train_ratio: float, val_ratio: float,
                test_ratio: float, stratify_on: List[str],
                seed: int = 42) -> Dict[str, List[str]]:
    """Split recordings into train/val/test with joint stratification.

    Implementation note: rather than using sklearn's ``train_test_split`` twice
    (which only stratifies on one variable), we form the *cross product* of the
    stratification columns into a single composite stratum key, then split
    within each stratum independently. This gives exact joint stratification.

    Small strata (< 3 recordings) cannot be split three ways; they are assigned
    to train, and the number of such strata is returned for the summary so the
    imperfection is visible rather than hidden.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    df = census.copy()

    missing = [c for c in stratify_on if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot stratify on missing columns: {missing}")

    df["_stratum"] = df[stratify_on].astype(str).agg("|".join, axis=1)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    tiny_strata = 0

    for stratum, group in df.groupby("_stratum", sort=True):
        ids = group["recording_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)

        if n < 3:
            # Too small to divide three ways without emptying a split.
            splits["train"].extend(ids)
            tiny_strata += 1
            continue

        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        # Guarantee at least one recording lands in test.
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1

        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train: n_train + n_val])
        splits["test"].extend(ids[n_train + n_val:])

    for key in splits:
        splits[key] = sorted(splits[key])
    splits["_n_tiny_strata"] = tiny_strata      # type: ignore[assignment]
    return splits


def assert_no_leakage(splits: Dict[str, List[str]]) -> Dict[str, object]:
    """Hard check that no recording ID appears in more than one split.

    Returns a report dict. Raises ``AssertionError`` on violation - the whole
    run should die here rather than produce a beautiful, meaningless number.
    """
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    total_overlap = sum(len(v) for v in overlaps.values())

    report = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_total": len(train | val | test),
        "overlaps": {k: len(v) for k, v in overlaps.items()},
        "overlap_examples": {k: v[:5] for k, v in overlaps.items() if v},
        "passed": total_overlap == 0,
    }

    if total_overlap > 0:
        raise AssertionError(
            f"PATIENT LEAKAGE DETECTED: {total_overlap} recording IDs appear in "
            f"more than one split. Details: {report['overlap_examples']}. "
            "Every metric computed from these splits would be invalid."
        )
    return report


def split_statistics(census: pd.DataFrame, splits: Dict[str, List[str]]
                     ) -> Dict[str, Dict[str, object]]:
    """Per-split class balance and site composition, for the phase summary."""
    stats: Dict[str, Dict[str, object]] = {}
    for split in ["train", "val", "test"]:
        subset = census[census["recording_id"].isin(splits[split])]
        if subset.empty:
            stats[split] = {"n": 0}
            continue
        stats[split] = {
            "n_recordings": int(len(subset)),
            "n_normal": int((subset["label"] == 0).sum()),
            "n_abnormal": int((subset["label"] == 1).sum()),
            "pct_abnormal": round(100.0 * float(subset["label"].mean()), 2),
            "total_minutes": round(float(subset["duration_s"].sum()) / 60.0, 1),
            "subdb_distribution": subset["subdb"].value_counts().to_dict(),
        }
    return stats


def stratification_quality(census: pd.DataFrame, splits: Dict[str, List[str]]
                           ) -> Dict[str, float]:
    """How well did stratification hold?

    Reports the maximum absolute deviation, across splits, between the split's
    abnormal rate and the global abnormal rate. Under 2 percentage points is
    good; a large value means a stratum was too small to divide cleanly and
    should be reported as a limitation.
    """
    global_rate = float(census["label"].mean())
    deviations = {}
    for split in ["train", "val", "test"]:
        subset = census[census["recording_id"].isin(splits[split])]
        if subset.empty:
            continue
        deviations[split] = abs(float(subset["label"].mean()) - global_rate)

    site_dev = {}
    global_sites = census["subdb"].value_counts(normalize=True)
    for split in ["train", "val", "test"]:
        subset = census[census["recording_id"].isin(splits[split])]
        if subset.empty:
            continue
        split_sites = subset["subdb"].value_counts(normalize=True)
        aligned = global_sites.align(split_sites, fill_value=0.0)
        site_dev[split] = float(np.abs(aligned[0] - aligned[1]).max())

    return {
        "global_abnormal_rate": round(global_rate, 4),
        "max_class_deviation_pp": round(100 * max(deviations.values()), 3) if deviations else 0.0,
        "per_split_class_deviation_pp": {k: round(100 * v, 3) for k, v in deviations.items()},
        "max_site_deviation_pp": round(100 * max(site_dev.values()), 3) if site_dev else 0.0,
    }


def expand_to_segments(segment_index: pd.DataFrame,
                       splits: Dict[str, List[str]]) -> pd.DataFrame:
    """Attach a ``split`` column to a segment table via its recording ID."""
    mapping: Dict[str, str] = {}
    for split in ["train", "val", "test"]:
        for rec_id in splits[split]:
            mapping[rec_id] = split
    out = segment_index.copy()
    out["split"] = out["recording_id"].map(mapping)
    return out


def leave_one_subdb_out(census: pd.DataFrame, held_out: str,
                        val_ratio: float = 0.15,
                        seed: int = 42) -> Dict[str, List[str]]:
    """Cross-site stress-test split: one whole sub-database becomes the test set.

    Used in the robustness experiment. This is a *harder* setting than the main
    split because the test site's equipment, noise floor and class prevalence
    are all unseen. A large drop here is honest evidence about generalisation
    and is reported as such rather than buried.
    """
    rng = np.random.default_rng(seed)
    test_ids = census.loc[census["subdb"] == held_out, "recording_id"].tolist()
    remaining = census.loc[census["subdb"] != held_out, "recording_id"].tolist()
    rng.shuffle(remaining)
    n_val = int(round(val_ratio * len(remaining)))
    return {
        "train": sorted(remaining[n_val:]),
        "val": sorted(remaining[:n_val]),
        "test": sorted(test_ids),
    }
