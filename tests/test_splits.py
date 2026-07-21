"""Split integrity - the tests that matter most.

If any of these fail, every number in the paper is invalid. They run in
milliseconds and there is no excuse for not running them.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.splits import (
    make_splits, assert_no_leakage, split_statistics, stratification_quality,
    expand_to_segments, leave_one_subdb_out,
)


def test_splits_are_disjoint(toy_census):
    splits = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    splits.pop("_n_tiny_strata", None)
    report = assert_no_leakage(splits)
    assert report["passed"]
    assert sum(report["overlaps"].values()) == 0


def test_all_recordings_assigned(toy_census):
    splits = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    splits.pop("_n_tiny_strata", None)
    assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert assigned == set(toy_census["recording_id"])


def test_leakage_is_detected():
    """The guard must actually fire - a check that never fails is not a check."""
    bad = {"train": ["a", "b", "c"], "val": ["c", "d"], "test": ["e"]}
    with pytest.raises(AssertionError, match="PATIENT LEAKAGE"):
        assert_no_leakage(bad)


def test_class_stratification_holds(toy_census):
    splits = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    splits.pop("_n_tiny_strata", None)
    quality = stratification_quality(toy_census, splits)
    # Within 10 percentage points on a 120-recording toy set; the real dataset
    # (3000+ recordings) achieves far tighter agreement.
    assert quality["max_class_deviation_pp"] < 10.0


def test_site_stratification_holds(toy_census):
    splits = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    splits.pop("_n_tiny_strata", None)
    stats = split_statistics(toy_census, splits)
    for split in ["train", "val", "test"]:
        assert len(stats[split]["subdb_distribution"]) >= 2


def test_splits_are_deterministic(toy_census):
    a = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    b = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    assert a["train"] == b["train"] and a["test"] == b["test"]


def test_different_seeds_give_different_splits(toy_census):
    a = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=1)
    b = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=2)
    assert a["train"] != b["train"]


def test_ratios_must_sum_to_one(toy_census):
    with pytest.raises(ValueError):
        make_splits(toy_census, 0.7, 0.7, 0.7, ["label"], seed=42)


def test_segment_expansion_preserves_recording_split(toy_census):
    import pandas as pd

    splits = make_splits(toy_census, 0.7, 0.15, 0.15, ["label", "subdb"], seed=42)
    splits.pop("_n_tiny_strata", None)
    segments = pd.DataFrame({
        "recording_id": np.repeat(toy_census["recording_id"].values, 3),
        "segment_index": np.tile([0, 1, 2], len(toy_census)),
    })
    expanded = expand_to_segments(segments, splits)
    # Every segment of a recording must land in that recording's split.
    per_recording = expanded.groupby("recording_id")["split"].nunique()
    assert (per_recording == 1).all()


def test_leave_one_subdb_out_holds_out_everything(toy_census):
    splits = leave_one_subdb_out(toy_census, "training-a", seed=42)
    held_out = set(toy_census.loc[toy_census["subdb"] == "training-a", "recording_id"])
    assert set(splits["test"]) == held_out
    assert not (set(splits["train"]) & held_out)
