"""I/O guards, config validation and phase-summary behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.io import (
    save_npy, load_npy, save_json, load_json, require_artifacts, describe_array,
    NumpyJSONEncoder,
)
from src.config.loader import deep_merge, config_diff, hash_config, ConfigDict, apply_overrides
from src.config.schema import validate_config, ConfigValidationError, derived_shapes, usable_fft_bins
from src.utils.summary import PhaseSummary, render_markdown, render_pipeline_status


def test_save_npy_refuses_nans(tmp_path):
    array = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        save_npy(tmp_path / "bad.npy", array)


def test_save_npy_refuses_infs(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        save_npy(tmp_path / "bad.npy", np.array([1.0, np.inf]))


def test_npy_roundtrip(tmp_path):
    array = np.random.rand(10, 5).astype(np.float32)
    path = save_npy(tmp_path / "nested" / "x.npy", array)
    assert path.exists()                      # parent dirs auto-created
    np.testing.assert_array_equal(load_npy(path), array)


def test_missing_artifact_names_the_producing_phase(tmp_path):
    with pytest.raises(FileNotFoundError, match="02_extract_features"):
        load_npy(tmp_path / "data" / "processed" / "mfcc" / "features_train.npy")


def test_require_artifacts_lists_everything_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        require_artifacts([tmp_path / "a.npy", tmp_path / "b.npy"], phase="test")
    assert "a.npy" in str(exc.value) and "b.npy" in str(exc.value)


def test_json_encoder_handles_numpy(tmp_path):
    payload = {"a": np.int64(3), "b": np.float32(1.5), "c": np.array([1, 2]),
               "d": np.bool_(True), "e": Path("/tmp/x")}
    path = save_json(tmp_path / "x.json", payload)
    loaded = load_json(path)
    assert loaded["a"] == 3 and loaded["b"] == 1.5 and loaded["c"] == [1, 2]


def test_json_encoder_nulls_non_finite():
    text = json.dumps({"x": np.float32("nan")}, cls=NumpyJSONEncoder)
    assert json.loads(text)["x"] is None


def test_describe_array_reports_nan_count():
    described = describe_array(np.array([1.0, np.nan, np.inf]))
    assert described["n_nan"] == 1 and described["n_inf"] == 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_deep_merge_only_replaces_declared_leaves():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 99}})
    assert merged == {"a": {"b": 99, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1            # base untouched


def test_config_diff_finds_exact_changes():
    base = {"a": {"b": 1, "c": 2}}
    other = {"a": {"b": 99, "c": 2}}
    diff = config_diff(base, other)
    assert list(diff) == ["a.b"]
    assert diff["a.b"] == {"base": 1, "current": 99}


def test_hash_is_stable_under_key_reordering():
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})


def test_hash_changes_when_a_value_changes():
    assert hash_config({"a": 1}) != hash_config({"a": 2})


def test_attribute_access_and_dotted_paths():
    cfg = ConfigDict({"a": {"b": {"c": 42}}})
    assert cfg.a.b.c == 42
    assert cfg.get_path("a.b.c") == 42
    cfg.set_path("a.b.d", 7)
    assert cfg.a.b.d == 7


def test_cli_overrides_coerce_types():
    cfg = ConfigDict({"n": 1, "flag": False, "name": "x"})
    apply_overrides(cfg, ["n=5", "flag=true", "name=hello"])
    assert cfg.n == 5 and cfg.flag is True and cfg.name == "hello"


def test_real_config_validates(cfg):
    validate_config(cfg.to_dict())        # must not raise


def test_validator_catches_the_mel_resolution_trap(cfg):
    broken = cfg.to_dict()
    broken["features"]["n_mels"] = 200
    with pytest.raises(ConfigValidationError, match="usable FFT bins"):
        validate_config(broken)


def test_validator_rejects_segment_level_splits(cfg):
    broken = cfg.to_dict()
    broken["splits"]["level"] = "segment"
    with pytest.raises(ConfigValidationError, match="leaks"):
        validate_config(broken)


def test_validator_rejects_non_frequency_wavelet_ordering(cfg):
    broken = cfg.to_dict()
    broken["features"]["pwp"]["node_order"] = "natural"
    with pytest.raises(ConfigValidationError, match="scrambles"):
        validate_config(broken)


def test_validator_rejects_bad_split_ratios(cfg):
    broken = cfg.to_dict()
    broken["splits"]["train_ratio"] = 0.9
    with pytest.raises(ConfigValidationError, match="sum to 1"):
        validate_config(broken)


def test_validator_rejects_too_many_mfcc_coefficients(cfg):
    broken = cfg.to_dict()
    broken["features"]["mfcc"]["n_mfcc"] = 999
    with pytest.raises(ConfigValidationError):
        validate_config(broken)


def test_derived_shapes_match_documented_values(cfg):
    shapes = derived_shapes(cfg.to_dict())
    assert shapes["segment_samples"] == 6000          # 3 s at 2 kHz
    assert shapes["logmel_shape"][0] == cfg["features"]["n_mels"]
    # 13 coefficients x 3 streams x 6 statistics
    assert shapes["mfcc_dim"] == 13 * 3 * 6
    assert shapes["usable_fft_bins_in_band"] >= cfg["features"]["n_mels"]


def test_usable_fft_bins_arithmetic():
    # sr=2000, n_fft=256 -> 7.8125 Hz per bin; 25-400 Hz spans ~48 bins.
    assert 45 <= usable_fft_bins(2000, 256, 25, 400) <= 52


# --------------------------------------------------------------------------- #
# Phase summaries
# --------------------------------------------------------------------------- #
def test_summary_writes_both_formats(tmp_path, cfg):
    summary = PhaseSummary("99_test", cfg)
    summary.add_finding("x", 1.0)
    summary.add_assertion("check", True, "fine")
    paths = summary.write(tmp_path)
    assert paths["json"].exists() and paths["markdown"].exists()
    payload = load_json(paths["json"])
    assert payload["status"] == "success"
    assert payload["config_hash"] == cfg["_config_hash"]


def test_summary_rejects_unknown_verdicts(cfg):
    summary = PhaseSummary("99_test", cfg)
    with pytest.raises(ValueError, match="Verdict must be"):
        summary.add_claim_verdict("C1", "something", "probably_true")


def test_summary_records_splits_touched(cfg):
    summary = PhaseSummary("99_test", cfg)
    summary.touch_split("train", "val")
    summary.touch_split("train")            # idempotent
    assert summary.splits_touched == ["train", "val"]


def test_markdown_render_includes_failures(cfg):
    summary = PhaseSummary("99_test", cfg)
    summary.add_assertion("bad_check", False, "it broke")
    text = render_markdown(summary.to_dict())
    assert "bad_check" in text and "FAIL" in text


def test_pipeline_status_audits_test_split_usage(cfg):
    summaries = [
        {"phase": "04_train", "status": "success", "duration_seconds": 1,
         "splits_touched": ["train", "val"], "artifacts_written": [], "warnings": []},
        {"phase": "06_eval", "status": "success", "duration_seconds": 1,
         "splits_touched": ["test"], "artifacts_written": [], "warnings": []},
    ]
    text = render_pipeline_status(summaries)
    assert "Test-Set Audit" in text
    assert "06_eval" in text
