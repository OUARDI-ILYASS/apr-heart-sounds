"""Phase summaries - the state tracker that makes the pipeline auditable.

Every script ends by writing ``reports/<phase>/summary.json`` (machine-readable
contract) and ``reports/<phase>/summary.md`` (what a human reads). A phase that
did not write a summary is considered not to have run.

Three things live in a summary and each has a purpose:

* **artifacts_written** - with shapes and dtypes. This is the handoff contract:
  the next phase can verify its inputs are what it expects without loading them.
* **assertions** - named PASS/FAIL checks (e.g. no patient leakage). These turn
  methodological promises into artifacts you can point at.
* **claim_verdicts** - each experimental claim C1..C5 marked supported / weak /
  contradicted, with the evidence that decided it.
"""

from __future__ import annotations

import getpass
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import ensure_dir, save_json, describe_array, load_json

# Valid verdict vocabulary - deliberately small so summaries stay comparable.
VERDICTS = {"supported", "weak", "contradicted", "pending", "not_applicable"}


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def environment_info() -> Dict[str, Any]:
    """Capture everything needed to reproduce a run on another machine."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "packages": {},
    }
    for module in ["numpy", "scipy", "sklearn", "pandas", "librosa", "pywt",
                   "torch", "shap", "matplotlib", "joblib"]:
        try:
            mod = __import__(module)
            info["packages"][module] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["packages"][module] = None

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "cuda": torch.version.cuda,
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2
                ),
            }
        else:
            info["gpu"] = None
    except ImportError:
        info["gpu"] = None
    return info


class PhaseSummary:
    """Accumulates everything a phase should report, then writes JSON + Markdown.

    Usage::

        summary = PhaseSummary("02_extract_features", cfg)
        summary.add_input("data/interim/segments", n_files=18432)
        summary.add_artifact("data/processed/mfcc/features_train.npy", array=X)
        summary.add_parameters({"n_mfcc": 13})
        summary.add_finding("mfcc_dim", 234, "Feature vector length per segment")
        summary.add_assertion("no_patient_leakage", True, "0 shared recording IDs")
        summary.write(reports_dir)
    """

    def __init__(self, phase: str, cfg: Any = None, next_phase: Optional[str] = None):
        self.phase = phase
        self.next_phase = next_phase
        self.started = datetime.now(timezone.utc)
        self.status = "running"
        self.config_hash = cfg.get("_config_hash") if cfg is not None else None
        self.experiment_name = (
            cfg.get("experiment", {}).get("name") if cfg is not None else None
        )
        self.seed = cfg.get("experiment", {}).get("seed") if cfg is not None else None

        self.inputs: List[Dict[str, Any]] = []
        self.artifacts: List[Dict[str, Any]] = []
        self.parameters: Dict[str, Any] = {}
        self.findings: Dict[str, Any] = {}
        self.finding_notes: Dict[str, str] = {}
        self.assertions: List[Dict[str, Any]] = []
        self.claim_verdicts: Dict[str, Dict[str, Any]] = {}
        self.warnings: List[str] = []
        self.splits_touched: List[str] = []
        self.timings: Dict[str, float] = {}
        self.tables: Dict[str, Any] = {}
        self.notes: List[str] = []

    # -- recording ---------------------------------------------------------
    def add_input(self, path: str | Path, **meta: Any) -> None:
        self.inputs.append({"path": str(path), **meta})

    def add_artifact(self, path: str | Path, array: Any = None, **meta: Any) -> None:
        entry: Dict[str, Any] = {"path": str(path)}
        if array is not None:
            try:
                entry.update(describe_array(array))
            except Exception:
                pass
        p = Path(path)
        if p.exists() and p.is_file():
            entry.setdefault("mb", round(p.stat().st_size / 1024 ** 2, 3))
        entry.update(meta)
        self.artifacts.append(entry)

    def add_parameters(self, params: Dict[str, Any]) -> None:
        self.parameters.update(params)

    def add_finding(self, key: str, value: Any, note: str = "") -> None:
        self.findings[key] = value
        if note:
            self.finding_notes[key] = note

    def add_table(self, name: str, rows: Any) -> None:
        """Attach a small result table (list of dicts) for the Markdown render."""
        self.tables[name] = rows

    def add_assertion(self, name: str, passed: bool, detail: str = "") -> None:
        self.assertions.append(
            {"name": name, "result": "PASS" if passed else "FAIL", "detail": detail}
        )

    def add_claim_verdict(self, claim_id: str, statement: str, verdict: str,
                          evidence: str = "", detail: str = "") -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"Verdict must be one of {sorted(VERDICTS)}, got '{verdict}'")
        self.claim_verdicts[claim_id] = {
            "statement": statement, "verdict": verdict,
            "evidence": evidence, "detail": detail,
        }

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def touch_split(self, *splits: str) -> None:
        """Declare which data splits this phase read. Audited across phases."""
        for split in splits:
            if split not in self.splits_touched:
                self.splits_touched.append(split)

    def set_timings(self, timings: Dict[str, float]) -> None:
        self.timings = timings

    # -- output ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        ended = datetime.now(timezone.utc)
        return {
            "phase": self.phase,
            "experiment": self.experiment_name,
            "status": self.status,
            "timestamp_start": self.started.isoformat(),
            "timestamp_end": ended.isoformat(),
            "duration_seconds": round((ended - self.started).total_seconds(), 2),
            "config_hash": self.config_hash,
            "seed": self.seed,
            "environment": environment_info(),
            "splits_touched": self.splits_touched,
            "inputs_loaded": self.inputs,
            "artifacts_written": self.artifacts,
            "parameters": self.parameters,
            "key_findings": self.findings,
            "finding_notes": self.finding_notes,
            "tables": self.tables,
            "assertions": self.assertions,
            "claim_verdicts": self.claim_verdicts,
            "warnings": self.warnings,
            "notes": self.notes,
            "timings_seconds": self.timings,
            "next_phase": self.next_phase,
        }

    def write(self, reports_dir: str | Path, status: str = "success") -> Dict[str, Path]:
        self.status = status
        out_dir = ensure_dir(Path(reports_dir) / f"phase_{self.phase}")
        payload = self.to_dict()
        json_path = save_json(out_dir / "summary.json", payload)
        md_path = out_dir / "summary.md"
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _md_table(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "_(none)_\n"
    if isinstance(rows, dict):
        rows = [{"key": k, "value": v} for k, v in rows.items()]
    columns = columns or list({k for row in rows for k in row})
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            elif isinstance(value, (list, tuple)):
                cells.append("×".join(str(v) for v in value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body]) + "\n"


_VERDICT_ICON = {
    "supported": "✅", "weak": "⚠️", "contradicted": "❌",
    "pending": "⏳", "not_applicable": "–",
}


def render_markdown(payload: Dict[str, Any]) -> str:
    """Render a summary dict as a readable Markdown report."""
    status_icon = {"success": "✅", "failed": "❌", "running": "⏳"}.get(payload["status"], "•")
    lines: List[str] = []
    add = lines.append

    add(f"# Phase Summary — `{payload['phase']}`")
    add("")
    add(f"{status_icon} **Status:** {payload['status']}  ")
    add(f"**Experiment:** `{payload.get('experiment')}`  ")
    add(f"**Config hash:** `{payload.get('config_hash')}`  •  **Seed:** `{payload.get('seed')}`  ")
    add(f"**Started:** {payload['timestamp_start']}  •  **Duration:** {payload['duration_seconds']} s  ")
    if payload.get("splits_touched"):
        add(f"**Data splits read:** `{', '.join(payload['splits_touched'])}`  ")
    add("")

    if payload.get("warnings"):
        add("> [!WARNING]")
        for warning in payload["warnings"]:
            add(f"> - {warning}")
        add("")

    # Assertions first - they are pass/fail gates
    if payload.get("assertions"):
        add("## Assertions")
        add("")
        rows = [
            {"Check": a["name"],
             "Result": ("✅ PASS" if a["result"] == "PASS" else "❌ FAIL"),
             "Detail": a.get("detail", "")}
            for a in payload["assertions"]
        ]
        add(_md_table(rows, ["Check", "Result", "Detail"]))

    if payload.get("claim_verdicts"):
        add("## Claim Verdicts")
        add("")
        rows = [
            {"Claim": cid,
             "Statement": v["statement"],
             "Verdict": f"{_VERDICT_ICON.get(v['verdict'], '')} {v['verdict']}",
             "Evidence": v.get("evidence", "")}
            for cid, v in payload["claim_verdicts"].items()
        ]
        add(_md_table(rows, ["Claim", "Statement", "Verdict", "Evidence"]))

    if payload.get("key_findings"):
        add("## Key Findings")
        add("")
        notes = payload.get("finding_notes", {})
        for key, value in payload["key_findings"].items():
            if isinstance(value, dict):
                add(f"- **{key}:**")
                for sub_key, sub_value in value.items():
                    formatted = f"{sub_value:.4f}" if isinstance(sub_value, float) else sub_value
                    add(f"  - `{sub_key}`: {formatted}")
            elif isinstance(value, float):
                add(f"- **{key}:** {value:.4f}" + (f" — {notes[key]}" if key in notes else ""))
            else:
                add(f"- **{key}:** `{value}`" + (f" — {notes[key]}" if key in notes else ""))
        add("")

    for name, rows in (payload.get("tables") or {}).items():
        add(f"## Table: {name}")
        add("")
        add(_md_table(rows))

    if payload.get("artifacts_written"):
        add("## Artifacts Written")
        add("")
        rows = [
            {"Path": a["path"],
             "Shape": a.get("shape", "—"),
             "Dtype": a.get("dtype", "—"),
             "MB": a.get("mb", "—")}
            for a in payload["artifacts_written"]
        ]
        add(_md_table(rows, ["Path", "Shape", "Dtype", "MB"]))

    if payload.get("inputs_loaded"):
        add("## Inputs Loaded")
        add("")
        for item in payload["inputs_loaded"]:
            extra = ", ".join(f"{k}={v}" for k, v in item.items() if k != "path")
            add(f"- `{item['path']}`" + (f" ({extra})" if extra else ""))
        add("")

    if payload.get("parameters"):
        add("## Parameters Used")
        add("")
        add("```yaml")
        for key, value in payload["parameters"].items():
            add(f"{key}: {value}")
        add("```")
        add("")

    if payload.get("notes"):
        add("## Notes")
        add("")
        for note in payload["notes"]:
            add(f"- {note}")
        add("")

    if payload.get("timings_seconds"):
        add("## Timing Breakdown")
        add("")
        for key, value in sorted(payload["timings_seconds"].items(),
                                 key=lambda kv: -float(kv[1])):
            add(f"- `{key}`: {float(value):.2f} s")
        add("")

    env = payload.get("environment", {})
    add("<details><summary>Environment</summary>")
    add("")
    add(f"- Python {env.get('python')} on {env.get('platform')}")
    add(f"- Git commit: `{env.get('git_commit')}` (dirty: {env.get('git_dirty')})")
    if env.get("gpu"):
        gpu = env["gpu"]
        add(f"- GPU: {gpu['name']} ×{gpu['count']}, CUDA {gpu['cuda']}, {gpu['total_memory_gb']} GB")
    else:
        add("- GPU: none detected")
    packages = {k: v for k, v in (env.get("packages") or {}).items() if v}
    add(f"- Packages: {', '.join(f'{k}=={v}' for k, v in packages.items())}")
    add("")
    add("</details>")
    add("")
    if payload.get("next_phase"):
        add(f"➡️ **Next phase:** `{payload['next_phase']}`")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Cross-phase aggregation
# --------------------------------------------------------------------------- #
def collect_summaries(reports_dir: str | Path) -> List[Dict[str, Any]]:
    """Load every phase summary found, ordered by phase name."""
    out = []
    for path in sorted(Path(reports_dir).glob("phase_*/summary.json")):
        try:
            out.append(load_json(path))
        except Exception:
            continue
    return out


def render_pipeline_status(summaries: List[Dict[str, Any]]) -> str:
    """Build the top-level PIPELINE_STATUS.md dashboard."""
    lines: List[str] = ["# Pipeline Status", ""]
    if not summaries:
        lines.append("_No phases have been run yet._")
        return "\n".join(lines) + "\n"

    lines.append(f"_Generated from {len(summaries)} phase summaries._")
    lines.append("")
    lines.append("## Phase Overview")
    lines.append("")
    rows = []
    for summary in summaries:
        failed = [a["name"] for a in summary.get("assertions", []) if a["result"] == "FAIL"]
        rows.append({
            "Phase": summary["phase"],
            "Status": {"success": "✅", "failed": "❌"}.get(summary["status"], "⏳"),
            "Duration": f"{summary.get('duration_seconds', 0):.0f}s",
            "Artifacts": len(summary.get("artifacts_written", [])),
            "Warnings": len(summary.get("warnings", [])),
            "Failed checks": ", ".join(failed) if failed else "—",
        })
    lines.append(_md_table(rows, ["Phase", "Status", "Duration", "Artifacts",
                                 "Warnings", "Failed checks"]))

    # Claim roll-up across all phases
    claims: Dict[str, Dict[str, Any]] = {}
    for summary in summaries:
        for cid, verdict in (summary.get("claim_verdicts") or {}).items():
            claims[cid] = {**verdict, "phase": summary["phase"]}
    if claims:
        lines.append("## Claim–Evidence Roll-up")
        lines.append("")
        rows = [
            {"Claim": cid,
             "Statement": v["statement"],
             "Verdict": f"{_VERDICT_ICON.get(v['verdict'], '')} {v['verdict']}",
             "Evidence": v.get("evidence", ""),
             "Decided in": v["phase"]}
            for cid, v in sorted(claims.items())
        ]
        lines.append(_md_table(rows, ["Claim", "Statement", "Verdict",
                                     "Evidence", "Decided in"]))

    # Leakage audit: which phases read the test split?
    test_readers = [s["phase"] for s in summaries if "test" in (s.get("splits_touched") or [])]
    lines.append("## Test-Set Audit")
    lines.append("")
    lines.append(
        f"Phases that read the **test** split: "
        f"{', '.join(f'`{p}`' for p in test_readers) if test_readers else '_none_'}"
    )
    lines.append("")
    if len(test_readers) > 3:
        lines.append(
            "> [!WARNING]\n"
            "> More than three phases touched the test split. Model selection must "
            "happen on validation data only; verify none of these phases used test "
            "results to choose hyperparameters."
        )
        lines.append("")

    all_warnings = [(s["phase"], w) for s in summaries for w in (s.get("warnings") or [])]
    if all_warnings:
        lines.append("## All Warnings")
        lines.append("")
        for phase, warning in all_warnings:
            lines.append(f"- `{phase}`: {warning}")
        lines.append("")

    return "\n".join(lines) + "\n"
