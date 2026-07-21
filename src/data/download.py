"""PhysioNet/CinC 2016 acquisition and raw-data census.

The dataset is distributed as one directory per sub-database, each containing
``.wav`` recordings, ``.hea`` WFDB headers, and a ``REFERENCE.csv`` with one
row per recording: ``<recording_id>,<label>`` where ``-1`` is Normal and ``1``
is Abnormal.

PROFESSOR Q: "Why are there six sub-databases and does it matter?"
A: Each sub-database (training-a .. training-f) was collected by a different
   research group, at a different site, with different stethoscopes and in
   different acoustic environments. Class balance also varies wildly between
   them: training-a is roughly 70% abnormal, training-e is roughly 95% normal.
   This matters enormously. If you split randomly without stratifying on
   sub-database, the model can learn to recognise the *recording device* and
   infer the label from it, because device correlates with site and site
   correlates with prevalence. That is a shortcut, not a diagnosis. We
   stratify splits jointly on (label, sub-database) to prevent it, and we
   report per-sub-database performance so the shortcut would be visible if it
   existed.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a single file with a simple progress readout."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            print(f"\r  {dest.name}: {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    print()
    tmp.rename(dest)
    return dest


def download_dataset(raw_dir: str | Path, base_url: str,
                     subdatabases: List[str], force: bool = False) -> Dict[str, str]:
    """Fetch and unpack the training sub-databases.

    PhysioNet serves both a full zip and per-subdatabase zips. We use the
    per-subdatabase route so a failure mid-way only costs one archive.

    NOTE: if your machine is behind a proxy that blocks physionet.org, download
    the archive manually and place it in ``data/raw/`` - this function will
    detect an already-extracted directory and skip the fetch.
    """
    raw_dir = ensure_dir(raw_dir)
    status: Dict[str, str] = {}

    for subdb in subdatabases:
        target = raw_dir / subdb
        if target.exists() and any(target.glob("*.wav")) and not force:
            status[subdb] = "already_present"
            continue

        url = f"{base_url.rstrip('/')}/{subdb}.zip"
        archive = raw_dir / f"{subdb}.zip"
        try:
            download_file(url, archive, force=force)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw_dir)
            archive.unlink(missing_ok=True)
            status[subdb] = "downloaded"
        except Exception as exc:  # network, 404, proxy, ...
            status[subdb] = f"failed: {type(exc).__name__}: {exc}"

    # PhysioNet nests the folders one level deep in some mirrors; flatten it.
    for subdb in subdatabases:
        nested = raw_dir / subdb / subdb
        if nested.exists():
            for item in nested.iterdir():
                shutil.move(str(item), str(raw_dir / subdb / item.name))
            nested.rmdir()

    return status


def verify_dataset(raw_dir: str | Path, subdatabases: List[str],
                   expected_counts: Optional[Dict[str, int]] = None) -> Dict[str, Dict]:
    """Check every sub-database is present and internally consistent."""
    raw_dir = Path(raw_dir)
    report: Dict[str, Dict] = {}

    for subdb in subdatabases:
        folder = raw_dir / subdb
        entry: Dict[str, object] = {"present": folder.exists()}
        if not folder.exists():
            report[subdb] = entry
            continue

        wavs = sorted(folder.glob("*.wav"))
        heas = sorted(folder.glob("*.hea"))
        reference = folder / "REFERENCE.csv"

        entry.update(
            n_wav=len(wavs),
            n_hea=len(heas),
            has_reference=reference.exists(),
            wav_hea_match=len(wavs) == len(heas),
        )
        if reference.exists():
            df = pd.read_csv(reference, header=None, names=["recording_id", "label"])
            entry["n_reference_rows"] = len(df)
            entry["reference_matches_wav"] = len(df) == len(wavs)
        if expected_counts and subdb in expected_counts:
            entry["expected_n"] = expected_counts[subdb]
            entry["count_ok"] = len(wavs) == expected_counts[subdb]
        report[subdb] = entry

    return report


def build_raw_census(raw_dir: str | Path, subdatabases: List[str],
                     label_map: Dict[str, int]) -> pd.DataFrame:
    """Build one table describing every recording in the dataset.

    Columns: recording_id, subdb, label, label_name, path, duration_s, sr,
    n_samples. Durations are read from the WFDB header where possible (cheap)
    and fall back to reading the wav header only if needed.
    """
    raw_dir = Path(raw_dir)
    rows: List[Dict[str, object]] = []

    for subdb in subdatabases:
        folder = raw_dir / subdb
        reference = folder / "REFERENCE.csv"
        if not reference.exists():
            continue

        labels = pd.read_csv(reference, header=None, names=["recording_id", "label"])
        for _, row in labels.iterrows():
            rec_id = str(row["recording_id"]).strip()
            wav_path = folder / f"{rec_id}.wav"
            if not wav_path.exists():
                continue

            raw_label = int(row["label"])
            mapped = label_map.get(str(raw_label), label_map.get(raw_label, 0 if raw_label < 0 else 1))

            sr, n_samples = _read_header(folder / f"{rec_id}.hea", wav_path)
            rows.append({
                "recording_id": rec_id,
                "subdb": subdb,
                "label": int(mapped),
                "label_name": "Abnormal" if mapped == 1 else "Normal",
                "path": str(wav_path.relative_to(raw_dir.parent.parent))
                        if raw_dir.parent.parent in wav_path.parents else str(wav_path),
                "sr": sr,
                "n_samples": n_samples,
                "duration_s": round(n_samples / sr, 3) if sr else np.nan,
            })

    return pd.DataFrame(rows).sort_values(["subdb", "recording_id"]).reset_index(drop=True)


def _read_header(hea_path: Path, wav_path: Path):
    """Read (sample_rate, n_samples) from a WFDB header, falling back to wav."""
    if hea_path.exists():
        try:
            first = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            parts = first.split()
            # WFDB header line 1: <record> <n_sig> <fs> <n_samples>
            if len(parts) >= 4:
                return int(float(parts[2])), int(float(parts[3]))
        except Exception:
            pass
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return int(info.samplerate), int(info.frames)
    except Exception:
        return 0, 0


def census_statistics(census: pd.DataFrame) -> Dict[str, object]:
    """Summary statistics for the phase-00 report."""
    if census.empty:
        return {"n_recordings": 0}

    per_subdb = (
        census.groupby("subdb")
        .agg(n=("recording_id", "count"),
             n_abnormal=("label", "sum"),
             mean_duration_s=("duration_s", "mean"),
             total_minutes=("duration_s", lambda s: s.sum() / 60.0))
        .reset_index()
    )
    per_subdb["n_normal"] = per_subdb["n"] - per_subdb["n_abnormal"]
    per_subdb["pct_abnormal"] = (100 * per_subdb["n_abnormal"] / per_subdb["n"]).round(1)

    return {
        "n_recordings": int(len(census)),
        "n_normal": int((census["label"] == 0).sum()),
        "n_abnormal": int((census["label"] == 1).sum()),
        "pct_abnormal": round(100.0 * census["label"].mean(), 2),
        "imbalance_ratio": round(
            float((census["label"] == 0).sum()) / max(1, int((census["label"] == 1).sum())), 2
        ),
        "duration_total_hours": round(float(census["duration_s"].sum()) / 3600.0, 2),
        "duration_min_s": round(float(census["duration_s"].min()), 2),
        "duration_median_s": round(float(census["duration_s"].median()), 2),
        "duration_max_s": round(float(census["duration_s"].max()), 2),
        "sample_rates": sorted(census["sr"].unique().tolist()),
        "per_subdatabase": per_subdb.to_dict(orient="records"),
    }
