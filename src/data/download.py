"""PhysioNet/CinC 2016 acquisition and raw-data census.

The dataset is distributed as one directory per sub-database, each containing
``.wav`` recordings, ``.hea`` WFDB headers, and a ``REFERENCE.csv`` with one
row per recording: ``<recording_id>,<label>`` where ``-1`` is Normal and ``1``
is Abnormal.

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
import wfdb



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


import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Dict

def download_dataset(raw_dir: str | Path, base_url: str,
                     subdatabases: List[str], force: bool = False) -> Dict[str, str]:
    """Fetch the training sub-databases concurrently using system wget."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}

    def _download_subset(subdb: str) -> tuple[str, str]:
        """Helper function to download a single subset."""
        target = raw_dir / subdb
        if target.exists() and any(target.glob("*.wav")) and not force:
            return subdb, "already_present"

        url = f"{base_url.rstrip('/')}/{subdb}/"
        print(f"Starting download: {subdb}...")
        
        try:
            # Added '-q' to prevent concurrent progress bars from mangling the console
            subprocess.run([
                "wget", "-q",
                "-r", "-N", "-c", "-np", "-nH",
                "--cut-dirs=3", 
                "-P", str(raw_dir), 
                url
            ], check=True)
            return subdb, "downloaded"
            
        except subprocess.CalledProcessError as exc:
            return subdb, f"failed: wget error {exc.returncode}"
        except FileNotFoundError:
            return subdb, "failed: wget is not installed or not in PATH"

    # Execute downloads in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks to the thread pool
        futures = {executor.submit(_download_subset, subdb): subdb for subdb in subdatabases}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            subdb, stat = future.result()
            status[subdb] = stat
            print(f"[{stat.upper()}] {subdb}")

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
