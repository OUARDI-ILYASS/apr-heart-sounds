# Phase Summary — `00_download_data`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T21:14:15.224070+00:00  •  **Duration:** 2766.19 s  

## Assertions

| Check | Result | Detail |
|---|---|---|
| all_subdatabases_present | ✅ PASS | 6 sub-databases found |
| wav_header_counts_match | ✅ PASS | every .wav has a matching .hea |
| single_sample_rate | ✅ PASS | sample rates present: [2000] |

## Key Findings

- **n_recordings:** `3240`
- **pct_abnormal:** 20.5200 — Class imbalance drives every design decision downstream
- **imbalance_ratio:** 3.8700
- **duration_total_hours:** 20.2200
- **prevalence_spread_pp:** 68.9000 — Difference in abnormal rate between the most and least pathological sub-database. A large spread means site identity is predictive of the label, which is exactly the shortcut the stratified splits prevent.

## Table: per_subdatabase

| n_normal | n | pct_abnormal | total_minutes | mean_duration_s | subdb | n_abnormal |
|---|---|---|---|---|---|---|
| 117 | 409 | 71.4000 | 222.1351 | 32.5871 | training-a | 292 |
| 386 | 490 | 21.2000 | 65.1705 | 7.9801 | training-b | 104 |
| 7 | 31 | 77.4000 | 25.5413 | 49.4349 | training-c | 24 |
| 27 | 55 | 50.9000 | 13.8854 | 15.1477 | training-d | 28 |
| 1958 | 2141 | 8.5000 | 823.2785 | 23.0718 | training-e | 183 |
| 80 | 114 | 29.8000 | 62.9236 | 33.1177 | training-f | 34 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/data/interim/raw_census.csv | — | — | 0.2600 |
| /workspace/apr-heart-sounds/data/interim/raw_census_stats.json | — | — | 0.0020 |

## Parameters Used

```yaml
download_status: {'training-d': 'downloaded', 'training-c': 'downloaded', 'training-f': 'downloaded', 'training-b': 'downloaded', 'training-a': 'downloaded', 'training-e': 'downloaded'}
subdatabases: ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
raw_dir: /workspace/apr-heart-sounds/data/raw/physionet2016
```

## Notes

- Abnormal prevalence ranges over 69 percentage points across sub-databases. Splits are stratified jointly on (label, subdb) in phase 01, and per-site metrics are reported in phase 06 so any site shortcut would be visible.

## Timing Breakdown

- `_total`: 2766.19 s
- `download`: 2705.09 s
- `census`: 60.89 s
- `verify`: 0.14 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `01_preprocess_audio`
