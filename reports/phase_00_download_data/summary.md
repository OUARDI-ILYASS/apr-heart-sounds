# Phase Summary — `00_download_data`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `8ae8dd4e`  •  **Seed:** `42`  
**Started:** 2026-07-21T19:05:51.818657+00:00  •  **Duration:** 89.78 s  

## Assertions

| Check | Result | Detail |
|---|---|---|
| all_subdatabases_present | ✅ PASS | 6 sub-databases found |
| wav_header_counts_match | ✅ PASS | every .wav has a matching .hea |
| single_sample_rate | ✅ PASS | sample rates present: [2000] |

## Key Findings

- **n_recordings:** `2693`
- **pct_abnormal:** 22.5800 — Class imbalance drives every design decision downstream
- **imbalance_ratio:** 3.4300
- **duration_total_hours:** 16.7600
- **prevalence_spread_pp:** 69.5000 — Difference in abnormal rate between the most and least pathological sub-database. A large spread means site identity is predictive of the label, which is exactly the shortcut the stratified splits prevent.

## Table: per_subdatabase

| subdb | n | n_abnormal | pct_abnormal | total_minutes | mean_duration_s | n_normal |
|---|---|---|---|---|---|---|
| training-a | 409 | 292 | 71.4000 | 222.1351 | 32.5871 | 117 |
| training-b | 490 | 104 | 21.2000 | 65.1705 | 7.9801 | 386 |
| training-c | 31 | 24 | 77.4000 | 25.5413 | 49.4349 | 7 |
| training-d | 55 | 28 | 50.9000 | 13.8854 | 15.1477 | 27 |
| training-e | 1594 | 126 | 7.9000 | 616.1284 | 23.1918 | 1468 |
| training-f | 114 | 34 | 29.8000 | 62.9236 | 33.1177 | 80 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/data/interim/raw_census.csv | — | — | 0.2160 |
| /workspace/apr-heart-sounds/data/interim/raw_census_stats.json | — | — | 0.0020 |

## Parameters Used

```yaml
download_status: {'training-c': 'already_present', 'training-d': 'already_present', 'training-a': 'already_present', 'training-b': 'already_present', 'training-f': 'already_present', 'training-e': 'already_present'}
subdatabases: ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
raw_dir: /workspace/apr-heart-sounds/data/raw/physionet2016
```

## Notes

- Abnormal prevalence ranges over 70 percentage points across sub-databases. Splits are stratified jointly on (label, subdb) in phase 01, and per-site metrics are reported in phase 06 so any site shortcut would be visible.

## Timing Breakdown

- `_total`: 89.78 s
- `census`: 89.17 s
- `verify`: 0.21 s
- `download`: 0.02 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `01_preprocess_audio`
