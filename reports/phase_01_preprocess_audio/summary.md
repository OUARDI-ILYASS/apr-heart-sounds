# Phase Summary — `01_preprocess_audio`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T22:00:28.200491+00:00  •  **Duration:** 135.31 s  
**Data splits read:** `train, val, test`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| no_patient_leakage | ✅ PASS | train=2269, val=487, test=484, 0 recording IDs shared between any pair of splits |
| stratification_within_2pp | ✅ PASS | max class deviation 0.07 pp |

## Key Findings

- **max_class_deviation_pp:** 0.0700 — Largest gap between a split's abnormal rate and the global rate
- **max_site_deviation_pp:** 0.2420
- **n_segments_total:** `37386`
- **n_segments_train:** `30695`
- **pct_abnormal_segments_train:** 23.2700
- **n_segments_val:** `3329`
- **pct_abnormal_segments_val:** 23.0100
- **n_segments_test:** `3362`
- **pct_abnormal_segments_test:** 25.3100
- **n_recordings_excluded:** `0`
- **exclusion_breakdown:**
  - `load_error`: 0
  - `quality`: 0
  - `no_segments`: 0
- **mean_spikes_removed:** 0.4800 — Mean friction transients suppressed per recording

## Table: split_composition

| split | n_recordings | n_segments | pct_abnormal |
|---|---|---|---|
| train | 2269 | 30695 | 23.2700 |
| val | 487 | 3329 | 23.0100 |
| test | 484 | 3362 | 25.3100 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/data/interim/splits.json | — | — | 0.0420 |
| /workspace/apr-heart-sounds/data/interim/segments_train.npy | 30695×6000 | float32 | 702.5500 |
| /workspace/apr-heart-sounds/data/interim/segments_val.npy | 3329×6000 | float32 | 76.1900 |
| /workspace/apr-heart-sounds/data/interim/segments_test.npy | 3362×6000 | float32 | 76.9500 |
| /workspace/apr-heart-sounds/data/interim/segment_index.csv | — | — | 1.5820 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/interim/raw_census.csv` (n_recordings=3240)

## Parameters Used

```yaml
target_sr: 2000
bandpass: [25.0, 400.0]
segment_seconds: 3.0
train_hop_seconds: 1.5
eval_hop_seconds: 3.0
normalize: zscore
```

## Timing Breakdown

- `_total`: 135.30 s
- `process_train`: 88.72 s
- `process_test`: 19.54 s
- `process_val`: 19.44 s
- `splits`: 0.17 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `02_extract_features`
