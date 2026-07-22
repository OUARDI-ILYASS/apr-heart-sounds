# Phase Summary — `01_preprocess_audio`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `8ae8dd4e`  •  **Seed:** `42`  
**Started:** 2026-07-21T19:07:28.070968+00:00  •  **Duration:** 129.98 s  
**Data splits read:** `train, val, test`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| no_patient_leakage | ✅ PASS | train=1886, val=405, test=402, 0 recording IDs shared between any pair of splits |
| stratification_within_2pp | ✅ PASS | max class deviation 0.19 pp |

## Key Findings

- **max_class_deviation_pp:** 0.1890 — Largest gap between a split's abnormal rate and the global rate
- **max_site_deviation_pp:** 0.2620
- **n_segments_total:** `30935`
- **n_segments_train:** `25365`
- **pct_abnormal_segments_train:** 26.6900
- **n_segments_val:** `2814`
- **pct_abnormal_segments_val:** 25.3400
- **n_segments_test:** `2756`
- **pct_abnormal_segments_test:** 27.3200
- **n_recordings_excluded:** `0`
- **exclusion_breakdown:**
  - `load_error`: 0
  - `quality`: 0
  - `no_segments`: 0
- **mean_spikes_removed:** 0.4800 — Mean friction transients suppressed per recording

## Table: split_composition

| n_recordings | split | n_segments | pct_abnormal |
|---|---|---|---|
| 1886 | train | 25365 | 26.6900 |
| 405 | val | 2814 | 25.3400 |
| 402 | test | 2756 | 27.3200 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/data/interim/splits.json | — | — | 0.0350 |
| /workspace/apr-heart-sounds/data/interim/segments_train.npy | 25365×6000 | float32 | 580.5600 |
| /workspace/apr-heart-sounds/data/interim/segments_val.npy | 2814×6000 | float32 | 64.4100 |
| /workspace/apr-heart-sounds/data/interim/segments_test.npy | 2756×6000 | float32 | 63.0800 |
| /workspace/apr-heart-sounds/data/interim/segment_index.csv | — | — | 1.3050 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/interim/raw_census.csv` (n_recordings=2693)

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

- `_total`: 129.98 s
- `process_train`: 88.12 s
- `process_val`: 19.01 s
- `process_test`: 18.62 s
- `splits`: 0.17 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `02_extract_features`
