# Phase Summary — `02_extract_features`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `8ae8dd4e`  •  **Seed:** `42`  
**Started:** 2026-07-21T19:09:43.650460+00:00  •  **Duration:** 562.81 s  
**Data splits read:** `train, val, test`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| mfcc_features_non_degenerate | ✅ PASS | train feature std = 1.000000 |
| logmel_features_non_degenerate | ✅ PASS | train feature std = 1.000000 |
| pwp_all_bands_populated | ✅ PASS | nodes per band: [2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 3] |
| pwp_features_non_degenerate | ✅ PASS | train feature std = 1.000000 |

## Key Findings

- **mfcc_scaler_report:**
  - `n_features`: 234
  - `n_near_constant`: 0
  - `near_constant_features`: []
  - `scale_min`: 0.0258
  - `scale_max`: 49.5179
- **mfcc_dim:** `234`
- **mfcc_window_ms:** 128.0000
- **logmel_n_degenerate_bands:** `0` — Mel bands with (near) zero training variance
- **logmel_dim:** `6016`
- **logmel_shape:** `[32, 188]` — CNN input (n_mels, n_frames)
- **logmel_freq_resolution_hz:** 7.8100
- **pwp_scaler_report:**
  - `n_features`: 84
  - `n_near_constant`: 0
  - `near_constant_features`: []
  - `scale_min`: 0.0044
  - `scale_max`: 16.9875
- **pwp_dim:** `84`

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/models/scalers/mfcc_scaler.joblib | — | — | 0.0060 |
| /workspace/apr-heart-sounds/data/processed/mfcc/features_train.npy | 25365×234 | float32 | 22.6400 |
| /workspace/apr-heart-sounds/data/processed/mfcc/features_val.npy | 2814×234 | float32 | 2.5100 |
| /workspace/apr-heart-sounds/data/processed/mfcc/features_test.npy | 2756×234 | float32 | 2.4600 |
| /workspace/apr-heart-sounds/data/processed/mfcc/feature_names.json | — | — | 0.0040 |
| /workspace/apr-heart-sounds/models/scalers/logmel_norm.json | — | — | 0.0010 |
| /workspace/apr-heart-sounds/data/processed/logmel/features_train.npy | 25365×32×188 | float32 | 582.1100 |
| /workspace/apr-heart-sounds/data/processed/logmel/features_val.npy | 2814×32×188 | float32 | 64.5800 |
| /workspace/apr-heart-sounds/data/processed/logmel/features_test.npy | 2756×32×188 | float32 | 63.2500 |
| /workspace/apr-heart-sounds/data/processed/logmel/feature_names.json | — | — | 0.1260 |
| /workspace/apr-heart-sounds/models/scalers/pwp_scaler.joblib | — | — | 0.0020 |
| /workspace/apr-heart-sounds/data/processed/pwp/features_train.npy | 25365×84 | float32 | 8.1300 |
| /workspace/apr-heart-sounds/data/processed/pwp/features_val.npy | 2814×84 | float32 | 0.9000 |
| /workspace/apr-heart-sounds/data/processed/pwp/features_test.npy | 2756×84 | float32 | 0.8800 |
| /workspace/apr-heart-sounds/data/processed/pwp/feature_names.json | — | — | 0.0020 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/interim/segment_index.csv` (n_segments=30935)

## Parameters Used

```yaml
mfcc_config: {'domain': 'mfcc', 'sr': 2000.0, 'output_shape': [234], 'n_features': 234, 'n_fft': 256, 'hop_length': 32, 'n_mels': 32, 'n_mfcc': 13, 'fmin': 25.0, 'fmax': 400.0, 'use_delta': True, 'use_delta2': True, 'aggregations': ['mean', 'std', 'min', 'max', 'skew', 'kurtosis'], 'n_streams': 3, 'window_ms': 128.0, 'hop_ms': 16.0}
logmel_config: {'domain': 'logmel', 'sr': 2000.0, 'output_shape': [32, 188], 'n_features': 6016, 'scale': 'mel', 'n_fft': 256, 'hop_length': 32, 'n_mels': 32, 'n_frames': 188, 'fmin': 25.0, 'fmax': 400.0, 'window_ms': 128.0, 'hop_ms': 16.0, 'freq_resolution_hz': 7.81}
pwp_config: {'domain': 'pwp', 'sr': 2000.0, 'output_shape': [84], 'n_features': 84, 'wavelet': 'db4', 'level': 6, 'node_order': 'freq', 'n_nodes_total': 64, 'n_nodes_kept': 25, 'node_width_hz': 15.625, 'n_perceptual_bands': 12, 'perceptual_grouping': True, 'band_edges_hz': [25.0, 50.6, 77.2, 104.6, 133.1, 162.5, 193.0, 224.6, 257.3, 291.1, 326.2, 362.4, 400.0], 'descriptors': ['log_energy', 'rel_energy', 'shannon_entropy', 'std', 'skew', 'kurtosis', 'max_abs'], 'nodes_per_band': [2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 3]}
```

## Timing Breakdown

- `_total`: 562.81 s
- `pwp_train`: 299.44 s
- `mfcc_train`: 141.33 s
- `pwp_val`: 33.47 s
- `pwp_test`: 32.88 s
- `mfcc_val`: 16.01 s
- `mfcc_test`: 15.46 s
- `logmel_train`: 14.94 s
- `logmel_val`: 1.68 s
- `logmel_test`: 1.63 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `03_cluster_features`
