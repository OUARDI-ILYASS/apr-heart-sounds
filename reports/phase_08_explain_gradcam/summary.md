# Phase Summary — `08_explain_gradcam`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T23:38:32.667808+00:00  •  **Duration:** 12.56 s  
**Data splits read:** `test`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| gradcam_passes_randomization_check | ✅ PASS | |r| with random model = 0.051 (threshold 0.30) |

## Key Findings

- **gradcam_vs_gradcampp_correlation:** 0.3921 — High agreement means the conclusion is not an artefact of the specific CAM variant chosen
- **gradcam_energy_correlation:** 0.0678 — Correlation between the temporal attribution profile and the signal envelope. High values would mean the CNN is an energy detector rather than a murmur detector.

## Table: layer_sensitivity

| mean_temporal_concentration | layer | correlation_with_last |
|---|---|---|
| 0.0425 | conv_block_1 | None |
| 0.0234 | conv_block_2 | 0.5456 |
| 0.1352 | conv_block_3 | 0.1297 |
| 0.0740 | conv_block_4 | 0.0177 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/figures/fig_average_cams.pdf | — | — | 0.1940 |
| /workspace/apr-heart-sounds/results/xai/gradcam/gradcam_results.npz | 3362×32×188 | float64 | 154.3100 |
| /workspace/apr-heart-sounds/results/xai/gradcam/gradcam_analysis.json | — | — | 0.0260 |

## Parameters Used

```yaml
target_layer: conv_block_4
normalize: unit_mass
upsample_mode: bilinear
```

## Notes

- Grad-CAM's native resolution here is 2 frequency x 11 time cells. We therefore make temporal claims only; frequency claims come from the PWP SHAP analysis in phase 07, where the band mapping is exact.

## Timing Breakdown

- `_total`: 12.55 s
- `gradcam`: 1.13 s
- `energy_confound`: 0.87 s
- `gradcampp`: 0.21 s
- `layer_sensitivity`: 0.20 s
- `sanity_randomization`: 0.18 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `09_cycle_alignment`
