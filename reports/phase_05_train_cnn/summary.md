# Phase Summary — `05_train_cnn`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T23:21:58.577912+00:00  •  **Duration:** 85.97 s  
**Data splits read:** `train, val`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| cnn_beats_chance | ✅ PASS | val recording MAcc = 0.8894 |
| train_val_gap_acceptable | ✅ PASS | final gap = +0.0537 |

## Key Findings

- **n_parameters:** `105778`
- **best_epoch:** `8`
- **early_stopped:** `True`
- **val_macc_segment:** 0.9153
- **val_macc_recording:** 0.8894
- **training_minutes:** 1.3000
- **gradcam_feature_map_shape:** `[128, 2, 11]` — Grad-CAM spatial resolution before upsampling

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/models/cnn/cnn_logmel.pt | — | — | 0.4220 |
| /workspace/apr-heart-sounds/results/evaluation/cnn_training.json | — | — | 0.0110 |
| /workspace/apr-heart-sounds/figures/fig_training_curves.pdf | — | — | 0.0120 |

## Parameters Used

```yaml
epochs_configured: 60
epochs_run: 20
batch_size: 64
learning_rate: 0.001
augmentation: {'enabled': True, 'time_shift': True, 'time_shift_max_frac': 0.2, 'gaussian_noise': True, 'noise_std': 0.01, 'spec_augment': False}
device: cuda
amp: True
```

## Notes

- Weights are restored from the best validation epoch, not the last. The test split has not been read in this phase.

## Timing Breakdown

- `_total`: 85.96 s
- `training`: 80.79 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `06_evaluate_models`
