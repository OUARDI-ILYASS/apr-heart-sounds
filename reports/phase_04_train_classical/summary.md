# Phase Summary — `04_train_classical`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T22:16:45.255604+00:00  •  **Duration:** 3906.48 s  
**Data splits read:** `train, val`  

> [!WARNING]
> - rf: train-CV gap is 0.143 (train 0.947 vs CV 0.867). The model is memorising the training folds; consider stronger regularisation (lower C for SVM, higher min_samples_leaf for RF).
> - rf: train-CV gap is 0.198 (train 0.945 vs CV 0.814). The model is memorising the training folds; consider stronger regularisation (lower C for SVM, higher min_samples_leaf for RF).

## Key Findings

- **svm_mfcc_cv_macc:** 0.8744
- **svm_mfcc_val_macc_recording:** 0.8461
- **svm_mfcc_threshold:** 0.2300
- **rf_mfcc_cv_macc:** 0.8669
- **rf_mfcc_val_macc_recording:** 0.8657
- **rf_mfcc_threshold:** 0.4400
- **svm_pwp_cv_macc:** 0.8488
- **svm_pwp_val_macc_recording:** 0.7757
- **svm_pwp_threshold:** 0.2200
- **rf_pwp_cv_macc:** 0.8141
- **rf_pwp_val_macc_recording:** 0.8248
- **rf_pwp_threshold:** 0.3600
- **best_classical_model:** `rf_mfcc`

## Table: classical_validation

| overfit_gap | features | cv_macc | model | cv_std | val_sensitivity | val_specificity | n_features | val_macc_recording |
|---|---|---|---|---|---|---|---|---|
| 0.1431 | mfcc | 0.8669 | rf_mfcc | 0.0045 | 0.9200 | 0.8114 | 234 | 0.8657 |
| 0.0407 | mfcc | 0.8744 | svm_mfcc | 0.0079 | 0.7800 | 0.9121 | 234 | 0.8461 |
| 0.1982 | pwp | 0.8141 | rf_pwp | 0.0229 | 0.8900 | 0.7597 | 84 | 0.8248 |
| 0.0732 | pwp | 0.8488 | svm_pwp | 0.0143 | 0.6600 | 0.8915 | 84 | 0.7757 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/models/classical/svm_mfcc.joblib | — | — | 9.6440 |
| /workspace/apr-heart-sounds/models/classical/rf_mfcc.joblib | — | — | 10.7160 |
| /workspace/apr-heart-sounds/models/classical/svm_pwp.joblib | — | — | 3.9260 |
| /workspace/apr-heart-sounds/models/classical/rf_pwp.joblib | — | — | 13.3640 |
| /workspace/apr-heart-sounds/results/evaluation/classical_cv.json | — | — | 0.0320 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/processed/mfcc/features_train.npy` (shape=[30695, 234])
- `/workspace/apr-heart-sounds/data/processed/pwp/features_train.npy` (shape=[30695, 84])

## Notes

- All numbers in this phase are cross-validation or validation scores. The test split has not been read. Hyperparameter and threshold choices are now frozen.

## Timing Breakdown

- `_total`: 3906.48 s
- `gridsearch_rf_mfcc`: 2060.57 s
- `gridsearch_rf_pwp`: 724.98 s
- `gridsearch_svm_mfcc`: 660.08 s
- `gridsearch_svm_pwp`: 448.36 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `05_train_cnn`
