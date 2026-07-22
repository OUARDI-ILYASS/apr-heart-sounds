# Phase Summary — `04_train_classical`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `f915b259`  •  **Seed:** `42`  
**Started:** 2026-07-22T00:10:14.863018+00:00  •  **Duration:** 3530.99 s  
**Data splits read:** `train, val`  

> [!WARNING]
> - rf: train-CV gap is 0.165 (train 0.949 vs CV 0.842). The model is memorising the training folds; consider stronger regularisation (lower C for SVM, higher min_samples_leaf for RF).
> - rf: train-CV gap is 0.234 (train 0.950 vs CV 0.793). The model is memorising the training folds; consider stronger regularisation (lower C for SVM, higher min_samples_leaf for RF).

## Key Findings

- **svm_mfcc_cv_macc:** 0.8512
- **svm_mfcc_val_macc_recording:** 0.8060
- **svm_mfcc_threshold:** 0.2300
- **rf_mfcc_cv_macc:** 0.8421
- **rf_mfcc_val_macc_recording:** 0.8335
- **rf_mfcc_threshold:** 0.5300
- **svm_pwp_cv_macc:** 0.8211
- **svm_pwp_val_macc_recording:** 0.7481
- **svm_pwp_threshold:** 0.2000
- **rf_pwp_cv_macc:** 0.7935
- **rf_pwp_val_macc_recording:** 0.8102
- **rf_pwp_threshold:** 0.4400
- **best_classical_model:** `rf_mfcc`

## Table: classical_validation

| n_features | val_sensitivity | model | val_specificity | features | cv_macc | val_macc_recording | cv_std | overfit_gap |
|---|---|---|---|---|---|---|---|---|
| 234 | 0.9130 | rf_mfcc | 0.7540 | mfcc | 0.8421 | 0.8335 | 0.0131 | 0.1655 |
| 84 | 0.8696 | rf_pwp | 0.7508 | pwp | 0.7935 | 0.8102 | 0.0168 | 0.2341 |
| 234 | 0.7174 | svm_mfcc | 0.8946 | mfcc | 0.8512 | 0.8060 | 0.0126 | 0.0930 |
| 84 | 0.6304 | svm_pwp | 0.8658 | pwp | 0.8211 | 0.7481 | 0.0111 | 0.0846 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/models/classical/svm_mfcc.joblib | — | — | 10.2010 |
| /workspace/apr-heart-sounds/models/classical/rf_mfcc.joblib | — | — | 4.1740 |
| /workspace/apr-heart-sounds/models/classical/svm_pwp.joblib | — | — | 4.1270 |
| /workspace/apr-heart-sounds/models/classical/rf_pwp.joblib | — | — | 4.6690 |
| /workspace/apr-heart-sounds/results/evaluation/classical_cv.json | — | — | 0.0320 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/processed/mfcc/features_train.npy` (shape=[25365, 234])
- `/workspace/apr-heart-sounds/data/processed/pwp/features_train.npy` (shape=[25365, 84])

## Notes

- All numbers in this phase are cross-validation or validation scores. The test split has not been read. Hyperparameter and threshold choices are now frozen.

## Timing Breakdown

- `_total`: 3530.99 s
- `gridsearch_rf_mfcc`: 1745.84 s
- `gridsearch_rf_pwp`: 866.55 s
- `gridsearch_svm_mfcc`: 608.18 s
- `gridsearch_svm_pwp`: 299.95 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `05_train_cnn`
