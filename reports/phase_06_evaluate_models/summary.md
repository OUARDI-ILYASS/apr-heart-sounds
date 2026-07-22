# Phase Summary — `06_evaluate_models`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `f915b259`  •  **Seed:** `42`  
**Started:** 2026-07-22T01:11:03.311043+00:00  •  **Duration:** 19.12 s  
**Data splits read:** `test`  

> [!WARNING]
> - Per-site MAcc varies by 0.368. Report this prominently: it indicates the model does not transfer uniformly across acquisition conditions, and the pooled metric hides that.

## Assertions

| Check | Result | Detail |
|---|---|---|
| no_gross_site_shortcut | ❌ FAIL | per-site MAcc spread = 0.368 |

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C2 | The log-Mel CNN outperforms the best classical feature/model pairing. | ✅ supported | CNN MAcc 0.8707 vs classical 0.8607 (delta +0.0100, McNemar p=0.0000) |
| C3 | All models substantially exceed the majority-class baseline on MAcc. | ✅ supported | best MAcc 0.8707 vs 0.500 for the always-Normal baseline |

## Key Findings

- **majority_class_accuracy:** 0.7761 — What a model that always predicts Normal achieves. Any reported accuracy must be read against this number, which is why MAcc is our primary metric.
- **site_macc_spread:** 0.3682 — Range of MAcc across sub-databases for the best model. A large spread suggests the model exploits site-specific cues rather than pathology.
- **rf_mfcc_ece:** 0.1400
- **rf_pwp_ece:** 0.1463
- **svm_mfcc_ece:** 0.0610
- **svm_pwp_ece:** 0.0661
- **cnn_logmel_ece:** 0.1262
- **rf_mfcc_test_macc:** 0.8354
- **rf_pwp_test_macc:** 0.8129
- **svm_mfcc_test_macc:** 0.8607
- **svm_pwp_test_macc:** 0.8263
- **cnn_logmel_test_macc:** 0.8707
- **best_model:** `cnn_logmel`
- **best_test_macc:** 0.8707
- **n_significant_comparisons:** `7`

## Table: trivial_baselines

| accuracy | macc | baseline | specificity | sensitivity |
|---|---|---|---|---|
| 0.7761 | 0.5000 | always_normal | 1.0000 | 0.0000 |
| 0.2239 | 0.5000 | always_abnormal | 0.0000 | 1.0000 |
| 0.6095 | 0.4756 | random_stratified | 0.7179 | 0.2333 |

## Table: per_subdatabase

| n | note | accuracy | group | prevalence | macc | specificity | sensitivity |
|---|---|---|---|---|---|---|---|
| 61 |  | 0.7869 | training-a | 0.7210 | 0.6718 | 0.4118 | 0.9318 |
| 73 |  | 0.5616 | training-b | 0.2050 | 0.6253 | 0.5172 | 0.7333 |
| 4 | insufficient data or single class |  | training-c |  |  |  |  |
| 8 |  | 0.6250 | training-d | 0.5000 | 0.6250 | 0.2500 | 1.0000 |
| 239 |  | 0.9874 | training-e | 0.0790 | 0.9932 | 0.9864 | 1.0000 |
| 17 |  | 0.7647 | training-f | 0.2940 | 0.6583 | 0.9167 | 0.4000 |

## Table: literature_comparison

| system | test_set | macc | specificity | sensitivity | comparable |
|---|---|---|---|---|---|
| This work (best) | our held-out split of the public training data | 0.8707 | 0.8526 | 0.8889 | True |
| Potes et al. 2016 (CinC winner) | official CinC 2016 hidden test set | 0.8600 | 0.7780 | 0.9420 | False |
| Rubin et al. 2016 (CNN) | official CinC 2016 hidden test set | 0.8390 | 0.9510 | 0.7270 | False |
| Zabihi et al. 2016 (ensemble) | official CinC 2016 hidden test set | 0.8590 | 0.8490 | 0.8690 | False |

## Table: test_results_recording_level

| n | accuracy | roc_auc | macc_ci | macc | model | specificity | f1 | sensitivity |
|---|---|---|---|---|---|---|---|---|
| 402 | 0.8607 | 0.9470 | [0.832, 0.907] | 0.8707 | cnn_logmel | 0.8526 | 0.7407 | 0.8889 |
| 402 | 0.7960 | 0.8941 | [0.832, 0.888] | 0.8607 | svm_mfcc | 0.7436 | 0.6822 | 0.9778 |
| 402 | 0.7935 | 0.8944 | [0.797, 0.871] | 0.8354 | rf_mfcc | 0.7596 | 0.6640 | 0.9111 |
| 402 | 0.7488 | 0.8767 | [0.792, 0.855] | 0.8263 | svm_pwp | 0.6859 | 0.6327 | 0.9667 |
| 402 | 0.7587 | 0.8593 | [0.775, 0.849] | 0.8129 | rf_pwp | 0.7147 | 0.6284 | 0.9111 |

## Table: pairwise_mcnemar

| statistic | comparison | significant | b_only_correct | a_only_correct | p_value | test |
|---|---|---|---|---|---|---|
| 16.4880 | cnn_logmel vs rf_mfcc | True | 7 | 34 | 0.0000 | chi2_with_continuity_correction |
| 23.1880 | cnn_logmel vs rf_pwp | True | 14 | 55 | 0.0000 | chi2_with_continuity_correction |
| 28.0580 | cnn_logmel vs svm_pwp | True | 12 | 57 | 0.0000 | chi2_with_continuity_correction |
| 12.5000 | cnn_logmel vs svm_mfcc | True | 12 | 38 | 0.0004 | chi2_with_continuity_correction |
| 12.0000 | svm_mfcc vs svm_pwp | True | 4 | 23 | 0.0005 | chi2_with_continuity_correction |
| 7.2250 | rf_mfcc vs svm_pwp | True | 11 | 29 | 0.0072 | chi2_with_continuity_correction |
| 6.3230 | rf_pwp vs svm_mfcc | True | 23 | 8 | 0.0119 | chi2_with_continuity_correction |
| 4.6940 | rf_mfcc vs rf_pwp | True | 11 | 25 | 0.0303 | chi2_with_continuity_correction |
| 9.0000 | rf_pwp vs svm_pwp | False | 9 | 13 | 0.5235 | exact_binomial |
| 0.0000 | rf_mfcc vs svm_mfcc | False | 13 | 12 | 1.0000 | chi2_with_continuity_correction |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/figures/fig_confusion_matrices.pdf | — | — | 0.0150 |
| /workspace/apr-heart-sounds/figures/fig_roc_curves.pdf | — | — | 0.0190 |
| /workspace/apr-heart-sounds/figures/fig_pr_curves.pdf | — | — | 0.0160 |
| /workspace/apr-heart-sounds/figures/fig_model_comparison.pdf | — | — | 0.0110 |
| /workspace/apr-heart-sounds/figures/fig_per_site_macc.pdf | — | — | 0.0130 |
| /workspace/apr-heart-sounds/figures/fig_calibration.pdf | — | — | 0.0160 |
| /workspace/apr-heart-sounds/results/evaluation/test_results.json | — | — | 0.0520 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/models/classical/rf_mfcc.joblib`
- `/workspace/apr-heart-sounds/models/classical/rf_pwp.joblib`
- `/workspace/apr-heart-sounds/models/classical/svm_mfcc.joblib`
- `/workspace/apr-heart-sounds/models/classical/svm_pwp.joblib`
- `/workspace/apr-heart-sounds/models/cnn/cnn_logmel.pt`

## Notes

- This is the first phase to read the test split. All model and threshold selection happened in phases 04-05 on training and validation data, as recorded in the splits_touched field of those phases' summaries.
- The literature rows in the comparison table were obtained on the official PhysioNet/CinC 2016 hidden test set, which was never publicly released. Our numbers come from a held-out split of the public training data. The two are therefore NOT directly comparable, and the table carries a 'comparable' column saying so.

## Timing Breakdown

- `_total`: 19.12 s
- `predict_svm_mfcc`: 6.11 s
- `predict_svm_pwp`: 2.14 s
- `predict_cnn`: 0.67 s
- `predict_rf_mfcc`: 0.14 s
- `predict_rf_pwp`: 0.09 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `07_explain_shap`
