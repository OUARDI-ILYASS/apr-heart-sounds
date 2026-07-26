# Phase Summary — `06_evaluate_models`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T23:24:09.610845+00:00  •  **Duration:** 21.02 s  
**Data splits read:** `test`  

> [!WARNING]
> - Per-site MAcc varies by 0.509. Report this prominently: it indicates the model does not transfer uniformly across acquisition conditions, and the pooled metric hides that.

## Assertions

| Check | Result | Detail |
|---|---|---|
| no_gross_site_shortcut | ❌ FAIL | per-site MAcc spread = 0.509 |

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C2 | The log-Mel CNN outperforms the best classical feature/model pairing. | ❌ contradicted | CNN MAcc 0.8652 vs classical 0.8657 (delta -0.0004, McNemar p=0.0007) |
| C3 | All models substantially exceed the majority-class baseline on MAcc. | ✅ supported | best MAcc 0.8657 vs 0.500 for the always-Normal baseline |

## Key Findings

- **majority_class_accuracy:** 0.7955 — What a model that always predicts Normal achieves. Any reported accuracy must be read against this number, which is why MAcc is our primary metric.
- **site_macc_spread:** 0.5090 — Range of MAcc across sub-databases for the best model. A large spread suggests the model exploits site-specific cues rather than pathology.
- **rf_mfcc_ece:** 0.1294
- **rf_pwp_ece:** 0.1500
- **svm_mfcc_ece:** 0.0439
- **svm_pwp_ece:** 0.0418
- **cnn_logmel_ece:** 0.1164
- **rf_mfcc_test_macc:** 0.8566
- **rf_pwp_test_macc:** 0.8199
- **svm_mfcc_test_macc:** 0.8657
- **svm_pwp_test_macc:** 0.8312
- **cnn_logmel_test_macc:** 0.8652
- **best_model:** `svm_mfcc`
- **best_test_macc:** 0.8657
- **n_significant_comparisons:** `7`

## Table: trivial_baselines

| specificity | baseline | sensitivity | accuracy | macc |
|---|---|---|---|---|
| 1.0000 | always_normal | 0.0000 | 0.7955 | 0.5000 |
| 0.0000 | always_abnormal | 1.0000 | 0.2045 | 0.5000 |
| 0.7481 | random_stratified | 0.2121 | 0.6384 | 0.4801 |

## Table: per_subdatabase

| n | group | specificity | prevalence | sensitivity | note | accuracy | macc |
|---|---|---|---|---|---|---|---|
| 61 | training-a | 0.0000 | 0.7210 | 0.9545 |  | 0.6885 | 0.4773 |
| 73 | training-b | 0.1552 | 0.2050 | 0.8667 |  | 0.3014 | 0.5109 |
| 4 | training-c |  |  |  | insufficient data or single class |  |  |
| 8 | training-d | 0.0000 | 0.5000 | 1.0000 |  | 0.5000 | 0.5000 |
| 321 | training-e | 0.9727 | 0.0870 | 1.0000 |  | 0.9751 | 0.9863 |
| 17 | training-f | 0.5833 | 0.2940 | 0.8000 |  | 0.6471 | 0.6917 |

## Table: literature_comparison

| specificity | test_set | system | sensitivity | macc | comparable |
|---|---|---|---|---|---|
| 0.7818 | our held-out split of the public training data | This work (best) | 0.9495 | 0.8657 | True |
| 0.7780 | official CinC 2016 hidden test set | Potes et al. 2016 (CinC winner) | 0.9420 | 0.8600 | False |
| 0.9510 | official CinC 2016 hidden test set | Rubin et al. 2016 (CNN) | 0.7270 | 0.8390 | False |
| 0.8490 | official CinC 2016 hidden test set | Zabihi et al. 2016 (ensemble) | 0.8690 | 0.8590 | False |

## Table: test_results_recording_level

| f1 | n | specificity | sensitivity | model | accuracy | macc | roc_auc | macc_ci |
|---|---|---|---|---|---|---|---|---|
| 0.6787 | 484 | 0.7818 | 0.9495 | svm_mfcc | 0.8161 | 0.8657 | 0.9217 | [0.834, 0.894] |
| 0.7097 | 484 | 0.8416 | 0.8889 | cnn_logmel | 0.8512 | 0.8652 | 0.9606 | [0.828, 0.900] |
| 0.6620 | 484 | 0.7636 | 0.9495 | rf_mfcc | 0.8017 | 0.8566 | 0.9061 | [0.824, 0.886] |
| 0.6338 | 484 | 0.7532 | 0.9091 | svm_pwp | 0.7851 | 0.8312 | 0.8770 | [0.794, 0.863] |
| 0.5963 | 484 | 0.6701 | 0.9697 | rf_pwp | 0.7314 | 0.8199 | 0.8644 | [0.790, 0.847] |

## Table: pairwise_mcnemar

| p_value | statistic | test | b_only_correct | comparison | a_only_correct | significant |
|---|---|---|---|---|---|---|
| 0.0000 | 41.6540 | chi2_with_continuity_correction | 10 | cnn_logmel vs rf_pwp | 68 | True |
| 0.0000 | 17.7960 | chi2_with_continuity_correction | 11 | cnn_logmel vs svm_pwp | 43 | True |
| 0.0000 | 24.7500 | chi2_with_continuity_correction | 5 | rf_mfcc vs rf_pwp | 39 | True |
| 0.0000 | 28.0700 | chi2_with_continuity_correction | 49 | rf_pwp vs svm_mfcc | 8 | True |
| 0.0002 | 14.2050 | chi2_with_continuity_correction | 35 | rf_pwp vs svm_pwp | 9 | True |
| 0.0007 | 11.5000 | chi2_with_continuity_correction | 11 | cnn_logmel vs rf_mfcc | 35 | True |
| 0.0104 | 6.5640 | chi2_with_continuity_correction | 11 | cnn_logmel vs svm_mfcc | 28 | True |
| 0.0250 | 5.0260 | chi2_with_continuity_correction | 12 | svm_mfcc vs svm_pwp | 27 | True |
| 0.2100 | 8.0000 | exact_binomial | 15 | rf_mfcc vs svm_mfcc | 8 | False |
| 0.2684 | 1.2250 | chi2_with_continuity_correction | 16 | rf_mfcc vs svm_pwp | 24 | False |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/figures/fig_confusion_matrices.pdf | — | — | 0.0150 |
| /workspace/apr-heart-sounds/figures/fig_roc_curves.pdf | — | — | 0.0190 |
| /workspace/apr-heart-sounds/figures/fig_pr_curves.pdf | — | — | 0.0170 |
| /workspace/apr-heart-sounds/figures/fig_model_comparison.pdf | — | — | 0.0110 |
| /workspace/apr-heart-sounds/figures/fig_per_site_macc.pdf | — | — | 0.0130 |
| /workspace/apr-heart-sounds/figures/fig_calibration.pdf | — | — | 0.0170 |
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

- `_total`: 21.02 s
- `predict_svm_mfcc`: 6.24 s
- `predict_svm_pwp`: 2.33 s
- `predict_cnn`: 0.59 s
- `predict_rf_mfcc`: 0.21 s
- `predict_rf_pwp`: 0.19 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `07_explain_shap`
