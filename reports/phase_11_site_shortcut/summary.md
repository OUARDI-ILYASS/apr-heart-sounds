# Phase Summary — `11_site_shortcut`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `abcfe9c4`  •  **Seed:** `42`  
**Started:** 2026-07-26T02:42:41.012241+00:00  •  **Duration:** 214.18 s  
**Data splits read:** `train`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| test_split_untouched | ✅ PASS | Phase 11 reads the train split only; the test partition is not opened, preserving the single-reader guarantee of phase 06. |

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C6 | The recording site is directly recoverable from the features, and the pooled performance is substantially a site-prevalence shortcut: controlling for site (leave-one-site-out, or within-site) removes most of the apparent skill. | ✅ supported | site balanced-accuracy up to 0.647 vs 0.167 chance; diagnosis MAcc drops by up to 0.382 from pooled CV to leave-one-site-out; within-site diagnosis MAcc averages 0.730 |

## Key Findings

- **mfcc_site_balanced_accuracy:** 0.6470
- **mfcc_pooled_cv_macc:** 0.8399
- **mfcc_loso_macro_macc:** 0.4575
- **mfcc_pooled_minus_loso:** 0.3824
- **mfcc_within_site_macro_macc:** 0.7563
- **pwp_site_balanced_accuracy:** 0.5495
- **pwp_pooled_cv_macc:** 0.8096
- **pwp_loso_macro_macc:** 0.4720
- **pwp_pooled_minus_loso:** 0.3377
- **pwp_within_site_macro_macc:** 0.7031

## Table: site_shortcut_summary

| pooled_macc | chance | site_bal_acc | loso_macc | pooled_minus_loso | within_site_macc | domain |
|---|---|---|---|---|---|---|
| 0.8399 | 0.1667 | 0.6470 | 0.4575 | 0.3824 | 0.7563 | mfcc |
| 0.8096 | 0.1667 | 0.5495 | 0.4720 | 0.3377 | 0.7031 | pwp |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/results/site_shortcut/mfcc_site_shortcut.json | — | — | 0.0020 |
| /workspace/apr-heart-sounds/results/site_shortcut/pwp_site_shortcut.json | — | — | 0.0020 |
| /workspace/apr-heart-sounds/figures/fig_site_shortcut.pdf | — | — | 0.0160 |

## Inputs Loaded

- `/workspace/apr-heart-sounds/data/processed/mfcc/features_train.npy` (shape=[30695, 234])
- `/workspace/apr-heart-sounds/data/processed/pwp/features_train.npy` (shape=[30695, 84])

## Notes

- Estimator is a fixed, untuned balanced random forest across all three blocks; the quantity of interest is the gap between evaluation protocols, not an absolute score, so per-fold tuning is deliberately avoided. Sites with a single class or too few recordings in a fold are reported as skipped rather than imputed.

## Timing Breakdown

- `_total`: 214.18 s
- `mfcc_loso`: 46.92 s
- `mfcc_pooled_cv`: 38.10 s
- `pwp_loso`: 25.05 s
- `mfcc_site_clf`: 24.17 s
- `mfcc_within_site`: 23.10 s
- `pwp_pooled_cv`: 20.20 s
- `pwp_within_site`: 19.76 s
- `pwp_site_clf`: 14.96 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-45-generic-x86_64-with-glibc2.35
- Git commit: `603801b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.67 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.5, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

