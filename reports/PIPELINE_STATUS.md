# Pipeline Status

_Generated from 11 phase summaries._

## Phase Overview

| Phase | Status | Duration | Artifacts | Warnings | Failed checks |
|---|---|---|---|---|---|
| 00_download_data | ✅ | 2766s | 2 | 0 | — |
| 01_preprocess_audio | ✅ | 135s | 5 | 0 | — |
| 02_extract_features | ✅ | 653s | 15 | 0 | — |
| 03_cluster_features | ✅ | 174s | 4 | 0 | — |
| 04_train_classical | ✅ | 3906s | 5 | 2 | — |
| 05_train_cnn | ✅ | 86s | 3 | 0 | — |
| 06_evaluate_models | ✅ | 21s | 7 | 1 | no_gross_site_shortcut |
| 07_explain_shap | ✅ | 831s | 6 | 2 | — |
| 08_explain_gradcam | ✅ | 13s | 3 | 0 | — |
| 09_cycle_alignment | ✅ | 37s | 6 | 0 | — |
| 11_build_report_assets | ✅ | 3s | 1 | 0 | test_split_not_used_for_selection |

## Claim–Evidence Roll-up

| Claim | Statement | Verdict | Evidence | Decided in |
|---|---|---|---|---|
| C1 | The Normal/Abnormal distinction is recoverable by unsupervised clustering of the feature spaces. | ❌ contradicted | best ARI vs diagnosis at k=2 is 0.028, close to chance; ARI vs recording site reaches 0.201 | 03_cluster_features |
| C2 | The log-Mel CNN outperforms the best classical feature/model pairing. | ❌ contradicted | CNN MAcc 0.8652 vs classical 0.8657 (delta -0.0004, McNemar p=0.0007) | 06_evaluate_models |
| C3 | All models substantially exceed the majority-class baseline on MAcc. | ✅ supported | best MAcc 0.8657 vs 0.500 for the always-Normal baseline | 06_evaluate_models |
| C4 | Models trained on different feature domains attribute importance to the same frequency region. | ⚠️ weak | mean pairwise correlation between frequency-attribution profiles = 0.514 | 07_explain_shap |
| C5 | CNN attribution concentrates in systole beyond what uniform temporal attention would produce. | ⚠️ weak | mean enrichment E_systole = 1.056 (1.0 = no preference); 33% of segments significant against the shuffled null; Cohen's d = 0.131 | 09_cycle_alignment |

## Test-Set Audit

Phases that read the **test** split: `01_preprocess_audio`, `02_extract_features`, `06_evaluate_models`, `07_explain_shap`, `08_explain_gradcam`, `09_cycle_alignment`

> [!WARNING]
> More than three phases touched the test split. Model selection must happen on validation data only; verify none of these phases used test results to choose hyperparameters.

## All Warnings

- `04_train_classical`: rf: train-CV gap is 0.143 (train 0.947 vs CV 0.867). The model is memorising the training folds; consider stronger regularisation (lower C for SVM, higher min_samples_leaf for RF).
- `04_train_classical`: rf: train-CV gap is 0.198 (train 0.945 vs CV 0.814). The model is memorising the training folds; consider stronger regularisation (lower C for SVM, higher min_samples_leaf for RF).
- `06_evaluate_models`: Per-site MAcc varies by 0.509. Report this prominently: it indicates the model does not transfer uniformly across acquisition conditions, and the pooled metric hides that.
- `07_explain_shap`: svm_mfcc: KernelSHAP is a Monte-Carlo approximation. Its per-feature ranking is only as stable as the rank_stability check reports.
- `07_explain_shap`: svm_pwp: KernelSHAP is a Monte-Carlo approximation. Its per-feature ranking is only as stable as the rank_stability check reports.

