# Phase Summary — `11_build_report_assets`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T03:50:41.522258+00:00  •  **Duration:** 2.6 s  

## Assertions

| Check | Result | Detail |
|---|---|---|
| test_split_not_used_for_selection | ❌ FAIL | Phases 03-05 (clustering, classical training, CNN training) must not appear in this list. Actual readers: ['01_preprocess_audio', '02_extract_features', '06_evaluate_models', '07_explain_shap', '08_explain_gradcam', '09_cycle_alignment'] |

## Key Findings

- **phases_reading_test_split:** `['01_preprocess_audio', '02_extract_features', '06_evaluate_models', '07_explain_shap', '08_explain_gradcam', '09_cycle_alignment']`
- **n_latex_tables:** `7`
- **n_figures_copied:** `77`
- **claim_verdicts:**
  - `C1`: contradicted
  - `C2`: supported
  - `C3`: supported
  - `C4`: weak
  - `C5`: contradicted

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/reports/PIPELINE_STATUS.md | — | — | 0.0030 |

## Notes

- Claims ['C1', 'C4', 'C5'] are weak or contradicted. These must be stated as such in the paper. A contradicted claim reported honestly is a result; a contradicted claim quietly dropped is misconduct.

## Timing Breakdown

- `_total`: 2.60 s
- `collect_summaries`: 0.08 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

