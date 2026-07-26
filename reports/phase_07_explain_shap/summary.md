# Phase Summary — `07_explain_shap`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T23:24:36.636057+00:00  •  **Duration:** 830.52 s  
**Data splits read:** `train, test`  

> [!WARNING]
> - svm_mfcc: KernelSHAP is a Monte-Carlo approximation. Its per-feature ranking is only as stable as the rank_stability check reports.
> - svm_pwp: KernelSHAP is a Monte-Carlo approximation. Its per-feature ranking is only as stable as the rank_stability check reports.

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C4 | Models trained on different feature domains attribute importance to the same frequency region. | ⚠️ weak | mean pairwise correlation between frequency-attribution profiles = 0.514 |

## Key Findings

- **rf_mfcc_peak_frequency_hz:** 36.4000
- **rf_mfcc_centroid_hz:** 212.5000
- **rf_mfcc_top_feature:** `mfcc3_mean`
- **rf_pwp_peak_frequency_hz:** 37.8000
- **rf_pwp_centroid_hz:** 180.8000
- **rf_pwp_top_feature:** `pwp_b0_25-51Hz_shannon_entropy`
- **svm_mfcc_peak_frequency_hz:** 36.4000
- **svm_mfcc_centroid_hz:** 212.5000
- **svm_mfcc_top_feature:** `mfcc3_mean`
- **svm_pwp_peak_frequency_hz:** 37.8000
- **svm_pwp_centroid_hz:** 161.0000
- **svm_pwp_top_feature:** `pwp_b0_25-51Hz_shannon_entropy`
- **peak_frequencies_hz:** `[36.4, 37.8, 36.4, 37.8]`
- **n_models_peaking_in_murmur_band:** `0/4` — Models whose attribution peaks in the 100-300 Hz range where systolic murmur energy is clinically expected

## Table: shap_frequency_summary

| mapping | peak_hz | exact | centroid_hz | explainer | model |
|---|---|---|---|---|---|
| abs(DCT basis) projection | 36.4000 | True | 212.5000 | TreeSHAP | rf_mfcc |
| exact band aggregation | 37.8000 | True | 180.8000 | TreeSHAP | rf_pwp |
| abs(DCT basis) projection | 36.4000 | False | 212.5000 | KernelSHAP | svm_mfcc |
| exact band aggregation | 37.8000 | False | 161.0000 | KernelSHAP | svm_pwp |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/results/xai/shap/rf_mfcc_shap.json | — | — | 0.0090 |
| /workspace/apr-heart-sounds/results/xai/shap/rf_pwp_shap.json | — | — | 0.0060 |
| /workspace/apr-heart-sounds/results/xai/shap/svm_mfcc_shap.json | — | — | 0.0090 |
| /workspace/apr-heart-sounds/results/xai/shap/svm_pwp_shap.json | — | — | 0.0060 |
| /workspace/apr-heart-sounds/results/xai/shap/frequency_profiles.json | — | — | 0.0160 |
| /workspace/apr-heart-sounds/figures/fig_frequency_attribution.pdf | — | — | 0.0160 |

## Notes

- SHAP values are computed on test predictions, with the background distribution drawn from training data. Explaining training predictions would describe what the model memorised rather than how it generalises.
- MFCC frequency attributions are an approximation: cepstral coefficients are a DCT of the log-mel spectrum, so SHAP mass is redistributed through the magnitude of the DCT basis and the sign is discarded. PWP attributions are exact, because each PWP feature belongs to exactly one frequency band by construction. Where the two disagree, the PWP result is the reliable one.

## Timing Breakdown

- `_total`: 830.51 s
- `shap_svm_mfcc`: 580.90 s
- `shap_svm_pwp`: 219.74 s
- `shap_rf_pwp`: 12.08 s
- `shap_rf_mfcc`: 9.57 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `08_explain_gradcam`
