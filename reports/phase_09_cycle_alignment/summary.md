# Phase Summary — `09_cycle_alignment`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T23:38:50.347747+00:00  •  **Duration:** 37.34 s  
**Data splits read:** `test`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| heart_rate_physiologically_plausible | ✅ PASS | mean detected heart rate = 82.0 bpm |

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C5 | CNN attribution concentrates in systole beyond what uniform temporal attention would produce. | ⚠️ weak | mean enrichment E_systole = 1.056 (1.0 = no preference); 33% of segments significant against the shuffled null; Cohen's d = 0.131 |

## Key Findings

- **segmenter_usable_fraction:** 0.6651 — Fraction of test windows the segmenter could label confidently. Everything below the threshold is EXCLUDED from the alignment analysis, and this number is the honest denominator.
- **segmenter_mean_confidence:** 0.5587
- **mean_heart_rate_bpm:** 82.0000
- **tp_minus_fp_enrichment:** -0.0302 — Positive values mean the model's systolic focus is stronger when it is right than when it is wrong - evidence that attention tracks real evidence rather than being a fixed habit of the architecture.
- **cnn_enrichment_systole:** 1.0556
- **alignment_inclusion_rate:** 0.6648
- **frac_segments_significant:** 0.3300
- **alignment_effect_size_d:** 0.1312

## Table: alignment_by_outcome

| n | mean_mass_systole | category | mean_enrichment_systole | frac_enriched_systole |
|---|---|---|---|---|
| 409 | 0.2668 | true_positive | 1.0847 | 0.6103 |
| 1600 | 0.2356 | true_negative | 1.0391 | 0.5225 |
| 200 | 0.2754 | false_positive | 1.1149 | 0.6350 |
| 26 | 0.2779 | false_negative | 1.1569 | 0.4615 |

## Table: alignment_summary

| mass_S1 | model | E_diastole | n_segments | E_systole | E_S2 | E_S1 | mass_systole | mass_S2 | inclusion_rate | mass_diastole |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1900 | cnn_logmel_gradcam | 1.0670 | 2235 | 1.0560 | 0.7530 | 1.0010 | 0.2454 | 0.1500 | 0.6648 | 0.4146 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/results/segmentation/test_segmentation.json | — | — | 0.0780 |
| /workspace/apr-heart-sounds/results/xai/alignment.json | — | — | 0.0030 |
| /workspace/apr-heart-sounds/figures/fig_alignment_cnn.pdf | — | — | 0.0130 |
| /workspace/apr-heart-sounds/figures/fig_alignment_comparison.pdf | — | — | 0.0100 |
| /workspace/apr-heart-sounds/figures/fig_gradcam_example.pdf | — | — | 0.0480 |
| /workspace/apr-heart-sounds/figures/supplementary/fig_segmentation_example.pdf | — | — | 0.0350 |

## Notes

- The enrichment statistic is a ratio of attribution mass to time budget, so 1.0 is the exact value produced by uniform temporal attention. This framing is what makes the number interpretable: 'X% of attention was in systole' is meaningless until compared against the fraction of time systole occupies.

## Timing Breakdown

- `_total`: 37.34 s
- `segmentation`: 18.23 s
- `alignment_cnn`: 6.07 s
- `significance`: 1.66 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Git commit: `80dd5ad` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `10_run_ablations`
