# Phase Summary — `09_cycle_alignment`

✅ **Status:** success  
**Experiment:** `baseline_full`  
**Config hash:** `bfdef2da`  •  **Seed:** `42`  
**Started:** 2026-07-22T03:50:03.817130+00:00  •  **Duration:** 33.81 s  
**Data splits read:** `test`  

## Assertions

| Check | Result | Detail |
|---|---|---|
| heart_rate_physiologically_plausible | ✅ PASS | mean detected heart rate = 81.4 bpm |

## Claim Verdicts

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C5 | CNN attribution concentrates in systole beyond what uniform temporal attention would produce. | ❌ contradicted | mean enrichment E_systole = 0.936 (1.0 = no preference); 28% of segments significant against the shuffled null; Cohen's d = 0.021 |

## Key Findings

- **segmenter_usable_fraction:** 0.6615 — Fraction of test windows the segmenter could label confidently. Everything below the threshold is EXCLUDED from the alignment analysis, and this number is the honest denominator.
- **segmenter_mean_confidence:** 0.5567
- **mean_heart_rate_bpm:** 81.4000
- **tp_minus_fp_enrichment:** 0.0862 — Positive values mean the model's systolic focus is stronger when it is right than when it is wrong - evidence that attention tracks real evidence rather than being a fixed habit of the architecture.
- **cnn_enrichment_systole:** 0.9356
- **alignment_inclusion_rate:** 0.6607
- **frac_segments_significant:** 0.2750
- **alignment_effect_size_d:** 0.0212

## Table: alignment_by_outcome

| category | mean_enrichment_systole | n | mean_mass_systole | frac_enriched_systole |
|---|---|---|---|---|
| true_positive | 0.9472 | 392 | 0.2275 | 0.4005 |
| true_negative | 0.9495 | 1194 | 0.2198 | 0.4564 |
| false_positive | 0.8609 | 215 | 0.2195 | 0.3209 |
| false_negative | 0.6827 | 20 | 0.1612 | 0.2000 |

## Table: alignment_summary

| E_S1 | E_systole | mass_S1 | E_diastole | n_segments | mass_S2 | mass_systole | inclusion_rate | mass_diastole | model | E_S2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9530 | 0.9360 | 0.1864 | 1.0920 | 1821 | 0.1583 | 0.2208 | 0.6607 | 0.4346 | cnn_logmel_gradcam | 0.7660 |

## Artifacts Written

| Path | Shape | Dtype | MB |
|---|---|---|---|
| /workspace/apr-heart-sounds/results/segmentation/test_segmentation.json | — | — | 0.0640 |
| /workspace/apr-heart-sounds/results/xai/alignment.json | — | — | 0.0030 |
| /workspace/apr-heart-sounds/figures/fig_alignment_cnn.pdf | — | — | 0.0130 |
| /workspace/apr-heart-sounds/figures/fig_alignment_comparison.pdf | — | — | 0.0100 |
| /workspace/apr-heart-sounds/figures/fig_gradcam_example.pdf | — | — | 0.0490 |
| /workspace/apr-heart-sounds/figures/supplementary/fig_segmentation_example.pdf | — | — | 0.0360 |

## Notes

- The enrichment statistic is a ratio of attribution mass to time budget, so 1.0 is the exact value produced by uniform temporal attention. This framing is what makes the number interpretable: 'X% of attention was in systole' is meaningless until compared against the fraction of time systole occupies.

## Timing Breakdown

- `_total`: 33.81 s
- `segmentation`: 15.54 s
- `alignment_cnn`: 5.41 s
- `significance`: 1.75 s

<details><summary>Environment</summary>

- Python 3.11.10 on Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Git commit: `e70663b` (dirty: True)
- GPU: NVIDIA RTX 4000 Ada Generation ×1, CUDA 12.4, 19.55 GB
- Packages: numpy==2.4.6, scipy==1.17.1, sklearn==1.9.0, pandas==3.0.3, librosa==0.11.0, pywt==1.8.0, torch==2.4.1+cu124, shap==0.51.0, matplotlib==3.11.1, joblib==1.5.3

</details>

➡️ **Next phase:** `10_run_ablations`
