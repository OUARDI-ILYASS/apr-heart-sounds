# Interpretable Phonocardiogram Classification

Audio Pattern Recognition course project.

**Report:** `./report.pdf` — IEEE conference format, 10 pages with references.

---

## Summary

The project addresses binary Normal/Abnormal classification of phonocardiogram
recordings from the PhysioNet/CinC Challenge 2016 corpus. Three feature
representations that differ in kind — cepstral (MFCC), spectral (log-Mel) and
perceptual wavelet packet (PWP) — are compared across three classifier families
(SVM, random forest, CNN) under a single recording-level protocol. The best
configuration reaches a modified accuracy of 0.866, comparable to published
entries on this corpus.

The contribution is not that number. Each performance claim is paired with a
diagnostic capable of refuting it, and two of the three expectations the study
began with were contradicted by the evidence. They are reported in the same voice
as the results that held.

| Diagnostic | Result |
|---|---|
| k-means on the unsupervised feature spaces | recovers the **recording site** (ARI 0.201) far better than the **diagnosis** (ARI 0.018) |
| Per-sub-database performance | MAcc spans **0.986 → 0.477**; three of five sites at or below a constant predictor |
| Grad-CAM systolic enrichment | 1.056, and *no higher* on correct predictions (1.085) than on false alarms (1.115) |

Abnormal prevalence varies across the six collection sites by 68.9 percentage
points, so site identity alone predicts the label. The pooled metric is carried
by `training-e`, which supplies 66 % of test recordings at 8.7 % prevalence.

Conclusion: on this corpus the aggregate metric overstates what the models have
learned.

---

## Results

| Model | Acc. | Se. | Sp. | MAcc | 95 % CI | AUC |
|---|---|---|---|---|---|---|
| SVM–MFCC | 0.816 | 0.949 | 0.782 | **0.866** | [0.834, 0.894] | 0.922 |
| CNN–log-Mel | 0.851 | 0.889 | 0.842 | 0.865 | [0.828, 0.900] | **0.961** |
| RF–MFCC | 0.802 | 0.949 | 0.764 | 0.857 | [0.824, 0.886] | 0.906 |
| SVM–PWP | 0.785 | 0.909 | 0.753 | 0.831 | [0.794, 0.863] | 0.877 |
| RF–PWP | 0.731 | 0.970 | 0.670 | 0.820 | [0.790, 0.847] | 0.864 |

All intervals overlap. Trivial baseline: a constant *Normal* predictor scores
0.795 accuracy at 0.500 MAcc, which is why MAcc rather than accuracy is the
primary metric.

Five claims were declared in advance with thresholds fixed in code. Two were
contradicted (class structure recoverable without labels; CNN beats the best
classical model), one supported, two weak. All are reported as scored.

| # | Claim | Verdict | Evidence | Scored in |
|---|---|---|---|---|
| C1 | Class structure is recoverable without labels | **contradicted** | best ARI vs diagnosis 0.028, vs site 0.201 | phase 03 |
| C2 | The CNN outperforms the best classical pairing | **contradicted** | 0.865 vs 0.866, CIs overlap | phase 06 |
| C3 | All models exceed trivial baselines | supported | 0.820–0.866 vs 0.500 | phase 06 |
| C4 | Frequency attribution agrees across feature domains | weak | mean pairwise correlation 0.514 | phase 07 |
| C5 | Attribution concentrates in systole | weak | E = 1.056, d = 0.131, 33 % significant | phase 09 |


---

## Method

```
signal → 2 kHz → 25–400 Hz zero-phase Butterworth → spike removal
       → per-recording z-norm → 3 s windows
                    │
       ┌────────────┼────────────┐
     MFCC        log-Mel        PWP
    (234-d)     (32×188)      (84-d)
       │            │            │
   SVM · RF       CNN        SVM · RF
       └────────────┼────────────┘
              mean probability
            → recording decision
```

Three feature domains computed from identical windows, so any difference is
attributable to the representation. Window scores aggregated to a recording
decision by averaging probabilities.

Explainability: SHAP (exact for forests, kernel for SVMs) projected onto
frequency; Grad-CAM on the CNN, tested against uniform and permutation nulls for
enrichment in the cardiac cycle, with model-randomisation and energy-confound
sanity checks reported before the findings.

---

## Pipeline

```
00_download_data        fetch corpus, build census
01_preprocess_audio     filter → spike removal → normalise → split → window
02_extract_features     MFCC │ log-Mel │ PWP
03_cluster_features     k-means sweep + PCA/t-SNE
04_train_classical      SVM + Random Forest, grouped CV
05_train_cnn            2-D CNN on log-Mel
06_evaluate_models      first and only phase to open the test split
07_explain_shap         TreeSHAP + KernelSHAP → frequency attribution
08_explain_gradcam      Grad-CAM + sanity checks
09_cycle_alignment      alignment metric and null tests
10_build_report_assets  dashboard + LaTeX tables generated from result JSON
```

Each phase reads and writes artifacts on disk, so any phase can be re-run in
isolation. Each writes a summary to `reports/phase_*/summary.md` recording its
findings, assertions, and which data splits it read.

---

## Running

```bash
pip install -r requirements.txt

bash scripts/run_all.sh
```

Roughly 1–2 hours end to end on a single GPU, dominated by feature extraction.
Individual phases via `make data preprocess features cluster classical cnn
evaluate shap gradcam alignment report`.

---

## Layout

```
configs/          config.yaml
src/              library code, no side effects at import
  config/         loading, hashing, fail-fast validation
  data/           download, preprocessing, segmentation, splits
  features/       MFCC · log-Mel · PWP extractors + scaling
  models/         SVM/RF builders, CNN, trainer, inference
  clustering/     k-means, validity indices, PCA/t-SNE
  evaluation/     metrics, bootstrap, McNemar, calibration
  xai/            SHAP, Grad-CAM, cardiac segmenter, alignment metric
  visualization/  figure builders
scripts/          00–10 orchestrators + run_all.sh
figures/          generated figures
reports/          per-phase summaries + PIPELINE_STATUS.md
paper/            IEEE report: main.tex, sections/, tables/, figs/, refs.bib
```
---

## Reproducibility

One seed propagated to `random`, `numpy`, `torch` and all sklearn estimators.
Config SHA-256 hashed and recorded in every phase summary. Environment captured
per phase. LaTeX tables generated from result JSON and `\input{}` directly, so no
number in the report is retyped by hand.


### Requirements

- Python 3.10 or later
- ~8 GB RAM; a CUDA GPU is optional and reduces phase 05 substantially
- ~6 GB disk for the corpus and intermediates
- LaTeX with `IEEEtran.cls` and `IEEEtran.bst` to build the report

Total runtime is approximately two hours end to end on CPU, dominated by phase 02
(feature extraction) and phase 05 (CNN training); a GPU reduces the latter to a
few minutes. Every phase can be re-run in isolation, and `run_all.sh --from 04`
resumes from any point, since phases communicate only through files on disk.

---

## Data availability

PhysioNet/CinC Challenge 2016 — 3,240 recordings, 20.2 hours, 2 kHz, six
sub-databases, 20.5 % abnormal. Distributed under the ODC-BY 1.0 licence and
obtained via `scripts/00_download_data.py`. **The corpus is not redistributed in
this repository.**

Primary references:

> Liu, C. et al. (2016). An open access database for the evaluation of heart
> sound algorithms. *Physiological Measurement*, 37(12), 2181–2213.

> Clifford, G. D. et al. (2016). Classification of normal/abnormal heart sound
> recordings: the PhysioNet/Computing in Cardiology Challenge 2016. *Computing in
> Cardiology*, 43, 609–612.