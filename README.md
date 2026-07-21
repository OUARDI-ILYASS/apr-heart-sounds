# Interpretable Phonocardiogram Classification

Audio Pattern Recognition course project. Binary Normal/Abnormal classification
of heart sound recordings (PhysioNet/CinC Challenge 2016), comparing three
feature domains and three model families, with a **quantitatively evaluated**
explainability analysis.

---

## What is actually new here

Most applied XAI sections show three heatmaps and assert that the model "focuses
on clinically relevant regions". That assertion is unfalsifiable as written.
This project makes it testable.

Given an attribution map (Grad-CAM for the CNN, SHAP for the classical models)
and a cardiac state map obtained independently of the model, we compute the
**enrichment**

```
E_s  =  (fraction of attribution mass inside state s)
        ────────────────────────────────────────────
        (fraction of time occupied by state s)
```

`E = 1` is exactly what uniform temporal attention produces, so `E > 1` is a
real preference and `E < 1` is avoidance. Since most clinically important
murmurs are systolic, a model that has learned pathology rather than an
artefact should show `E_systole > 1` — a prediction stated *before* looking at
the results, and tested against uniform, shuffled and state-shuffled nulls with
a permutation test.

The three sanity checks that decide whether any of it means anything are part of
the pipeline, not optional extras:

| Check | What it would invalidate |
|---|---|
| Adebayo model-randomisation | Grad-CAM maps being architecture artefacts rather than learned structure |
| Energy-confound correlation | "Attends to systole" really meaning "attends to whatever is loud" |
| Per-sub-database performance | The classifier recognising the recording site instead of the pathology |

---

## Pipeline

```
00_download_data        fetch corpus, build raw census
01_preprocess_audio     bandpass → spike removal → normalise → split → segment
02_extract_features     MFCC (234-d) │ log-Mel (32×188) │ PWP (84-d)
03_cluster_features     k-means sweep + PCA/t-SNE          ← course requirement
04_train_classical      SVM + Random Forest, grouped CV
05_train_cnn            2-D CNN on log-Mel (PyTorch)
06_evaluate_models      ★ first and only phase to open the test split
07_explain_shap         TreeSHAP + KernelSHAP → frequency attribution
08_explain_gradcam      Grad-CAM + sanity checks
09_cycle_alignment      ★ the alignment metric and its null tests
10_run_ablations        7 single-parameter ablations
11_build_report_assets  dashboard + auto-generated LaTeX tables
```

Every phase reads artifacts from disk and writes artifacts to disk, so any phase
can be re-run in isolation and the pipeline resumed from any point. Every phase
writes a summary (`reports/phase_*/summary.md`) recording what it produced, what
it asserted, and **which data splits it read**.

---

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For a local NVIDIA GPU, install the CUDA build of torch FIRST:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Verify the code before committing hours of compute
make test

# 3. Run everything
bash scripts/run_all.sh

# ...or phase by phase
make data preprocess features cluster classical cnn evaluate
make shap gradcam alignment ablations report
```

Expected runtime on a single modern GPU: roughly 2–4 hours end to end, dominated
by phase 02 (feature extraction, CPU-bound) and phase 10 (ablations, which re-run
earlier phases).

If your network blocks `physionet.org`, download the six `training-*.zip`
archives manually into `data/raw/physionet2016/` and run
`python scripts/00_download_data.py --skip-download`.

---

## Layout

```
configs/          config.yaml (master) + 7 ablation overrides
src/              pure library code — no side effects at import
  config/         loading, deep-merge, hashing, fail-fast validation
  data/           download, preprocessing, segmentation, splits
  features/       MFCC · log-Mel · PWP extractors + scaling
  models/         SVM/RF builders, CNN, trainer, inference
  clustering/     k-means, validity indices, PCA/t-SNE
  evaluation/     metrics, bootstrap, McNemar, calibration
  xai/            SHAP, Grad-CAM, cardiac segmenter, alignment metric
  visualization/  IEEE-styled figure builders
scripts/          00–11 orchestrators + run_all.sh
tests/            94 unit tests
paper/            IEEE LaTeX skeleton; tables/ and figs/ are generated
docs/             PROFESSOR_QA.md — anticipated exam questions with answers
```

---

## Design decisions worth knowing before you read the code

**Splits are frozen at recording level and asserted disjoint.** A recording
contributes 5–20 segments that share a patient, a stethoscope and a murmur.
`assert_no_leakage` raises and kills the run if any ID appears in two splits.
Splits are stratified jointly on `(label, sub-database)`, because abnormal
prevalence varies enormously between the six collection sites — so site identity
predicts the label, and a random split would let a model score well by
recognising the equipment.

**Only phase 06 touches the test set.** Every phase records which splits it
read; the dashboard prints that audit trail automatically. Hyperparameters and
decision thresholds are frozen in phases 04–05 using validation data only.

**MAcc, not accuracy.** About 78% of recordings are Normal, so an unconditional
"Normal" predictor scores 78% accuracy at 0.50 MAcc. That baseline is printed in
the results table so nobody has to do the arithmetic. Ablation A3 removes class
weighting to demonstrate the failure mode deliberately.

**32 mel bands, not 64.** At sr = 2000 with n_fft = 256 the bin width is 7.8 Hz,
so only ~48 FFT bins fall in the 25–400 Hz band. Asking for 64 filters produces
empty, all-zero filters. librosa only warns; `src/config/schema.py` turns it
into a hard error.

**Zero-phase filtering.** The whole XAI argument is about *when* the model
attends, so a causal filter's frequency-dependent group delay would silently
misalign 200 Hz murmur energy against 40 Hz S1 energy.

**Wavelet packet nodes are requested in frequency order.** pywt returns natural
(Paley) order by default, which is a bit-reversed permutation of frequency
order. Indexing it as though it were frequency-ordered scrambles the spectrum
and nothing crashes. The validator refuses any other setting.

**The segmenter is not Springer's HSMM, and we say so.** It is a simplified
envelope-based labeller used *only at evaluation time*, never during training.
A noisier segmenter adds variance to the alignment estimate and therefore makes
the claim harder to support — it cannot manufacture a positive result. Segments
below a confidence threshold are excluded and the exclusion rate is reported.

---

## Reproducibility

- One seed propagated to `random`, `numpy`, `torch` and every sklearn estimator.
- Config is SHA-256 hashed; the hash is recorded in every phase summary, so any
  number traces back to the exact parameters that produced it.
- Ablations are *partial* config files deep-merged onto the base, and the runner
  records the exact dotted key paths that differ — so "we changed one thing" is
  verifiable from the artifacts rather than taken on trust.
- Environment (Python, packages, git commit, GPU) captured per phase.
- LaTeX tables are generated from result JSON and `\input{}` directly, so no
  number in the paper is ever retyped by hand.

---

## Paper

`paper/main.tex` is the IEEE conference skeleton. The section files under
`paper/sections/` are **intentionally empty** — they are written after the
experiments have run, against `reports/PIPELINE_STATUS.md` and the generated
tables, so that no claim is written before the evidence for it exists.

`IEEEtran.cls` is not bundled. TeX Live and MiKTeX ship it; otherwise
`tlmgr install ieeetran` or download it from the IEEE templates page.

```bash
make paper    # pdflatex → bibtex → pdflatex ×2
```

---

## Data

PhysioNet/CinC Challenge 2016 — 3,126 recordings, 2 kHz, six sub-databases.
Released under the ODC-BY licence; cite Liu et al. (2016) and Clifford et al.
(2016). The corpus is **not** included in this repository.

---

## Exam preparation

`docs/PROFESSOR_QA.md` collects the questions an examiner is most likely to ask,
with answers and pointers to the code that backs each one. The same material
appears as `PROFESSOR Q:` comments at the corresponding decision points in the
source, so you can point at the file while answering.
