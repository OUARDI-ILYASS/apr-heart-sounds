# The pipeline, phase by phase — what each script does and why it does it that way

Covers the 11 phase scripts, `_bootstrap.py`, and `run_all.sh`. Read Part 1
first: the cross-cutting patterns explain roughly half of every individual
script, and once you know them the phases read quickly.

---

# PART 1 — The five design patterns that shape every script

## 1.1 One shared entry point (`_bootstrap.py`)

Every phase begins with the same three lines:

```python
cfg, logger, args = setup(PHASE, "description")
summary = PhaseSummary(PHASE, cfg, next_phase="...")
watch = Stopwatch()
```

`setup()` does four things identically everywhere: argument parsing, config
loading and validation, global seeding, logger creation.

**Why it exists.** The comment in the file states the real reason: *"an ablation
run and a baseline run go through exactly the same code path."* If each script
parsed its own arguments and seeded its own RNG, a variant run could differ from
the baseline in some way nobody intended, and the comparison would be
meaningless. Centralising it makes the two runs differ *only* in config.

**Three details worth noticing:**

- `PROJECT_ROOT` is inserted into `sys.path` so `src` imports work regardless of
  the directory you invoked from. Otherwise `python scripts/04_...` and
  `cd scripts && python 04_...` would behave differently.
- Thread limits (`OMP_NUM_THREADS=1` etc.) are set **before** NumPy/sklearn are
  imported. These variables are read at import time, so setting them later has
  no effect. Pinning to 1 thread also removes a source of run-to-run
  non-determinism in BLAS reductions — floating-point addition isn't
  associative, so a different thread count can change the last bits of a result.
- `validate_config()` runs **before any work**, with the comment: *"A bad n_mels
  should cost one second, not twenty minutes of feature extraction followed by a
  shape error."* Fail fast on cheap checks.

## 1.2 Artifact-driven phases

No phase passes anything to the next in memory. Each writes to disk; the next
calls `require_artifacts([...])` and reads. `run_all.sh --from 04` therefore
just works.

**Why.** Three payoffs. Resumability (a crash in phase 08 doesn't cost you the
40-minute feature extraction). Auditability (every intermediate is inspectable
after the fact). And it forces the interface between phases to be explicit — you
cannot accidentally depend on some variable that happened to be in scope.

The cost is disk and serialisation time. For a dataset this size that trade is
obviously right; for a 10 TB corpus it wouldn't be.

## 1.3 `PhaseSummary` — the audit system

This is the most distinctive thing in the codebase. Every phase accumulates
structured metadata and writes it as JSON + Markdown:

| Method | Records | Purpose |
|---|---|---|
| `add_finding(k, v, why)` | a number and *why it matters* | Feeds the dashboard and the paper |
| `add_assertion(name, bool, detail)` | a check that should hold | Turns invariants into recorded pass/fail |
| `add_warning(msg)` | something wrong but not fatal | Surfaces problems instead of burying them in a log |
| `add_claim_verdict(id, statement, verdict, evidence, note)` | a hypothesis and its outcome | The falsification ledger |
| `touch_split(*names)` | which data splits this phase read | **The leakage audit trail** |
| `add_table(name, rows)` | tabular results | Becomes a LaTeX table in phase 10 |
| `add_artifact(path, ...)` | files written | Provenance |

**Why `touch_split` is the clever one.** Test-set leakage is usually invisible —
it's an absence, a thing you failed to do. This turns it into a positive,
machine-checkable record. Phases 01–05 and 07 record `train`/`val`; **phase 06
is the only phase whose summary contains `test`** for model selection purposes,
and phase 10 re-runs an assertion over all summaries to confirm it. When your
examiner asks "how do you know you didn't tune on the test set?", the answer
isn't "I was careful" — it's a field in every phase's JSON.

**Why claim verdicts matter.** Each is declared with a statement, an
automatically-computed verdict, evidence, and a note explaining what a negative
result would mean. Phase 10 rolls them up and adds:

> *"A contradicted claim reported honestly is a result; a contradicted claim
> quietly dropped is misconduct."*

The verdicts are computed from thresholds *in code*, before you see them. That's
what stops post-hoc rationalisation: you can't decide the result was "weak
support" after seeing it, because the threshold was written first.

## 1.4 Fail loudly, degrade gracefully

Two different behaviours, applied deliberately:

- **Hard failure** (`return 1`, pipeline stops) when the result would be
  meaningless: no sub-databases found, empty census, no segments produced, no
  models to evaluate, no segment survived the confidence threshold.
- **Skip with a warning** when one branch is unavailable but the rest is still
  valid: a missing feature domain, a missing CNN checkpoint.

The distinction: does continuing produce a *wrong* answer or a *partial* one?
Wrong → stop. Partial → continue and record the gap.

## 1.5 Numbers are never retyped

Phase 10 generates the LaTeX tables from the same JSON the experiments wrote.
Its docstring frames this as an exam answer:

> *PROFESSOR Q: "How do you know the numbers in the paper match the
> experiments?" A: Because they are generated from the same JSON artifacts the
> experiments wrote... Nobody ever copies a number by hand, so the usual failure
> mode — a table that was correct three runs ago — cannot occur.*

---

# PART 2 — The phases

## Phase 00 — Download and census

**Does:** downloads the six sub-databases, verifies them, and builds
`raw_census.csv` — one row per recording (id, sub-database, label, duration,
sample rate) — plus dataset statistics.

**Why a census file at all.** Everything downstream reads this table instead of
walking the filesystem. The set of recordings used is then fixed and auditable
from one artifact. Walking the directory each time would mean an added or
corrupted file silently changes your dataset between phases.

**Checks it runs:**

- `all_subdatabases_present` — hard failure if not.
- `wav_header_counts_match` — every `.wav` has a `.hea`. A mismatch means a
  partial download, which would otherwise show up much later as a confusing
  load error.
- `single_sample_rate` — the whole pipeline assumes one rate; multiple rates get
  a warning even though phase 01 resamples anyway.

**The important part.** At the end it computes prevalence per sub-database and
takes the spread:

```python
spread = max(prevalences) - min(prevalences)   # 68.9 pp
if spread > 30:
    summary.add_note("... site identity is predictive of the label ...")
```

This is the entire paper's thesis, detected automatically in the *first* phase,
before any modelling. The design decision worth noticing is that the confound
is identified at data-inspection time and then tracked through every subsequent
phase — stratification in 01, ARI-vs-site in 03, per-site metrics in 06 — rather
than discovered by accident at the end.

## Phase 01 — Preprocess, split, segment

**Does:** builds the splits, then for each recording runs
resample → bandpass → spike removal → normalise → segment, and writes
`segments_{split}.npy` plus `segment_index.csv`.

**Order matters: splits are made first, before any audio is touched.** So the
split cannot possibly be influenced by anything computed from the signals.

**`make_splits` + `assert_no_leakage`.** Splitting is at the **recording** level
and stratified jointly on `(label, subdb)`. `assert_no_leakage` *raises* rather
than warns — the only place in the pipeline that does. That's correct: every
other problem produces a degraded result, but window-level leakage produces a
result that looks great and is worthless.

The joint stratification is why the prevalence confound doesn't get *worse*: if
you stratified on label alone, the test split could end up dominated by one
clinic by chance, and per-site analysis would be uninterpretable. Recorded
deviation is ≤0.07 pp on class and ≤0.24 pp on site.

**`_n_tiny_strata`.** Strata with fewer than 3 recordings can't be split three
ways, so they go entirely to train and the count is reported as a warning. Note
what this does *not* do: silently drop them, or force a split that would put one
recording in test. Disclosed deviation beats hidden compromise.

**Different hop per split** (`hop_for_split`): 50 % overlap on train as
augmentation, non-overlapping on val/test. Overlapping evaluation windows aren't
independent, so bootstrap intervals computed over them would be too narrow. Two
different reasons for two different behaviours, one function.

**Quality gating.** `quality_flags` rejects recordings before segmentation; the
exclusion breakdown (`load_error` / `quality` / `no_segments`) is recorded
separately, and a >5 % exclusion rate triggers:

> *"a high rate silently changes the population being evaluated"*

That is the right worry. If you drop the noisy recordings you're reporting
performance on an easier dataset than you claim.

**`n_spikes_removed` is carried into the segment index.** So spike density is
available later as a possible confound — it's a per-recording property that
might correlate with site.

## Phase 02 — Feature extraction

**Does:** runs three extractors over identical segments; fits scalers on train
only; writes `features_{split}.npy` per domain.

**One loop, three domains.** This is the experimental control. Because all three
read the same `segments_{split}.npy`, any performance difference is attributable
to the representation. Extracting them in separate scripts with separate
windowing would confound representation with preprocessing.

**Scaling fitted on TRAIN ONLY, and persisted.** `fit_scaler(features["train"])`
then applied to all splits, saved to `models/scalers/`. Fitting on the whole
dataset leaks test-set statistics (mean and variance) into training — a subtle
leak that inflates results by a little and is very common in student work.
Saving the scaler means phases 04–08 apply *exactly* the same transform rather
than re-deriving it.

**log-Mel is normalised differently** — per-band statistics, keeping the 2-D
structure, because the CNN needs a map not a vector. Handled as an explicit
branch rather than forcing one scaler interface to cover both.

**Diagnostics that catch specific known failures:**

- `n_degenerate_bands` — mel bands with near-zero training variance. This is the
  FFT-resolution problem: at 2 kHz with a 256-point FFT the bin width is
  7.81 Hz, so beyond ~48 bins inside 25–400 Hz you get empty filters. This check
  is how you'd notice.
- `n_near_constant` features → warning that they "carry no information and
  inflate the dimensionality."
- `pwp_all_bands_populated` — asserts every perceptual band got at least one
  wavelet packet node. This is the check that would catch the bit-reversal
  ordering bug: if you grouped nodes as though they were frequency-ordered, band
  membership would be wrong, and in the worst case a band would end up empty.
- `{domain}_features_non_degenerate` — train feature std > 1e-6. A constant
  feature matrix means the extractor is broken, and everything downstream would
  still "run" and produce a 0.5-MAcc model with no obvious cause.

## Phase 03 — k-means clustering and projections

**Does:** sweeps k ∈ {2..8} per domain, computes validity indices and ARI
against **both** diagnosis and site, fits PCA/t-SNE, writes the projection
figure.

**Fitted on the training split only.** The docstring is explicit about why:

> *"This is an unsupervised analysis, but it is still part of the modelling
> process, and fitting it on the full dataset would let test-set structure
> influence a figure in the paper."*

Most people would consider unsupervised analysis exempt from split discipline.
This is stricter, and correct.

**log-Mel is flattened and PCA-reduced before k-means**, with the reason logged
inline: *"Euclidean distance in 6000-D is dominated by noise."* This is distance
concentration — in high dimension the ratio between the nearest and farthest
neighbour distances tends to 1, so k-means becomes meaningless. Reducing first
is not a shortcut, it's a prerequisite.

**The design decision that makes the phase valuable:** `sweep_k(...,
y_true=labels, groups=sites)` computes ARI against the diagnosis *and* against
the recording site. Without the second reference, "ARI = 0.018" says only
"clustering didn't find the classes" — uninformative. With it, "class 0.018 vs
site 0.201" says the dominant variance encodes acquisition condition. A null
result is converted into a positive finding by adding a second hypothesis.

**Claim C1 is scored automatically:**

```python
if best_ari > 0.3:    verdict = "supported"
elif best_ari > 0.1:  verdict = "weak"
else:                 verdict = "contradicted"
```

Thresholds fixed in code before the result. Actual best ARI 0.028 →
**contradicted**. And the accompanying note reframes it constructively: a
contradicted C1 is what makes the supervised comparison in 04–06 non-trivial,
because it establishes the class boundary is a thin supervised direction rather
than the dominant structure.

## Phase 04 — SVM and random forest

**Does:** grid search per (model, domain) pair, evaluates on validation, picks a
decision threshold, saves models. **Does not open test.**

**`GroupKFold` + stratification, grouping by `recording_id`.** Same reasoning as
the split: windows of one recording must not straddle a CV fold, or the CV score
is optimistic and you select the wrong hyperparameters.

**Scored by balanced accuracy** — because that's MAcc, the metric the task is
defined by. Selecting on accuracy and reporting MAcc would be selecting for the
wrong thing on a 3.87:1 imbalanced dataset.

**`check_overfitting(report)`** compares train to CV score and emits a warning.
This is what surfaced the RF gaps (0.143 and 0.198) that the paper reports in
its limitations. The code found them; the paper didn't hide them.

**Threshold tuning on validation only**, then frozen:

```python
threshold, threshold_score = find_best_threshold(
    aggregated["y_true"], aggregated["y_prob"], metric="macc")
```

Note it's tuned on **recording-level** aggregated probabilities, matching how
the model will be used. And phase 06 reads this saved threshold rather than
re-tuning — which is the whole point. Tuning the threshold on test would be a
soft form of test-set fitting that's easy to do accidentally.

**Both segment-level and recording-level validation metrics are computed.** The
gap between them tells you how much aggregation is helping.

**The closing note** states plainly: *"The test split has not been read.
Hyperparameter and threshold choices are now frozen."*

## Phase 05 — CNN training

**Does:** trains the log-Mel CNN, early-stops on validation MAcc, restores best
weights, writes training curves.

**`class_weighting`** via inverse frequency — the same imbalance strategy as the
classical models, so the comparison stays fair.

**Best-epoch checkpointing, not last-epoch.** Stated explicitly in the closing
note. Using the final epoch would report a model that has already started
overfitting.

**Early stopping monitored on validation MAcc, not loss.** Loss and balanced
accuracy don't peak at the same epoch under class weighting; monitoring the
metric you actually care about is correct.

**The Grad-CAM constraint is logged at training time:**

```python
logger.info(f"Last conv feature map: {spec['feature_map_shape']} "
            "(this is the Grad-CAM resolution)")
```

This is unusual and good. Grad-CAM's spatial resolution is fixed by the
architecture, so the limit on what phase 08 can claim is determined *here*, in
phase 05, and recorded as `gradcam_feature_map_shape`. It's why the paper draws
only temporal conclusions from Grad-CAM and gets all frequency evidence from PWP
SHAP.

**`diagnose_training(history)`** inspects the curves and emits warnings
(divergence, plateau, train/val gap), plus an explicit
`train_val_gap_acceptable < 0.20` assertion.

**A CUDA warning if it silently falls back to CPU** — worth having, since the
symptom is otherwise just "this is taking forever."

## Phase 06 — Test-set evaluation

**Does:** everything, once. This is the only phase that opens test for
evaluation, and the docstring says so: *"Nothing is tuned here; the phase reads
models, computes numbers, and stops."*

**The audit line:**

```python
summary.touch_split("test")   # THE audit line
```

**What it computes, and why each piece:**

1. **Trivial baselines.** `always_normal` gives 0.795 accuracy at 0.500 MAcc.
   Recorded as a finding with the note *"Any reported accuracy must be read
   against this number, which is why MAcc is our primary metric."*
2. **Both aggregation rules.** `_evaluate_one` computes mean-probability *and*
   majority-vote, with the comment: *"so the reader can see the choice does not
   decide the conclusion."* This is a mini robustness check on a methodological
   choice that could otherwise look arbitrary.
3. **Bootstrap CIs resampled over recordings** — the correct unit of
   independence.
4. **All pairwise McNemar tests + Holm–Bonferroni.** McNemar because the models
   are evaluated on the same recordings (paired data); Holm because 10
   comparisons at α = 0.05 would give ~40 % chance of a spurious hit.
   `effect_size_cohens_h` is attached to each, so significance and magnitude are
   reported together.
5. **Per-site breakdown of the best model** — the shortcut check.
6. **Calibration curves and ECE.**
7. **Literature comparison** with the non-comparability caveat attached in code,
   not left to the writing stage.

**Claim C2 scoring** has a three-way rule that handles the awkward case:

```python
if p_value < 0.05 and difference > 0:        verdict = "supported"
elif abs(difference) < 0.03 and p >= 0.05:   verdict = "weak"
else: verdict = "contradicted" if difference < 0 else "weak"
```

and the note pre-writes the honest conclusion: if not significant, then the
compact CNN and a well-tuned classical model are indistinguishable, *"which is
itself worth reporting, since the classical model is far cheaper to train and to
explain."*

**The per-site assertion is the one that fired:**

```python
summary.add_assertion("no_gross_site_shortcut", site_spread < 0.25, ...)
if site_spread >= 0.25:
    summary.add_warning(f"Per-site MAcc varies by {site_spread:.3f}. "
                        "Report this prominently: ...")
```

Actual spread 0.509 → assertion **failed**, warning raised. The threshold was
set before the result, and the code's own instruction was to report it
prominently — which the paper does, as its headline. This is the single best
example in the codebase of the falsification design paying off.

## Phase 07 — SHAP

**Does:** TreeSHAP (exact) for the forests, KernelSHAP (approximate) for the
SVMs, then projects attributions onto frequency so different feature domains can
be compared.

**Explanations computed on test predictions, background drawn from train.**
Reasoning given in code: *"Explaining training predictions would describe what
the model memorised rather than how it generalises."* SHAP explains a deviation
from a reference distribution, so the background choice materially changes the
values — using training data as background is the standard, defensible choice.

**Exactness is tracked, not glossed.** `result["exact"]` is stored per model, a
warning is attached to every KernelSHAP run, and the closing note explains the
asymmetry:

> *MFCC frequency attributions are an approximation: cepstral coefficients are a
> DCT of the log-mel spectrum, so SHAP mass is redistributed through the
> magnitude of the DCT basis and the sign is discarded. PWP attributions are
> exact, because each PWP feature belongs to exactly one frequency band by
> construction. Where the two disagree, the PWP result is the reliable one.*

Two levels of approximation are distinguished: the *estimator* (Kernel vs Tree)
and the *frequency mapping* (DCT redistribution vs exact band membership). They
are independent, and conflating them would be easy.

**The frequency projection is the design idea.** Four models on two different
feature spaces can't be compared feature-by-feature. Projecting onto Hz gives
common ground, and `compare_frequency_profiles` then measures agreement.

**Claim C4 scoring:** supported >0.7, weak >0.4, else contradicted. Actual 0.514
→ **weak**. The note explains why agreement would be strong evidence: *"Agreement
across models with different inductive biases and different SHAP algorithms is
far stronger evidence than any single explanation. Disagreement would mean at
least one model is reading an artefact."*

**`n_models_peaking_in_murmur_band`** counts models peaking in 100–300 Hz where
systolic murmur energy is clinically expected. All four peaked at 25 Hz instead —
0/4. Another pre-registered expectation that didn't hold.

⚠️ **The `rank_stability` check is commented out** (lines ~148–166). See Part 3.

## Phase 08 — Grad-CAM and sanity checks

**Does:** Grad-CAM over the whole test set, class-averaged maps, and **three**
sanity checks.

The docstring frames the purpose precisely: these are *"two checks that decide
whether any of it means anything."*

**Check 1 — Adebayo model randomisation.** Recompute Grad-CAM from a randomly
re-initialised network. If the maps are similar, they're driven by architecture
and input statistics rather than learned weights. Threshold |r| < 0.30; actual
0.051 → passes. The failure branch is written and would have said:

> *"Every claim in the explainability section must be withdrawn or heavily
> qualified."*

That's a commitment made in advance, in code.

**Check 2 — layer sensitivity.** Does the conclusion survive choosing a
different target layer? Reported as a table.

**Check 3 — energy confound.** Correlate the temporal attribution profile with
the RMS envelope. This one is specific to *this* signal: S1 and S2 are the loud
events, so a loudness-driven map would confound any claim about cardiac phase.
Threshold 0.6; actual 0.068 → passes.

**Grad-CAM++ as a robustness variant.** Computed on the first 500 windows and
correlated with Grad-CAM. Actual 0.392 — this is the check that *didn't* pass
convincingly, and it's reported as a limitation rather than dropped.

**The resolution disclaimer** is attached here, referencing the shape recorded
in phase 05: temporal claims only, frequency claims come from phase 07 where the
band mapping is exact.

## Phase 09 — Cardiac-cycle alignment

**Does:** segments test audio into S1/systole/S2/diastole, measures attribution
mass per state against uniform and shuffled nulls, stratifies by outcome.

The docstring states the ambition: *"the phase that turns 'the model looks at
the right places' from an assertion into a hypothesis test."*

**Honest about the segmenter, loudly and repeatedly.** It's an envelope-based
labeller, not the Springer HSMM — said in the docstring, logged at runtime, and
noted in the summary. The defence is that it's used **only at evaluation time,
never during training**, so a noisier segmenter adds variance and makes the test
*harder* to pass. It cannot manufacture a positive result.

**The inclusion rate is treated as a first-class number:**

> *"Everything below the threshold is EXCLUDED from the alignment analysis, and
> this number is the honest denominator."*

66.5 % usable. A warning fires below 50 %.

**Physiological plausibility assertion:** mean detected heart rate must be
50–120 bpm. If the segmenter were locking onto something other than the cardiac
cycle, this would catch it. Actual 82 bpm.

**The enrichment statistic.** Attribution mass divided by time fraction, so
E = 1 is exactly uniform temporal attention. The closing note explains why this
normalisation is the point: *"'X% of attention was in systole' is meaningless
until compared against the fraction of time systole occupies."*

**Two nulls, not one.** The analytic uniform null (E = 1) and a permutation null
that shuffles the profile in time, preserving sparsity and dynamic range while
destroying any relation to the cardiac cycle. The second is the harder and more
honest test.

**Claim C5 scoring** requires three conditions simultaneously for "supported":

```python
if enrichment > 1.15 and fraction_significant > 0.5 and effect_size > 0.5:
```

Actual: 1.056, 33 %, d = 0.131 → **weak**. Requiring effect size alongside
significance is what prevents "p < 0.05 therefore the model is interpretable".

**The discriminating comparison — TP vs FP.** This is the sharpest idea in the
codebase:

```python
summary.add_finding("tp_minus_fp_enrichment", tp - fp,
    "Positive values mean the model's systolic focus is stronger when it is "
    "right than when it is wrong — evidence that attention tracks real evidence "
    "rather than being a fixed habit of the architecture.")
```

Actual −0.030. The systolic lean is marginally *stronger* on false alarms.
Enrichment alone could be an architectural habit; the TP−FP contrast is what
separates "attends to systole" from "uses systolic evidence". It's the check
that sinks the interpretability claim, and it was written before the answer was
known.

**Example figures are drawn only from confidently-segmented windows** — because
shading cardiac states on a window the segmenter couldn't parse would be
misleading. And a segmenter example goes to supplementary, so the substitution
is *auditable* rather than merely disclosed.

## Phase 10 — Report assets

**Does:** collects all phase summaries, renders `PIPELINE_STATUS.md`, generates
LaTeX tables from result JSON, copies figures, rolls up claim verdicts.

**Re-runs the test-split audit** across all summaries as an assertion, so the
leakage guarantee is verified at the end from the recorded evidence rather than
assumed.

**Tables generated from JSON** — the "no number is retyped" guarantee.

**Claim roll-up** with the misconduct note quoted in §1.3.

---

# PART 3 — Things I noticed that you should know about

Four are real, one is cosmetic. None affects a number in the paper.

### 1. The SHAP `rank_stability` check is commented out

In `07_explain_shap.py`, lines ~148–166 are commented, and `"rank_stability":
stability` is commented out of the payload. So the check does not run.

But this warning **does** still fire on every SVM:

```python
summary.add_warning(
    f"{tag}: KernelSHAP is a Monte-Carlo approximation. Its per-feature "
    "ranking is only as stable as the rank_stability check reports.")
```

It points at a check that produces nothing. Either re-enable the block or reword
the warning — as it stands, a reader of the summary would go looking for a
stability number that isn't there. The paper doesn't claim stability, so no
result is wrong; it's the audit trail that's inconsistent.

Worth knowing for the exam: this is exactly the check that would tell you whether
your top-2 PWP features (entropy of bands 0 and 9) are a stable finding or an
artifact of one background sample. Since it didn't run, "these are the features
the model relies on most" is the strongest claim available — which is what the
paper says.

### 2. Phase 10 is internally numbered 11

The file is `10_build_report_assets.py` and `run_all.sh` calls it phase 10, but
inside: `PHASE = "11_build_report_assets"`, the docstring says "Phase 11", and
the run instruction says `python scripts/11_build_report_assets.py`. Also
`next_phase="10_run_ablations"` in phase 09 points at a script that isn't in the
zip.

Consequence: the summary lands in `reports/phase_11_build_report_assets/` while
`run_all.sh`'s failure message would tell you to look in
`reports/phase_10_build_report_assets/run.log`. Cosmetic, but it'll waste two
minutes at exactly the wrong moment.

### 3. `escape_latex` is the source of the table bug I fixed earlier

This is the root cause of the `95\textbackslash{}\%` and `\$n\$` garbage in the
original generated tables. The captions and column names contain
*pre-escaped* LaTeX:

```python
caption="... 95\\% bootstrap confidence intervals ..."
column_names=[..., "95\\% CI", ...]
column_names=["Sub-database", "$n$", ...]
```

and then `escape_latex` escapes them **again** — the backslash becomes
`\textbackslash{}`, the `%` becomes `\%`, the `$` becomes `\$`. The fix is to
escape only the data cells (which come from JSON and genuinely need it) and
leave captions and column names alone, since those are authored LaTeX. Right now
`escape_latex` is applied to both.

I hand-corrected the output tables in the paper, so your `.tex` files are clean —
but if you re-run phase 10 it will overwrite them with the broken versions
again. Fix the function first, or keep the corrected tables out of the
regeneration path.

### 4. `run_all.sh --skip-ablations` is documented but not implemented

The header advertises it; the `case` statement doesn't handle it, so passing it
hits `*) echo "Unknown option"; exit 1`. Same for `--skip-download`, which *is*
handled (`SKIP_DOWNLOAD`) — that one works.

### 5. Leftover scaffolding in phase 04

`model, report = run_grid_search(...)  # was: search, report`, a stray comment
`# every later 'search.best_estimator_' becomes 'model'`, and a blank
double-indented gap after `y_prob_val`. Harmless, but a reader will pause on
them.

---

# PART 4 — The five-sentence version, for an oral

The pipeline is eleven artifact-driven phases sharing one entry point, so that
every run — baseline or ablation — goes through identical code and differs only
in config. Each phase writes a structured summary recording its findings,
assertions, warnings, and crucially which data splits it read, so the guarantee
that only phase 06 opened the test set for evaluation is a machine-checkable
field rather than a promise. Five claims were declared in advance with
thresholds fixed in code, and the pipeline scored them itself: C1 contradicted
(clusters track site, not diagnosis), C2 contradicted (CNN indistinguishable
from SVM), C3 supported, C4 weak, C5 weak. The per-site assertion
`no_gross_site_shortcut < 0.25` failed at 0.509 and its own warning text
instructed us to report it prominently, which is why it became the paper's
headline. And every table in the report is generated from those same JSON
artifacts, so no number was ever retyped.
