# Oral Exam Preparation — Anticipated Questions and Answers

This file collects the questions an examiner is most likely to ask, with the
answer and, where relevant, the file and function that backs it up. Each answer
is written so you can give the short version in one sentence and expand if
pressed.

The same material appears as `PROFESSOR Q:` comments inside the source, so you
can point at the code while answering.

---

## 1. Task, data and framing

**Q: Why heart sounds rather than a standard audio-classification benchmark?**

Three reasons. First, it is a domain where interpretability has real
consequences, which makes the XAI section a genuine contribution rather than a
box-ticking exercise. Second, PhysioNet/CinC 2016 has published baselines, so
the results are anchored to something. Third, the physiology gives us a
*testable prior*: murmurs are mostly systolic, so we can state in advance where
a well-behaved model's evidence should concentrate, and then test it. On ESC-50
there is no equivalent prior.

**Q: What exactly is the classification task?**

Binary: Normal vs Abnormal phonocardiogram, at recording level. "Abnormal"
in this corpus means confirmed cardiac pathology (valve disease, coronary
artery disease), confirmed by echocardiography or clinical diagnosis, not just
"a murmur was audible".

**Q: How large is the dataset and how imbalanced?**

About 3,126 recordings across six sub-databases, roughly 20-25% abnormal, so
about 3:1. Recording lengths range from ~5 s to ~120 s. Everything is 2 kHz mono.

**Q: What are the six sub-databases and why do they matter?**

Each was collected by a different group, at a different site, with different
equipment. Crucially, abnormal prevalence varies enormously between them
(training-a is majority abnormal; training-e is overwhelmingly normal). That
means *site identity predicts the label*, so a model can score well by
recognising the recording device. This is the central confound of the dataset
and three separate design decisions address it: joint stratification of splits,
per-recording amplitude normalisation, and per-site performance reporting.
See `src/data/splits.py` and `per_group_metrics` in `src/evaluation/metrics.py`.

---

## 2. Preprocessing

**Q: Why the 25–400 Hz band?**

Below 25 Hz the signal is respiration, movement and DC drift. Above 400 Hz there
is essentially no cardiac acoustic energy — what remains is stethoscope and
ambient noise. The band covers S1 (~25–45 Hz), S2 (~55–75 Hz) and murmur energy
up to a few hundred Hz.

**Q: Why zero-phase filtering?**

Because the entire explainability claim is about *when* in the cardiac cycle
the model attends. A causal filter has frequency-dependent group delay, so it
would shift 200 Hz murmur energy relative to 40 Hz S1 energy. `filtfilt` runs
the filter forwards and backwards, cancelling the phase response exactly.
There is a demonstration of this in `tests/test_features.py::
test_zero_phase_filtering_preserves_peak_position`.

**Q: Why second-order sections instead of `b, a` coefficients?**

A 4th-order bandpass is an 8th-order filter, and at our very low normalised
cutoff (25 Hz is 0.025 of Nyquist) direct-form coefficients are numerically
unstable. SOS is the robust factorisation.

**Q: What is spike removal and doesn't it destroy information?**

It suppresses friction transients from the stethoscope diaphragm being brushed
(Schmidt et al. 2010). The removed spans are typically 10–50 ms of clipped
non-cardiac noise. Leaving them in is worse: they are the loudest events in the
recording, so they dominate every energy descriptor and the normalisation
constant. We count and report removals.

**Q: Why normalise per recording rather than globally?**

Because gain is a site artefact. Different clinics used different preamps.
Global normalisation would preserve those differences, and gain correlates with
site, which correlates with prevalence — a direct shortcut. Per-recording
z-scoring removes that channel. We lose absolute intensity, but murmur *grade*
is judged relative to S1/S2 in the same recording anyway.

---

## 3. Segmentation into windows

**Q: Why 3-second windows?**

At 60–100 bpm a cardiac cycle is 0.6–1.0 s, so 3 s always contains at least
three complete cycles. A murmur present in every cycle is therefore guaranteed
to be represented, and the window is robust to the phase at which we happen to
cut. Ablation A6 tests 1 s windows, which can straddle an S1–S2 boundary and
contain no complete systole.

**Q: Why not segment into cardiac cycles instead?**

Two reasons. Practically, cycle segmentation needs a segmenter, and a segmenter
that fails on noisy recordings would silently bias which recordings survive.
Methodologically — and this is the important one — our XAI claim is that the
CNN discovers the systolic window *without being told where it is*. If we had
already cut at cycle boundaries, that discovery would be built into the input
and the claim would be circular.

**Q: You overlap training windows by 50%. Isn't that leakage?**

Not across splits. Splits are frozen at recording level, so overlapping windows
always land in the same split. Within training, overlap is just augmentation.
We deliberately use non-overlapping windows for validation and test, because
overlapping evaluation windows are statistically dependent and would narrow
confidence intervals without adding information. See `hop_for_split` in
`src/data/segmentation.py`.

---

## 4. Splits and leakage

**Q: How do you prevent patient leakage?**

Splits are made on `recording_id`, never on segments, and `assert_no_leakage`
raises an `AssertionError` — killing the run — if any ID appears in two splits.
The assertion result is recorded in the phase-01 summary, so it is an artifact
you can point at, not a verbal promise.

**Q: Why stratify on sub-database as well as on class?**

Because otherwise the test set could end up dominated by one clinic, and you
would be measuring cross-site transfer while believing you were measuring
pathology detection. We form the cross product of (label, subdb) into a
composite stratum and split within each stratum.

**Q: The challenge had an official test set. Why didn't you use it?**

It was never released — it stayed on the challenge server. Only the six
training sub-databases are public. So the honest options were to invent our own
split or hold out a whole sub-database. We do the former for the main results
and additionally report leave-one-sub-database-out as a cross-site stress test.
The paper states explicitly that our numbers are therefore **not** directly
comparable to published challenge scores, and the comparison table carries a
`comparable` column saying so.

**Q: How do I know your test set was not used for model selection?**

Every phase summary records which splits it read. Phases 03, 04 and 05 record
train and val only; phase 06 is the first to record test. The dashboard
(`reports/PIPELINE_STATUS.md`) prints that audit trail automatically. See
`touch_split` in `src/utils/summary.py`.

---

## 5. Features

**Q: MFCCs were designed for speech. Why use them on heart sounds?**

Two honest answers. They are the standard baseline in the CinC 2016 literature,
so including them makes our results comparable. And the cepstrum is really just
a decorrelated, compressed description of spectral *shape*, which is what
murmur-vs-no-murmur largely is. The part that transfers poorly is the mel
warping, and we address that head-on.

**Q: Is the mel scale even meaningful below 400 Hz?**

Barely, and we say so. Mel is approximately linear below ~1 kHz, so our
filterbank is very nearly a linear filterbank. Ablation A2 swaps in an
explicitly linear bank. If performance is unchanged, that is a legitimate
negative result: the "mel" in log-Mel is doing no work for PCG.

**Q: Why only 32 mel bands? Speech uses 40–128.**

Because of the FFT resolution. At sr = 2000 and n_fft = 256, the bin width is
7.8 Hz, so between 25 and 400 Hz there are only about 48 usable bins. Asking
for 64 filters over 48 bins produces empty, all-zero filters that inject
structural zeros into every feature vector. librosa only warns; our config
validator turns it into a hard error. See `validate_config` in
`src/config/schema.py`.

**Q: Why do you aggregate MFCCs over time instead of feeding the sequence?**

Because SVM and Random Forest need a fixed-length vector — they have no notion
of a time axis. That aggregation is exactly where the classical branch loses
temporal structure, and that loss is what the CNN branch avoids. The contrast
between the two is one of the findings, not an implementation accident.

**Q: What are the deltas for?**

Velocity and acceleration of the cepstral trajectory. Heart sounds are
transients with sharp onsets, so the rate of change carries onset information
the static coefficients miss. Ablation A4 removes them.

**Q: Why wavelet packets when you already have two spectral representations?**

Because MFCC and log-Mel are both STFT-based and assume local stationarity in
each window. A PCG is the opposite: S1 and S2 are 10–50 ms transients and
clicks are near-impulsive. The wavelet packet transform tiles the
time-frequency plane adaptively, which is the textbook motivation for using it
on PCG (Safara et al. 2013). This is why the three feature domains are
genuinely different rather than three flavours of the same thing.

**Q: What makes your wavelet packet features "perceptual"?**

The grouping step. A plain level-6 decomposition gives 64 uniform 15.6 Hz
subbands — a linear partition. We merge those into 12 bands whose edges follow
mel spacing, so the partition is narrow at low frequency and wide at high
frequency. We implement it by grouping rather than by pruning the tree, which
keeps the node-to-band mapping fixed across segments — a property SHAP needs.

**Q: What is the wavelet-packet frequency-ordering problem?**

A packet tree does not produce nodes in frequency order: at each level the
high-pass branch reverses the frequency axis of its children, so the natural
(Paley) ordering is a bit-reversed permutation of frequency ordering. pywt
returns natural order by default. Using it while assuming frequency order
silently scrambles the spectrum and nothing crashes. We pass `order='freq'`
and the validator refuses any other value.

**Q: Why db4 and level 6?**

db4 has compact support and four vanishing moments, a good match for sharp
onsets. Level 6 gives 15.6 Hz subbands, about 25 of which fall in our band —
enough that the perceptual grouping actually merges several nodes per band. At
level 5 there would be only ~13 nodes and the grouping would be cosmetic.
Ablation A7 substitutes Haar, which has one vanishing moment and poor frequency
localisation.

**Q: You mention best-basis selection but don't use it. Why?**

Because an adaptive basis differs from segment to segment, so feature index *k*
would mean different things in different samples. That breaks both the
classifier's input contract and SHAP's interpretation. We compute and report
the Coifman–Wickerhauser cost so we can say we considered it.

**Q: Where do you fit the scaler?**

On the training partition only, then applied unchanged to val and test. Fitting
on everything leaks test statistics into training. Inside cross-validation the
scaler is refitted per fold via an sklearn `Pipeline`, because otherwise every
fold would share statistics computed over the fold being validated.

---

## 6. Clustering (the unsupervised requirement)

**Q: What is unsupervised clustering for in a supervised project?**

It answers a question the classifiers cannot: is the Normal/Abnormal
distinction *geometrically present* before any label is used? If k=2 k-means
recovered the diagnosis, a classifier would be mostly reading off existing
structure. If it does not — which is what we expect — the class boundary is a
thin supervised direction inside a space whose dominant variance is something
else.

**Q: Your ARI is near zero. Isn't that a failed experiment?**

No, and this is the point I would emphasise. k-means optimises within-cluster
variance, and the dominant variance in PCG features is acquisition condition,
not pathology. We test that interpretation directly by *also* measuring cluster
alignment with sub-database. If clusters track site better than diagnosis, we
have identified what the feature space is organised by — a much stronger
statement than "clustering did not work". See `cluster_validity` and
`interpret_validity` in `src/clustering/validity.py`.

**Q: Why both PCA and t-SNE?**

They answer different questions and each misleads alone. PCA is linear with
interpretable axes and tells you how much variance lives in a few directions.
t-SNE preserves local neighbourhoods and can reveal non-linear structure PCA
flattens — but its global geometry is meaningless: inter-cluster distances and
cluster sizes in a t-SNE plot should not be read at all.

**Q: Could you use the t-SNE coordinates as features?**

No. t-SNE is transductive — it has no `transform` method, so it must be fitted
on exactly the points being plotted. Using it as a feature extractor would
embed test points using test-point neighbourhoods.

**Q: Why PCA-reduce the log-Mel maps before clustering?**

Flattened they are 32×188 = 6,016 dimensions. Euclidean distance in that many
dimensions is dominated by noise accumulated across dimensions — the
concentration-of-distances problem — so k-means there is close to meaningless.
We reduce to 50 components first.

---

## 7. Models

**Q: Why SVM and Random Forest specifically?**

The classical branch needs to be a credible baseline, not a leaderboard entry.
These two span very different hypothesis classes — a global, distance-based,
maximum-margin model versus a local, axis-aligned ensemble — so agreement
between them is informative. And RF admits an *exact* SHAP computation, which
anchors the explainability comparison.

**Q: Why StratifiedGroupKFold?**

Because the unit of observation is a segment but the unit of independence is a
recording. Segments from one recording share a patient, a stethoscope and a
murmur. Scattering them across folds means the model validates on segments
whose siblings it trained on, and CV scores become optimistic by a wide margin.

**Q: Why score cross-validation on balanced accuracy?**

Because balanced accuracy *is* MAcc = (Se+Sp)/2, the official challenge metric.
Selecting on plain accuracy would pick models that maximise the majority class.

**Q: Your CNN is small. Why not ResNet or a pretrained model?**

With ~3,000 recordings a 25M-parameter network would memorise the data. Ours has
~0.2M parameters, giving roughly 10 training segments per parameter. Transfer
learning from ImageNet is possible but the domain gap is large and it would
obscure the feature-domain comparison that is the point of the study. We state
that as an explicit scope boundary.

**Q: Why global average pooling instead of flattening?**

Flattening a (128, 2, 11) map gives a 2,816-dimensional vector, so the first FC
layer alone would hold ~180k parameters — most of the network's capacity in a
layer that also destroys spatial structure. GAP keeps the head tiny, makes the
model robust to where in the window a murmur occurs, and preserves the
channel-to-class relationship Grad-CAM exploits.

**Q: How do you handle class imbalance?**

Three ways: `class_weight='balanced'` for SVM,
`class_weight='balanced_subsample'` for RF, and inverse-frequency weighted
cross-entropy for the CNN. Weights are normalised to mean 1 so the loss
magnitude stays comparable when weighting is toggled. Ablation A3 turns it all
off and shows accuracy stays high while MAcc collapses toward 0.5.

**Q: Why not SpecAugment?**

Frequency masking could delete the murmur itself, turning an abnormal example
into a mislabelled normal one. It would also distort the frequency-axis
Grad-CAM analysis. We use circular time shift (the window phase is arbitrary,
so the label must be invariant to it) and Gaussian noise instead.

**Q: How do you go from 3-second windows to a recording-level decision?**

Two rules, both reported. Majority vote (ties break to abnormal, because a
missed pathology costs more than a false alarm in screening) and mean
probability, which is our primary rule because it preserves confidence
information. Reporting both makes the choice visible and lets a reader see it
does not decide the conclusion.

---

## 8. Evaluation

**Q: Why is accuracy the wrong headline metric here?**

Because ~78% of recordings are Normal, so an unconditional "Normal" predictor
scores 78% accuracy, 0% sensitivity and 0.50 MAcc. We report that baseline in
the results table so nobody has to do the arithmetic. MCC is also reported,
because it is 0 for any constant predictor.

**Q: Model A got 0.84 MAcc and model B got 0.82. Is A better?**

Not necessarily. With ~470 test recordings a two-point difference is well inside
sampling noise. We report bootstrap confidence intervals and McNemar's test. If
the CIs overlap and McNemar's p is large, the honest conclusion is that the
models are statistically indistinguishable — which is itself a finding,
especially when the simpler model is far cheaper.

**Q: Why McNemar and not a t-test?**

Because both models are evaluated on the same recordings, so the samples are
paired. McNemar uses exactly the right information — the two discordant cells —
and ignores cases where both models agree, which carry no evidence about a
difference.

**Q: You ran several comparisons. Did you correct for multiple testing?**

Yes, Holm–Bonferroni, which is uniformly more powerful than plain Bonferroni at
the same family-wise error rate. With six comparisons an uncorrected 0.05
threshold gives roughly a 26% chance of at least one spurious result.

**Q: What do you bootstrap over?**

Recordings, with replacement — the correct unit of independence. Bootstrapping
over segments would treat the eight segments of one recording as eight
independent observations and produce intervals about √8 times too narrow.

**Q: Are your probabilities calibrated?**

We measure it rather than assume it. Expected Calibration Error is reported per
model. RF probabilities are notoriously under-confident (they are vote
fractions) and a CNN trained with weighted loss is systematically shifted. A
large ECE makes the mean-probability aggregation shakier than majority vote,
and we say so.

**Q: How do you know the model isn't just recognising the recording site?**

We report MAcc per sub-database. Uniform performance across sites argues against
a site shortcut; a bright cell for one site would expose it. The phase-06
summary asserts that the spread stays below 0.25 and warns if it does not.

---

## 9. Explainability

**Q: Why both SHAP and Grad-CAM?**

Because they explain different model families and the *agreement between them*
is the evidence. Two models with completely different inductive biases,
explained by two different algorithms, converging on the same frequency region
is far stronger than any single explanation. Disagreement would mean at least
one is reading an artefact.

**Q: Why TreeSHAP for RF and KernelSHAP for SVM?**

TreeSHAP is exact and polynomial-time for tree ensembles. KernelSHAP is
model-agnostic but costs O(n_background × n_explained), so we subsample and
report the Monte-Carlo uncertainty rather than pretending it is exact.

**Q: SHAP assumes feature independence and your features are correlated. Doesn't
that break it?**

It is a genuine limitation. With correlated features SHAP splits credit among
the correlated group in a way that depends on the background distribution, so
individual rankings within a group are unstable. Our conclusions are therefore
drawn at the level of *groups* — a frequency band, or a coefficient's whole set
of statistics. We also report top-k rank stability across background resamples
(mean Jaccard overlap) so the reader can see how much the ordering moves.

**Q: MFCCs are a DCT of the log-mel spectrum, so a coefficient isn't tied to a
frequency. How do you produce a frequency attribution?**

By projecting SHAP mass through the magnitude of the DCT basis: coefficient *i*
reads mel band *m* with weight |cos(πi(m+½)/M)|. This is an approximation and we
label it as one — it discards the sign, so it answers "which bands feed the
important coefficients" rather than "in which direction". The PWP attribution
needs no such projection, because each PWP feature belongs to exactly one band
by construction. Where the two disagree, PWP wins. See
`map_mfcc_shap_to_frequency` and `map_pwp_shap_to_frequency`.

**Q: Why the last conv layer for Grad-CAM?**

Earlier layers have high spatial resolution but encode generic edges; the last
conv layer encodes class-discriminative structure while still retaining a
spatial grid. That is the standard recommendation, and we verify it: the
layer-sensitivity analysis recomputes maps from every block and checks the
temporal conclusion survives.

**Q: Your Grad-CAM grid is only 2×11. Isn't that too coarse?**

It is a real limitation and we handle it by restricting our claims. On the
frequency axis, 2 cells over 32 mel bands is too coarse to localise a murmur
spectrally, so **we make no frequency claims from Grad-CAM** — the frequency
evidence comes from PWP SHAP, where the mapping is exact. On the time axis, 11
cells over 3 s is ~273 ms per cell, comparable to systole itself, which is
coarse but sufficient to distinguish "attends during systole" from "attends
during diastole". That is the only temporal claim we make.

**Q: Saliency maps are known to be unreliable. How do you know yours mean
anything?**

We run the Adebayo model-randomisation sanity check: recompute Grad-CAM from a
randomly re-initialised copy of the network and correlate with the trained
model's maps. High correlation would mean the maps are architecture artefacts
and the XAI claims would have to be withdrawn. The result is an assertion in the
phase-08 summary, not an optional extra. We also compare Grad-CAM against
Grad-CAM++ so the conclusion is not a property of one variant.

**Q: Isn't the model just attending to the loud parts, which are S1 and S2?**

That is the confound the whole design is built to catch. We score all four
states separately, not "sounds versus silence": S1 and S2 are the loud events,
systole and diastole the quiet intervals between them. An energy-following model
would show high enrichment in S1/S2 and near-zero in systole; a murmur detector
shows the opposite. We also correlate the attribution profile directly against
the RMS envelope (`energy_confound_check`) and report that number.

---

## 10. The alignment metric (the main contribution)

**Q: What exactly does the alignment score measure?**

Given a unit-mass attribution map and an independently obtained cardiac state
map, the score for state *s* is the fraction of attribution mass falling inside
that state. On its own that is meaningless, so we report **enrichment**
E_s = (attribution mass in s) / (fraction of time occupied by s). E = 1 means no
preference; E > 1 means the model attends to that state more than chance.

**Q: Why is that framing important?**

Because "38% of attention was in systole" is uninterpretable until you know
systole occupies 33% of the recording. The ratio is what makes the number mean
something. This is precisely the step most applied XAI papers skip.

**Q: What are your null hypotheses?**

Three. Uniform (a flat profile, which scores E = 1 analytically). Shuffled —
the real attribution values permuted in time, which preserves the map's
sparsity and dynamic range while destroying any relation to the cycle; this is
the strong null and the one the permutation test uses. And state-shuffled, where
the real map is scored against a *different* segment's state map, controlling
for coincidental periodicity.

**Q: You substituted a simplified segmenter for the Springer HSMM. Doesn't that
invalidate the result?**

No, and the direction of the error matters. The segmenter is used **only at
evaluation time**, never during training. A noisier segmenter adds variance to
the alignment estimate and therefore makes our claim *harder* to support — it
cannot manufacture a positive result. We also report a confidence score per
segment, exclude low-confidence segments, and report the exclusion rate, so the
denominator is honest. And we publish an example segmentation figure so the
substitution is auditable rather than merely disclosed.

**Q: How does your segmenter decide which interval is systole?**

Physiology. At rest, systole is roughly a third of the cycle and diastole two
thirds, so inter-peak intervals alternate short–long. We determine the global
phase by comparing the mean length of even-indexed intervals against
odd-indexed ones, which is robust to a single mis-detected peak. This degrades
at tachycardia above ~120 bpm, where the two intervals converge — the
confidence score encodes that and those segments get excluded.

**Q: What is the most informative comparison in the XAI section?**

Stratifying alignment by outcome. If the model attends to systole on true
positives but not on false positives, its attention is tracking real evidence.
If the pattern is identical for both, the "focus" is a property of the
architecture rather than of the diagnosis.

**Q: What if the alignment claim comes out negative?**

Then we report it as negative. That is a publishable, honest result: it would
mean that although the CNN classifies well, its evidence is not localised to the
phase where murmurs occur, and the model should not be described as clinically
interpretable. The verdict machinery in `src/utils/summary.py` records
`contradicted` explicitly so it cannot be quietly dropped.

---

## 11. Engineering and reproducibility

**Q: Why no notebooks?**

Because a notebook's state depends on execution order, which makes results
irreproducible in practice. Every phase here is a script that reads artifacts
from disk and writes artifacts to disk, so any phase can be re-run in isolation
and the pipeline can be resumed from any point.

**Q: How do you guarantee the numbers in the paper match the experiments?**

The LaTeX tables are generated by `scripts/11_build_report_assets.py` from the
same JSON the experiments wrote, and `\input{}` directly into the source. No
number is ever retyped, so the usual failure mode — a table that was correct
three runs ago — cannot occur.

**Q: How do you know an ablation only changed one thing?**

Ablation configs are *partial* files deep-merged onto the base, so only the
declared leaf keys change. The runner additionally computes `config_diff` and
records the exact dotted key paths that differ, so the claim is verifiable from
the artifacts rather than from trust.

**Q: Are your results reproducible?**

Bit-for-bit on the same machine for everything except the CNN, where
`deterministic=True` forces cuDNN into deterministic algorithms at ~10–20%
throughput cost. The seed, config hash, git commit, package versions and GPU
model are recorded in every phase summary.

**Q: What is the config hash for?**

It is a SHA-256 of the semantic config content, so any result can be traced
back to the exact parameter set that produced it. It is invariant to key
ordering and to the working directory.

---

## 12. Limitations to volunteer before you are asked

Volunteering these makes you look in command of the work rather than defensive.

1. **Not comparable to challenge scores.** Our test set is a held-out split of
   the public training data, not the official hidden set.
2. **The segmenter is not Springer's.** Simplified, envelope-based,
   evaluation-only, with a reported exclusion rate.
3. **Grad-CAM frequency resolution is 2 cells.** We therefore make no frequency
   claims from Grad-CAM.
4. **MFCC frequency attribution is approximate.** The DCT projection is
   sign-agnostic.
5. **SHAP with correlated features.** Conclusions are drawn at group level, with
   rank stability reported.
6. **Binary task only.** The corpus does not support murmur subtyping
   (systolic vs diastolic, or by valve), which is what a clinician actually
   wants.
7. **No external validation.** Everything is one corpus; the
   leave-one-sub-database-out result is the closest we get to a transfer test.
8. **Recording quality is uncontrolled.** We exclude only structurally unusable
   recordings, deliberately, because aggressive quality filtering would improve
   headline numbers by removing the hard cases.

---

## 13. Questions to ask yourself while writing the paper

- For every number in the abstract: which JSON file did it come from?
- For every claim: is there a verdict recorded for it in a phase summary?
- For every figure: does the caption state what the reader should conclude, and
  what they should *not* conclude from it?
- Is any negative result missing from the paper that appears in the summaries?











PROFESSOR Q: "What is unsupervised clustering *for* in a supervised project?"
A: It answers a question the classifiers cannot: is the Normal/Abnormal
   distinction *geometrically present* in each feature space, before any label
   is used? If k=2 k-means recovers clusters that align with the diagnosis
   (high ARI/NMI), the representation itself separates the classes and a
   classifier is mostly reading off existing structure. If it does not - which
   is what we expect here - then the class boundary is a thin, supervised
   direction inside a space whose dominant variance is something else
   (recording site, noise level, heart rate). That is a genuinely useful
   negative result: it tells you the classifier is doing real work rather than
   thresholding an obvious cluster, and it sets expectations for how much a
   purely unsupervised approach could ever achieve on this task.

PROFESSOR Q: "Why should I not read a low ARI as failure?"
A: Because k-means optimises within-cluster variance, and the dominant
   variance in PCG features is not pathology - it is acquisition. We test that
   interpretation directly by also measuring cluster alignment with
   sub-database (site). If clusters align better with site than with diagnosis,
   we have identified *what* the feature space is actually organised by, which
   is a stronger statement than "clustering did not work".




PROFESSOR Q: "Why show both PCA and t-SNE?"
A: They answer different questions and each is misleading on its own.
   PCA is a *linear*, distance-preserving-in-the-large projection: it tells you
   how much of the total variance lives in a couple of directions, and its axes
   are interpretable (you can ask which features load on PC1). t-SNE is
   *non-linear* and optimises local neighbourhood preservation: it reveals
   manifold structure PCA would flatten, but its global geometry is not
   meaningful - inter-cluster distances and cluster sizes in a t-SNE plot mean
   essentially nothing, and the layout changes with perplexity. Showing PCA
   alone risks concluding "no structure" when the structure is non-linear;
   showing t-SNE alone risks over-reading artefacts of the embedding.

PROFESSOR Q: "Did you fit t-SNE on all the data?"
A: t-SNE is transductive - it has no ``transform`` method, so it must be fitted
   on exactly the points being plotted. That is fine for a *visualisation*, but
   it means t-SNE coordinates can never be used as features for a classifier;
   doing so would embed test points using test-point neighbourhoods. PCA, being
   a linear map, is fitted on train and applied to the rest.



PROFESSOR Q: "What happens if I set n_mels larger than the number of usable
              FFT bins?"
A: `validate_config` refuses to run. This is not a hypothetical: at sr=2000
   with n_fft=256 there are only ~48 FFT bins between 25 and 400 Hz, so asking
   for 64 mel filters silently produces empty (all-zero) filters. librosa emits
   a warning that is easy to miss, and the resulting feature vector contains
   structural zeros that quietly degrade every downstream model. We turn that
   warning into a hard error with an explanatory message.



PROFESSOR Q: "Why are there six sub-databases and does it matter?"
A: Each sub-database (training-a .. training-f) was collected by a different
   research group, at a different site, with different stethoscopes and in
   different acoustic environments. Class balance also varies wildly between
   them: training-a is roughly 70% abnormal, training-e is roughly 95% normal.
   This matters enormously. If you split randomly without stratifying on
   sub-database, the model can learn to recognise the *recording device* and
   infer the label from it, because device correlates with site and site
   correlates with prevalence. That is a shortcut, not a diagnosis. We
   stratify splits jointly on (label, sub-database) to prevent it, and we
   report per-sub-database performance so the shortcut would be visible if it
   existed.



preprocessing


PROFESSOR Q: "Why zero-phase filtering?"
A: A causal IIR filter introduces a frequency-dependent group delay. Our entire
   explainability argument rests on *when* in the cardiac cycle the model
   attends. If the filter shifted 200 Hz murmur energy by a few milliseconds
   relative to the 40 Hz S1 fundamental, the attribution maps would be
   comparing evidence that had been silently misaligned in time. filtfilt runs
   the filter forwards then backwards, cancelling the phase response exactly
   (at the cost of doubling the effective filter order, which we account for).




PROFESSOR Q: "Doesn't zeroing a span destroy information?"
A: The removed spans are typically 10-50 ms of clipped, non-cardiac
   transient - the diaphragm being brushed. Leaving them in is worse: they
   are the loudest events in the recording, so they dominate every
   energy-based descriptor and the per-recording normalisation constant.
   We count removals and report them, so the cost is visible rather than
   hidden.




PROFESSOR Q: "Why normalise per recording and not globally?"
A: Recording gain in this dataset is a site artefact, not a physiological
   variable. Different clinics used different digital stethoscopes with
   different preamp gains. A global normalisation would preserve those
   gain differences, and since gain correlates with site, and site
   correlates with class prevalence, the model could exploit loudness as a
   proxy for the label. Per-recording normalisation removes that channel
   entirely. The cost is that we discard absolute intensity - but murmur
   *grade* is judged relative to S1/S2 in the same recording anyway, so
   the clinically meaningful information is relative, not absolute.




PROFESSOR Q: "Why fixed windows rather than segmenting into cardiac cycles?"
A: Two reasons, one practical and one methodological.
   Practical: cycle-based segmentation requires a segmenter, and a segmenter
   that fails on noisy recordings would silently bias which recordings survive.
   Methodological: our XAI claim is that the CNN discovers the systolic window
   *without being told where it is*. If we had already cut the signal at cycle
   boundaries, that discovery would be built into the input representation and
   the claim would be circular. Fixed windows keep the cardiac timing latent.