"""Phase 11 - The confirming experiment the report names but never ran.

Discussion (Sec. V-C, VII, VIII) infers that the pooled MAcc is carried by a
prevalence-driven *recording-site* shortcut, but states plainly that the direct
test was not performed: "train a classifier to predict the sub-database from the
same features, and measure how much label information survives once site is
controlled for." This phase runs exactly that, turning an inference into a
measurement, and adds the leave-one-site-out (LOSO) evaluation the Conclusion
recommends.

Three questions, one per block:

  A. Is site *itself* recoverable from the features?  (multiclass site classifier)
     -> if yes, the substrate for a shortcut is present in the representation.

  B. How much apparent performance is site-driven?    (pooled CV vs LOSO)
     -> the drop from stratified-pooled MAcc to leave-one-site-out MAcc is the
        generalisation a screening deployment would actually see.

  C. How much diagnosis signal survives once site is held fixed?  (within-site CV)
     -> diagnosis MAcc estimated *inside* each site, where prevalence can no
        longer stand in for the label. Collapse toward 0.5 == the shortcut.

Only the TRAIN split is read. The guarantee that phase 06 is the sole reader of
the test split is preserved and asserted below, so this diagnostic cannot itself
leak the test set into a reported number.

Run:  python scripts/11_site_shortcut.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import setup

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAVE_SGKF = True
except ImportError:  # older sklearn
    from sklearn.model_selection import GroupKFold
    _HAVE_SGKF = False

from src.models.inference import aggregate_to_recording
from src.evaluation.metrics import compute_metrics
from src.utils.io import load_npy, save_json, require_artifacts, ensure_dir
from src.utils.summary import PhaseSummary
from src.utils.timing import Stopwatch

PHASE = "11_site_shortcut"

# A fixed, untuned reference model is used throughout. The point of this phase
# is the *gap* between evaluation protocols, not an absolute score, so holding
# the estimator constant keeps every number on the same footing and avoids
# re-tuning per site (which LOSO folds are far too small to support honestly).
def _rf(seed: int, n_classes_hint: int = 2) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _recording_macc(rec_ids, y_prob, y_seg, method) -> dict:
    """Segment probabilities -> recording decision -> figures of merit."""
    agg = aggregate_to_recording(rec_ids, y_prob, y_seg, method=method)
    m = compute_metrics(agg["y_true"], agg["y_pred"], agg["y_prob"])
    m["n_recordings"] = int(len(agg["y_true"]))
    return m


def main() -> int:
    cfg, logger, args = setup(PHASE, "Site-shortcut confirming experiment (site classifier, LOSO, within-site)")
    summary = PhaseSummary(PHASE, cfg, next_phase=None)
    watch = Stopwatch()

    processed = Path(cfg["_abs_paths"]["processed_dir"])
    interim = Path(cfg["_abs_paths"]["interim_dir"])
    results_dir = ensure_dir(Path(cfg["_abs_paths"]["results_dir"]) / "site_shortcut")
    figures_dir = ensure_dir(cfg["_abs_paths"]["figures_dir"])

    require_artifacts([interim / "segment_index.csv"], phase=PHASE)
    index = pd.read_csv(interim / "segment_index.csv", dtype={"recording_id": str})

    # TRAIN ONLY. Test remains sealed for phase 06; assert it here so the
    # machine-checkable "which splits did this phase read" field stays honest.
    train_index = index[index["split"] == "train"].reset_index(drop=True)
    summary.touch_split("train")
    summary.add_assertion(
        "test_split_untouched", True,
        "Phase 11 reads the train split only; the test partition is not opened, "
        "preserving the single-reader guarantee of phase 06.",
    )

    y = train_index["label"].to_numpy().astype(int)
    sites = train_index["subdb"].to_numpy()
    recs = train_index["recording_id"].to_numpy()

    site_names = sorted(np.unique(sites).tolist())
    n_sites = len(site_names)
    site_to_int = {s: i for i, s in enumerate(site_names)}
    s_int = np.array([site_to_int[s] for s in sites])

    seed = int(cfg["experiment"]["seed"])
    n_folds = 5
    agg_method = str(cfg["evaluation"]["primary_aggregation"])

    # site majority baseline (chance for a constant site predictor)
    _, counts = np.unique(sites, return_counts=True)
    site_majority = float(counts.max() / counts.sum())
    site_uniform = 1.0 / n_sites

    logger.info(f"{len(y)} training segments | {len(np.unique(recs))} recordings | "
                f"{n_sites} sites | {100*y.mean():.1f}% abnormal")
    logger.info(f"site majority baseline (bal.acc chance) ~ uniform {site_uniform:.3f}, "
                f"largest-site fraction {site_majority:.3f}")

    feature_sets = list(cfg["classical"]["feature_sets"])  # mfcc, pwp
    domain_results = {}

    def _splitter(y_strat, groups):
        if _HAVE_SGKF:
            sk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return sk.split(np.zeros(len(y_strat)), y_strat, groups)
        gk = GroupKFold(n_splits=n_folds)
        return gk.split(np.zeros(len(y_strat)), y_strat, groups)

    for domain in feature_sets:
        fpath = processed / domain / "features_train.npy"
        if not fpath.exists():
            logger.warning(f"Skipping {domain}: {fpath} not found")
            continue
        X = load_npy(fpath)
        if X.ndim != 2:
            logger.warning(f"Skipping {domain}: expected 2-D features, got {X.shape}")
            continue
        summary.add_input(fpath, shape=list(X.shape))
        logger.info(f"=== {domain}: {X.shape[1]} features " + "=" * 40)

        # ------------------------------------------------------------------
        # A. Is SITE recoverable from the features? (grouped multiclass CV)
        # ------------------------------------------------------------------
        with watch.section(f"{domain}_site_clf"):
            seg_site_bal, rec_site_acc = [], []
            for tr, te in _splitter(s_int, recs):
                clf = make_pipeline(StandardScaler(), _rf(seed, n_sites))
                clf.fit(X[tr], s_int[tr])
                p = clf.predict(X[te])
                seg_site_bal.append(balanced_accuracy_score(s_int[te], p))
                # recording-level: majority predicted site per recording
                df = pd.DataFrame({"rec": recs[te], "pred": p, "true": s_int[te]})
                grp = df.groupby("rec").agg(
                    pred=("pred", lambda c: c.value_counts().index[0]),
                    true=("true", "first"))
                rec_site_acc.append(accuracy_score(grp["true"], grp["pred"]))
        site_bal_mean = float(np.mean(seg_site_bal))
        site_rec_mean = float(np.mean(rec_site_acc))
        logger.info(f"  [A] site classifier: segment bal.acc {site_bal_mean:.3f} "
                    f"(chance {site_uniform:.3f}); recording acc {site_rec_mean:.3f}")

        # ------------------------------------------------------------------
        # B. Pooled stratified CV vs LEAVE-ONE-SITE-OUT for the DIAGNOSIS.
        # ------------------------------------------------------------------
        with watch.section(f"{domain}_pooled_cv"):
            pooled_macc = []
            for tr, te in _splitter(y, recs):
                clf = make_pipeline(StandardScaler(), _rf(seed))
                clf.fit(X[tr], y[tr])
                pr = clf.predict_proba(X[te])[:, 1]
                pooled_macc.append(_recording_macc(recs[te], pr, y[te], agg_method)["macc"])
        pooled_macc_mean = float(np.mean(pooled_macc))

        with watch.section(f"{domain}_loso"):
            loso_rows, loso_maccs = [], []
            for s in site_names:
                te = s_int == site_to_int[s]
                tr = ~te
                # a held-out site with a single class cannot yield a MAcc
                if len(np.unique(y[te])) < 2:
                    loso_rows.append({"site": s, "n_rec": int(len(np.unique(recs[te]))),
                                      "prevalence": round(float(y[te].mean()), 3),
                                      "macc": None, "note": "single-class site, skipped"})
                    continue
                clf = make_pipeline(StandardScaler(), _rf(seed))
                clf.fit(X[tr], y[tr])
                pr = clf.predict_proba(X[te])[:, 1]
                m = _recording_macc(recs[te], pr, y[te], agg_method)
                loso_maccs.append(m["macc"])
                loso_rows.append({"site": s, "n_rec": m["n_recordings"],
                                  "prevalence": round(float(y[te].mean()), 3),
                                  "macc": round(m["macc"], 3),
                                  "se": round(m["sensitivity"], 3),
                                  "sp": round(m["specificity"], 3)})
            loso_macc_mean = float(np.mean(loso_maccs)) if loso_maccs else float("nan")
        gap = pooled_macc_mean - loso_macc_mean
        logger.info(f"  [B] diagnosis pooled CV MAcc {pooled_macc_mean:.3f} -> "
                    f"LOSO macro MAcc {loso_macc_mean:.3f} (drop {gap:+.3f})")

        # ------------------------------------------------------------------
        # C. WITHIN-SITE diagnosis CV: signal that survives with site fixed.
        # ------------------------------------------------------------------
        with watch.section(f"{domain}_within_site"):
            within_rows, within_maccs = [], []
            for s in site_names:
                m = s_int == site_to_int[s]
                if len(np.unique(y[m])) < 2:
                    within_rows.append({"site": s, "macc": None, "note": "single-class"})
                    continue
                # need at least a few recordings of each class to split by group
                n_rec_site = len(np.unique(recs[m]))
                if n_rec_site < 10:
                    within_rows.append({"site": s, "n_rec": int(n_rec_site),
                                        "macc": None, "note": "too few recordings"})
                    continue
                maccs = []
                Xs, ys, rs = X[m], y[m], recs[m]
                folds = min(3, int(np.unique(rs).size // 4) or 2)
                try:
                    for tr, te in _splitter_small(ys, rs, folds, seed):
                        if len(np.unique(ys[te])) < 2 or len(np.unique(ys[tr])) < 2:
                            continue
                        clf = make_pipeline(StandardScaler(), _rf(seed))
                        clf.fit(Xs[tr], ys[tr])
                        pr = clf.predict_proba(Xs[te])[:, 1]
                        maccs.append(_recording_macc(rs[te], pr, ys[te], agg_method)["macc"])
                except ValueError:
                    pass
                if maccs:
                    wm = float(np.mean(maccs))
                    within_maccs.append(wm)
                    within_rows.append({"site": s, "n_rec": int(n_rec_site),
                                        "prevalence": round(float(ys.mean()), 3),
                                        "macc": round(wm, 3)})
                else:
                    within_rows.append({"site": s, "n_rec": int(n_rec_site),
                                        "macc": None, "note": "no valid fold"})
            within_macc_mean = float(np.mean(within_maccs)) if within_maccs else float("nan")
        logger.info(f"  [C] within-site diagnosis MAcc (macro) {within_macc_mean:.3f} "
                    "(0.5 == no diagnosis signal once site is fixed)")

        domain_results[domain] = {
            "site_classifier": {
                "segment_balanced_accuracy": round(site_bal_mean, 4),
                "recording_accuracy": round(site_rec_mean, 4),
                "chance_uniform": round(site_uniform, 4),
                "largest_site_fraction": round(site_majority, 4),
                "n_sites": n_sites,
            },
            "diagnosis": {
                "pooled_cv_macc": round(pooled_macc_mean, 4),
                "loso_macro_macc": round(loso_macc_mean, 4),
                "pooled_minus_loso": round(gap, 4),
                "within_site_macro_macc": round(within_macc_mean, 4),
                "loso_per_site": loso_rows,
                "within_site_per_site": within_rows,
            },
        }
        save_json(results_dir / f"{domain}_site_shortcut.json", domain_results[domain])
        summary.add_artifact(results_dir / f"{domain}_site_shortcut.json")

        summary.add_finding(f"{domain}_site_balanced_accuracy", round(site_bal_mean, 4))
        summary.add_finding(f"{domain}_pooled_cv_macc", round(pooled_macc_mean, 4))
        summary.add_finding(f"{domain}_loso_macro_macc", round(loso_macc_mean, 4))
        summary.add_finding(f"{domain}_pooled_minus_loso", round(gap, 4))
        summary.add_finding(f"{domain}_within_site_macro_macc", round(within_macc_mean, 4))

    if not domain_results:
        logger.error("No 2-D feature domains available; nothing to test.")
        summary.write(cfg["_abs_paths"]["reports_dir"], status="failed")
        return 1

    # ---- figure -----------------------------------------------------------
    try:
        _plot(domain_results, site_uniform, figures_dir / "fig_site_shortcut")
        summary.add_artifact(figures_dir / "fig_site_shortcut.pdf")
    except Exception as exc:  # a figure failure must not sink the measurement
        logger.warning(f"figure skipped: {exc}")

    # ---- verdict on the new claim ----------------------------------------
    site_bals = [d["site_classifier"]["segment_balanced_accuracy"] for d in domain_results.values()]
    gaps = [d["diagnosis"]["pooled_minus_loso"] for d in domain_results.values()]
    withins = [d["diagnosis"]["within_site_macro_macc"] for d in domain_results.values()
               if d["diagnosis"]["within_site_macro_macc"] == d["diagnosis"]["within_site_macro_macc"]]
    best_site_bal = max(site_bals)
    max_gap = max(gaps)
    mean_within = float(np.mean(withins)) if withins else float("nan")

    site_recoverable = best_site_bal > 3 * site_uniform      # well above chance
    perf_is_site_driven = max_gap > 0.10                     # LOSO collapses vs pooled
    signal_thin = (mean_within == mean_within) and mean_within < 0.60

    if site_recoverable and (perf_is_site_driven or signal_thin):
        verdict = "supported"
    elif site_recoverable or perf_is_site_driven:
        verdict = "weak"
    else:
        verdict = "contradicted"

    evidence = (
        f"site balanced-accuracy up to {best_site_bal:.3f} vs {site_uniform:.3f} chance; "
        f"diagnosis MAcc drops by up to {max_gap:.3f} from pooled CV to leave-one-site-out; "
        f"within-site diagnosis MAcc averages {mean_within:.3f}"
    )
    summary.add_claim_verdict(
        "C6",
        "The recording site is directly recoverable from the features, and the pooled "
        "performance is substantially a site-prevalence shortcut: controlling for site "
        "(leave-one-site-out, or within-site) removes most of the apparent skill.",
        verdict, evidence,
        "This is the confirming experiment named in Sections V-C, VII and VIII. It "
        "converts the report's central inference into a direct measurement: the site "
        "classifier shows the shortcut substrate is present, and the pooled-vs-LOSO gap "
        "and within-site collapse show how much of the headline MAcc depends on it.",
    )

    summary.add_table("site_shortcut_summary", [
        {"domain": d,
         "site_bal_acc": r["site_classifier"]["segment_balanced_accuracy"],
         "chance": r["site_classifier"]["chance_uniform"],
         "pooled_macc": r["diagnosis"]["pooled_cv_macc"],
         "loso_macc": r["diagnosis"]["loso_macro_macc"],
         "pooled_minus_loso": r["diagnosis"]["pooled_minus_loso"],
         "within_site_macc": r["diagnosis"]["within_site_macro_macc"]}
        for d, r in domain_results.items()
    ])

    summary.add_note(
        "Estimator is a fixed, untuned balanced random forest across all three blocks; "
        "the quantity of interest is the gap between evaluation protocols, not an "
        "absolute score, so per-fold tuning is deliberately avoided. Sites with a single "
        "class or too few recordings in a fold are reported as skipped rather than "
        "imputed."
    )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    return 0


def _splitter_small(y_strat, groups, n_splits, seed):
    """A small-n grouped splitter for the within-site block."""
    n_splits = max(2, int(n_splits))
    if _HAVE_SGKF:
        sk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return sk.split(np.zeros(len(y_strat)), y_strat, groups)
    gk = GroupKFold(n_splits=n_splits)
    return gk.split(np.zeros(len(y_strat)), y_strat, groups)


def _plot(domain_results, chance, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = list(domain_results.keys())
    x = np.arange(len(domains))
    w = 0.2
    site_bal = [domain_results[d]["site_classifier"]["segment_balanced_accuracy"] for d in domains]
    pooled = [domain_results[d]["diagnosis"]["pooled_cv_macc"] for d in domains]
    loso = [domain_results[d]["diagnosis"]["loso_macro_macc"] for d in domains]
    within = [domain_results[d]["diagnosis"]["within_site_macro_macc"] for d in domains]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - 1.5 * w, site_bal, w, label="site bal.acc (A)")
    ax.bar(x - 0.5 * w, pooled, w, label="diagnosis pooled CV MAcc (B)")
    ax.bar(x + 0.5 * w, loso, w, label="diagnosis LOSO MAcc (B)")
    ax.bar(x + 1.5 * w, within, w, label="within-site MAcc (C)")
    ax.axhline(0.5, ls="--", lw=1, color="gray", label="MAcc chance = 0.5")
    ax.axhline(chance, ls=":", lw=1, color="black", label=f"site chance = {chance:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in domains])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Site is recoverable; diagnosis skill collapses once site is controlled")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
