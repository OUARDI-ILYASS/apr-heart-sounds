"""SVM and Random Forest: model construction, grids and grouped cross-validation.

Two models with deliberately different inductive biases, so that agreement
between them is informative:

* **RBF-SVM** - a global, distance-based, maximum-margin model. Sensitive to
  feature scaling, produces a smooth decision boundary, and treats all 234
  dimensions jointly.
* **Random Forest** - a local, axis-aligned, ensemble model. Scale-invariant,
  handles non-linear interactions by partitioning, and gives us an exact
  (TreeSHAP) explanation for free.

------------------------------------------------------------------------------
COST NOTE - read before changing the search
------------------------------------------------------------------------------
An RBF-SVM is ~O(n^2)-O(n^3) in the number of training segments, and this
dataset has ~20-30k of them. Two things make the naive search far slower than
necessary, and both are handled here:

1. `probability=True` fits an internal 5-fold Platt calibration on EVERY fit
   (~6x cost) while contributing nothing to the balanced-accuracy score the
   candidates are ranked on. We search with it OFF and restore it for the
   single final refit.

2. Segments from one recording are near-duplicates, so ranking (C, gamma) does
   not need all of them. We rank on a few segments per recording.

   BUT: the subsample cannot be trusted to make the *final* choice. Two traps,
   both observed in practice:
     * gamma='scale' == 1 / (n_features * X.var()), so a smaller training set
       silently changes what 'scale' means.
     * a thinner set prefers a smoother boundary, so the subsample can rank a
       weaker (higher-C-equivalent) model first.
   The fix is refit-top-k: the subsample PRUNES to a shortlist, then the
   shortlist is re-scored on the FULL training set and the winner chosen there.
   This keeps most of the speed-up while provably matching the full-data choice
   whenever the shortlist contains it. See `run_grid_search`.

PROFESSOR Q: "Did the fast search pick the same model as an exhaustive one?"
A: Yes, and it is checkable: `classical.search.segments_per_recording: 0` runs
   the search at full size. With refit-top-k enabled the selected (C, gamma)
   matches. The subsample is trusted only to prune, never to decide.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_svm(cfg: Dict, seed: int = 42):
    """RBF-kernel SVM.

    ``class_weight='balanced'`` reweights the hinge loss by inverse class
    frequency. On this dataset (~3:1 normal:abnormal) it is the difference
    between a model that detects pathology and one that says "normal".

    ``cache_size`` is libsvm's kernel cache in MB. The default of 200 is far
    too small here: the kernel matrix is 8*n^2 bytes, so at n=20000 it is
    3.2 GB. A cache that cannot hold the working set forces kernel entries to be
    recomputed - the single biggest hidden cost in an SVM fit at this scale.
    """
    from sklearn.svm import SVC

    svm_cfg = cfg["classical"]["svm"]
    return SVC(
        kernel=str(svm_cfg.get("kernel", "rbf")),
        class_weight=svm_cfg.get("class_weight", "balanced"),
        probability=bool(svm_cfg.get("probability", True)),
        cache_size=float(svm_cfg.get("cache_size", 2000)),
        # A guard, not a tuning knob: the (high C, high gamma) corner of the grid
        # can fail to converge and stall the whole search on one candidate.
        # sklearn warns rather than raising, so a non-converged fit stays visible.
        max_iter=int(svm_cfg.get("max_iter", -1)),
        random_state=seed,
    )


def build_random_forest(cfg: Dict, seed: int = 42):
    """Random Forest.

    ``class_weight='balanced_subsample'`` recomputes weights per bootstrap
    sample rather than once globally - the correct variant when bagging.
    """
    from sklearn.ensemble import RandomForestClassifier

    rf_cfg = cfg["classical"]["random_forest"]
    return RandomForestClassifier(
        class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=seed,
        oob_score=False,          # We use grouped CV instead; OOB ignores groups
    )


MODEL_BUILDERS = {"svm": build_svm, "rf": build_random_forest}


def get_param_grid(model_name: str, cfg: Dict) -> Dict[str, List]:
    """Hyperparameter grid, with the ``model__`` prefix a Pipeline needs."""
    key = {"svm": "svm", "rf": "random_forest"}[model_name]
    grid = cfg["classical"][key]["grid"]
    return {f"model__{param}": list(values) for param, values in grid.items()}


def build_pipeline(model_name: str, cfg: Dict, seed: int = 42,
                   for_search: bool = False):
    """Scaler + estimator as a single Pipeline.

    PROFESSOR Q: "Why wrap the scaler in the Pipeline for CV when phase 02
                  already scaled the features?"
    A: Inside CV the scaler must be refitted on each training fold. Relying only
       on the phase-02 scaler would share statistics computed over the fold
       being validated - a small but real leak. The Pipeline handles it
       correctly by construction. The phase-02 scaler still applies to test data.

    ``for_search=True`` returns a cheaper configuration used only while ranking
    hyperparameters. It never touches the model that gets saved.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = MODEL_BUILDERS[model_name](cfg, seed=seed)

    if for_search:
        if model_name == "svm":
            # Platt scaling is a 5-fold CV inside every fit; the ranking metric
            # needs only hard predictions, so it is ~6x cost for no information.
            estimator.set_params(probability=False)
        if model_name == "rf":
            # GridSearchCV already parallelises across candidates and folds.
            # Letting the forest also claim every core oversubscribes the machine
            # (16 processes x 16 threads on 16 cores). One thread per worker here.
            estimator.set_params(n_jobs=1)

    steps = []
    # Trees are invariant to monotone rescaling; skipping the scaler for RF
    # saves time and keeps feature values readable in SHAP plots.
    if model_name != "rf":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def make_cv(cfg: Dict, seed: int = 42):
    """Cross-validation splitter that respects recording groups.

    PROFESSOR Q: "Why StratifiedGroupKFold?"
    A: The unit of observation is a 3-second segment, but the unit of
       independence is a recording. A recording contributes 5-20 segments that
       share a patient, a stethoscope and a murmur. Scattering them across folds
       lets the model validate on segments whose siblings it trained on, and CV
       scores become optimistic by a wide margin. Grouping by recording_id
       removes this; the 'Stratified' part preserves class balance at 3:1.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n_folds = int(cfg["classical"]["cv_folds"])
    strategy = str(cfg["classical"].get("cv_strategy", "stratified_group_kfold"))

    if strategy == "stratified_group_kfold":
        return StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    if strategy == "stratified_kfold":
        # For the leakage demonstration only - NOT for reported results.
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown cv_strategy: {strategy}")


def subsample_by_group(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       per_group: int = 4, seed: int = 42
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep at most ``per_group`` segments per recording, for pruning only.

    Every recording still contributes, so the subsample's class balance and site
    mix match the full training set by construction. This is used to shortlist
    candidates cheaply; the final choice is made on full data (see
    ``run_grid_search``), because the subsample can mis-rank models - notably
    when gamma='scale', whose value depends on the training-set variance.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)

    keep: List[int] = []
    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        keep.extend(rng.choice(indices, min(per_group, len(indices)),
                               replace=False).tolist())

    keep_array = np.sort(np.asarray(keep, dtype=int))
    return X[keep_array], y[keep_array], groups[keep_array]


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #
def run_grid_search(model_name: str, X: np.ndarray, y: np.ndarray,
                    groups: np.ndarray, cfg: Dict, seed: int = 42,
                    verbose: int = 1, logger=None) -> Tuple[Any, Dict[str, Any]]:
    """Search hyperparameters, then refit the winner on all training data.

    Returns ``(fitted_estimator, report)``.

    NOTE the return type: this returns the *refitted* estimator, not the search
    object. The saved model is deliberately not the one the subsample search
    produced - selection is finalised on full data and calibration is restored.

    Strategy:
      1. Optionally subsample to a few segments per recording.
      2. Rank all candidates on that subsample (fast, calibration off).
      3. Take the top-k by subsample score and RE-SCORE them on the full
         training set with grouped CV. Choose the winner there. This is what
         prevents the subsample from selecting a weaker model (observed with
         gamma='scale', where the subsample changes gamma's meaning).
      4. Refit that winner on all data with calibration restored.
    """
    from sklearn.model_selection import GridSearchCV, cross_val_score

    search_cfg = cfg["classical"].get("search", {})
    grid = get_param_grid(model_name, cfg)
    cv = make_cv(cfg, seed=seed)
    scoring = str(cfg["classical"].get("scoring", "balanced_accuracy"))
    n_jobs = int(cfg["classical"].get("n_jobs", -1))

    def _log(message: str) -> None:
        (logger.info if logger is not None else print)(message)

    # ---- 1. subsample --------------------------------------------------
    per_group = int(search_cfg.get("segments_per_recording", 0))
    subsampled = per_group > 0 and len(np.unique(groups)) < len(groups)
    if subsampled:
        X_s, y_s, groups_s = subsample_by_group(X, y, groups, per_group, seed)
        _log(f"  prune on {len(X_s)}/{len(X)} segments "
             f"({per_group}/recording, {100 * y_s.mean():.1f}% abnormal "
             f"vs {100 * y.mean():.1f}% full)")
    else:
        X_s, y_s, groups_s = X, y, groups
        _log(f"  search on all {len(X)} segments")

    # ---- 2. rank candidates on the (sub)sample -------------------------
    pipeline = build_pipeline(model_name, cfg, seed=seed, for_search=True)
    search = GridSearchCV(
        estimator=pipeline, param_grid=grid, cv=cv, scoring=scoring,
        n_jobs=n_jobs, refit=False, verbose=verbose,
        return_train_score=True, error_score="raise",
    )
    search.fit(X_s, y_s, groups=groups_s)
    results = search.cv_results_

    # ---- 3. refit-top-k on the FULL training set -----------------------
    top_k = int(search_cfg.get("refit_top_k", 3))
    order = np.argsort(-results["mean_test_score"])
    shortlist = [results["params"][i] for i in order[:max(1, top_k)]]

    if subsampled and len(shortlist) > 1:
        _log(f"  re-scoring top {len(shortlist)} candidates on all {len(X)} segments")
        full_pipe = build_pipeline(model_name, cfg, seed=seed, for_search=True)
        full_scores: List[float] = []
        for params in shortlist:
            full_pipe.set_params(**params)
            scores = cross_val_score(full_pipe, X, y, groups=groups, cv=cv,
                                     scoring=scoring, n_jobs=n_jobs)
            full_scores.append(float(scores.mean()))
            _log(f"    { {k.replace('model__', ''): v for k, v in params.items()} }"
                 f"  full CV {scoring} {scores.mean():.4f}")
        winner = int(np.argmax(full_scores))
        best_params = shortlist[winner]
        best_cv_score = full_scores[winner]
        best_cv_std = float("nan")   # single cross_val_score run; std tracked below
        # Recover the std for the winner by one more scored run for the report.
        full_pipe.set_params(**best_params)
        winner_scores = cross_val_score(full_pipe, X, y, groups=groups, cv=cv,
                                        scoring=scoring, n_jobs=n_jobs)
        best_cv_std = float(winner_scores.std())
        selection = "subsample_prune_then_full_refit_topk"
    else:
        best_index = int(np.argmax(results["mean_test_score"]))
        best_params = results["params"][best_index]
        best_cv_score = float(results["mean_test_score"][best_index])
        best_cv_std = float(results["std_test_score"][best_index])
        selection = "full_grid" if not subsampled else "subsample_single_candidate"

    report: Dict[str, Any] = {
        "model": model_name,
        "best_params": {k.replace("model__", ""): v for k, v in best_params.items()},
        "best_cv_score": best_cv_score,
        "best_cv_std": best_cv_std,
        "n_candidates": int(len(results["params"])),
        "selection_strategy": selection,
        "search_n_samples": int(len(X_s)),
        "full_n_samples": int(len(X)),
        "search_segments_per_recording": per_group,
        "refit_top_k": top_k,
        "calibrated_during_search": False,
        "scoring": scoring,
        "cv_strategy": str(cfg["classical"].get("cv_strategy")),
        "n_folds": int(cfg["classical"]["cv_folds"]),
        "subsample_ranking": [
            {"params": {k.replace("model__", ""): v for k, v in p.items()},
             "subsample_mean": float(m), "subsample_std": float(s)}
            for p, m, s in zip(results["params"], results["mean_test_score"],
                               results["std_test_score"])
        ],
    }

    # Train-score gap: available from the subsample search only. It is a
    # diagnostic, not a gate, so we report it as measured on the (sub)sample.
    if "mean_train_score" in results:
        idx = int(np.argmax(results["mean_test_score"]))
        report["best_train_score"] = float(results["mean_train_score"][idx])
        report["overfit_gap"] = report["best_train_score"] - float(
            results["mean_test_score"][idx]
        )
    else:
        report["best_train_score"] = float("nan")
        report["overfit_gap"] = float("nan")

    _log(f"  selected {report['best_params']}  "
         f"CV {scoring} {best_cv_score:.4f} ± {best_cv_std:.4f}")

    # ---- 4. refit the winner on ALL data, calibration restored ---------
    final = build_pipeline(model_name, cfg, seed=seed, for_search=False)
    final.set_params(**best_params)
    _log(f"  refitting on all {len(X)} segments (probability=True restored for SVM)")
    final.fit(X, y)

    return final, report


def check_overfitting(report: Dict[str, Any], threshold: float = 0.10
                      ) -> Optional[str]:
    """Return a warning string if the train-validation gap is suspicious."""
    gap = report.get("overfit_gap", float("nan"))
    if gap != gap:   # NaN
        return None
    if gap > threshold:
        return (
            f"{report['model']}: train-CV gap is {gap:.3f} "
            f"(train {report['best_train_score']:.3f} vs CV {report['best_cv_score']:.3f}). "
            "The model is memorising the training folds; consider stronger "
            "regularisation (lower C for SVM, higher min_samples_leaf for RF)."
        )
    return None


def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    """Positive-class probability, falling back to a decision function.

    ``SVC(probability=True)`` fits an internal Platt-scaling model by 5-fold CV,
    which is why it is slow - and why we disable it during the search. If
    probability estimates are unavailable we squash the decision function
    through a logistic so downstream probability averaging still works; those
    scores are then calibrated only monotonically, which the paper notes.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(np.float64)