"""SVM and Random Forest: model construction, grids and grouped cross-validation.

Two models with deliberately different inductive biases, so that agreement
between them is informative:

* **RBF-SVM** - a global, distance-based, maximum-margin model. Sensitive to
  feature scaling, produces a smooth decision boundary, and treats all 234
  dimensions jointly.
* **Random Forest** - a local, axis-aligned, ensemble model. Scale-invariant,
  handles non-linear interactions by partitioning, and gives us an exact
  (TreeSHAP) explanation for free.

PROFESSOR Q: "Why these two and not, say, XGBoost?"
A: The point of the classical branch is to be a *credible baseline* against the
   CNN, not to win a leaderboard. SVM and RF are the two models the CinC 2016
   entries actually used, they span two very different hypothesis classes, and
   crucially one of them (RF) admits an exact SHAP computation, which anchors
   the explainability comparison. Adding a third gradient-boosted model would
   add compute and a third set of hyperparameters without changing any
   conclusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_svm(cfg: Dict, seed: int = 42):
    """RBF-kernel SVM.

    ``class_weight='balanced'`` reweights the hinge loss by inverse class
    frequency. On this dataset (roughly 3:1 normal:abnormal) it is the
    difference between a model that detects pathology and one that has learned
    to say "normal".
    """
    from sklearn.svm import SVC

    svm_cfg = cfg["classical"]["svm"]
    return SVC(
        kernel=str(svm_cfg.get("kernel", "rbf")),
        class_weight=svm_cfg.get("class_weight", "balanced"),
        probability=bool(svm_cfg.get("probability", True)),
        random_state=seed,
        cache_size=500,
    )


def build_random_forest(cfg: Dict, seed: int = 42):
    """Random Forest.

    ``class_weight='balanced_subsample'`` recomputes the weights for each
    bootstrap sample rather than once globally, which is the correct variant
    when bagging.
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


def build_pipeline(model_name: str, cfg: Dict, seed: int = 42):
    """Scaler + estimator as a single Pipeline.

    PROFESSOR Q: "Why wrap the scaler inside the Pipeline for cross-validation
                  when you already scaled the features in phase 02?"
    A: Because inside cross-validation the scaler must be refitted on each
       training fold. If we relied only on the phase-02 scaler, every CV fold
       would share normalisation statistics computed over all training data,
       including the fold being validated. The effect is small but it is a real
       leak, and sklearn's Pipeline handles it correctly by construction. The
       phase-02 scaler still exists for the final model applied to test data.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = MODEL_BUILDERS[model_name](cfg, seed=seed)
    steps = []
    # Trees are invariant to monotone rescaling; skipping the scaler for RF
    # saves time and makes feature values directly readable in SHAP plots.
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
    A: Because our unit of observation is a 3-second segment, but our unit of
       independence is a recording. A recording contributes 5-20 segments that
       share a patient, a stethoscope, a noise floor and a murmur. If those
       segments are scattered across folds, the model validates on segments
       whose siblings it trained on, and CV scores become optimistic by a wide
       margin. Grouping by recording_id makes each fold's validation set truly
       unseen. The 'Stratified' half keeps class balance stable across folds,
       which matters at 3:1 imbalance.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n_folds = int(cfg["classical"]["cv_folds"])
    strategy = str(cfg["classical"].get("cv_strategy", "stratified_group_kfold"))

    if strategy == "stratified_group_kfold":
        return StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    if strategy == "stratified_kfold":
        # Available for the leakage demonstration only - NOT for reported results.
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown cv_strategy: {strategy}")


def run_grid_search(model_name: str, X: np.ndarray, y: np.ndarray,
                    groups: np.ndarray, cfg: Dict, seed: int = 42,
                    verbose: int = 1) -> Tuple[Any, Dict[str, Any]]:
    """Grid search with grouped, stratified CV. Returns (fitted_search, report).

    Scoring is ``balanced_accuracy``, which is exactly MAcc = (Se + Sp) / 2,
    the official PhysioNet/CinC 2016 metric. Selecting on plain accuracy here
    would pick models that maximise the majority class.
    """
    from sklearn.model_selection import GridSearchCV

    pipeline = build_pipeline(model_name, cfg, seed=seed)
    grid = get_param_grid(model_name, cfg)
    cv = make_cv(cfg, seed=seed)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=str(cfg["classical"].get("scoring", "balanced_accuracy")),
        cv=cv,
        n_jobs=int(cfg["classical"].get("n_jobs", -1)),
        refit=True,
        verbose=verbose,
        return_train_score=True,
        error_score="raise",     # Fail loudly rather than silently scoring NaN
    )
    search.fit(X, y, groups=groups)

    results = search.cv_results_
    best = int(search.best_index_)
    report = {
        "model": model_name,
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "best_cv_score": float(search.best_score_),
        "best_cv_std": float(results["std_test_score"][best]),
        "best_train_score": float(results["mean_train_score"][best]),
        # Train-validation gap is the cheapest overfitting diagnostic there is.
        "overfit_gap": float(results["mean_train_score"][best] - search.best_score_),
        "n_candidates": int(len(results["params"])),
        "scoring": str(cfg["classical"].get("scoring", "balanced_accuracy")),
        "cv_strategy": str(cfg["classical"].get("cv_strategy")),
        "n_folds": int(cfg["classical"]["cv_folds"]),
        "all_candidates": [
            {"params": {k.replace("model__", ""): v for k, v in params.items()},
             "mean_test": float(mean), "std_test": float(std)}
            for params, mean, std in zip(
                results["params"], results["mean_test_score"], results["std_test_score"]
            )
        ],
    }
    return search, report


def check_overfitting(report: Dict[str, Any], threshold: float = 0.10) -> Optional[str]:
    """Return a warning string if the train-validation gap is suspicious."""
    gap = report.get("overfit_gap", 0.0)
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
    which is why it is slow. If probability estimates are unavailable we
    squash the decision function through a logistic so that downstream
    probability averaging still works - and we note in the paper that those
    scores are calibrated only monotonically.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(np.float64)
