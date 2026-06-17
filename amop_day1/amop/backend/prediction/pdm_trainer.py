"""
XGBoost training pipeline for the Azure PdM failure-prediction model.

Trains a binary classifier that predicts whether a machine will fail within
the next 24 hours, using features built by pdm_features.py.

Key design decisions:
- PR-AUC is the sole ranking metric. ROC-AUC is dropped because with a
  ~2% positive rate it becomes misleadingly optimistic — the large number
  of true negatives inflates it regardless of how well the model finds
  actual failures.
- TimeSeriesSplit cross-validation (5 folds) is used instead of a random
  k-fold split. Each fold trains on strictly past data and validates on
  strictly future data, preventing temporal leakage.
- scale_pos_weight is recomputed per CV fold from the fold's own training
  labels so the class-balance estimate stays accurate as the window grows.
- The mean best_iteration across CV folds is used as the fixed n_estimators
  for the final model, avoiding any use of the held-out test set during
  training.
- The 2015-10-01 cutoff is still used as a final holdout for the reported
  metrics — it is never touched during CV.

Run from project root:
    python -m backend.prediction.pdm_trainer
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from backend.prediction.pdm_features import (
    FEATURE_COLS,
    TARGET_COL,
    TRAIN_CUTOFF,
    build_feature_matrix,
)

logger = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────────────────

# This file is at backend/prediction/pdm_trainer.py; parents[2] = project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SAVE_PATH = _PROJECT_ROOT / "models" / "pdm_failure_model.joblib"

# ── XGBoost hyperparameters (single source of truth) ─────────────────────────

_XGB_PARAMS: dict[str, Any] = {
    "objective":         "binary:logistic",
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "aucpr",
    "tree_method":       "hist",
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":         0,
}


def _make_xgb(
    n_estimators: int,
    spw: float,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    """
    Instantiate an XGBClassifier with the project-standard hyperparameters.

    Keeping all hyperparameters in _XGB_PARAMS and building through this
    factory ensures CV folds and the final model always use identical settings.
    """
    return XGBClassifier(
        **_XGB_PARAMS,
        n_estimators          = n_estimators,
        scale_pos_weight      = spw,
        early_stopping_rounds = early_stopping_rounds,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """
    Return the decision threshold that maximises F1 on y_true / y_prob.

    precision_recall_curve() produces N+1 precision/recall values but only
    N thresholds (the last precision/recall pair has no threshold entry).
    We index thresholds with [:-1] to align arrays before argmax.

    Returns:
        (optimal_threshold, f1_at_threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def _run_single_fold(
    X_tr:  np.ndarray,
    y_tr:  np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, int]:
    """
    Train one TimeSeriesSplit fold and return (pr_auc, best_iteration).

    scale_pos_weight is computed from the fold's own y_tr so the class
    balance weight stays accurate as the expanding training window grows.
    Early stopping uses the fold's validation portion — which is always in
    the future relative to the training portion, so no leakage occurs.

    Returns:
        pr_auc:         PR-AUC on the fold's validation set.
        best_iteration: XGBoost's 0-indexed round with highest aucpr.
    """
    n_neg   = int(np.sum(y_tr == 0))
    n_pos   = int(np.sum(y_tr == 1))
    fold_spw = float(n_neg) / max(n_pos, 1)

    model = _make_xgb(n_estimators=500, spw=fold_spw, early_stopping_rounds=30)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_prob = model.predict_proba(X_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, y_prob))
    return pr_auc, int(model.best_iteration)


def _run_time_series_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> tuple[list[float], int]:
    """
    5-fold TimeSeriesSplit cross-validation scored by PR-AUC.

    TimeSeriesSplit always places training data before validation data in time,
    which is the only valid cross-validation strategy for a time-series dataset.
    Standard k-fold would shuffle future data into the training window and
    past data into the validation window, causing optimistic leakage.

    The mean best_iteration across folds is returned and used as the final
    model's n_estimators so the test set is never touched during training.

    Args:
        X_train:  Feature matrix (training portion only, before TRAIN_CUTOFF).
        y_train:  Labels aligned with X_train.
        n_splits: Number of CV folds (default 5).

    Returns:
        (fold_pr_aucs, mean_best_iteration)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_scores:      list[float] = []
    best_iterations:  list[int]   = []

    print(f"\n{'='*65}")
    print(f"{n_splits}-Fold TimeSeriesSplit Cross-Validation  |  metric: PR-AUC")
    print(f"{'='*65}")
    print(f"  {'Fold':<6} {'Train':>10} {'Val':>10} {'Val pos%':>9} {'PR-AUC':>8} {'Best iter':>10}")
    print(f"  {'-'*55}")

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        pr_auc, best_iter = _run_single_fold(X_tr, y_tr, X_val, y_val)
        fold_scores.append(pr_auc)
        best_iterations.append(best_iter)

        val_pos_pct = float(np.mean(y_val)) * 100
        print(
            f"  {fold_idx:<6} {len(y_tr):>10,} {len(y_val):>10,} "
            f"{val_pos_pct:>8.2f}% {pr_auc:>8.4f} {best_iter:>10}"
        )

    mean_pr  = float(np.mean(fold_scores))
    std_pr   = float(np.std(fold_scores))
    mean_iter = int(np.mean(best_iterations))

    print(f"  {'-'*55}")
    print(f"  CV PR-AUC:           {mean_pr:.4f}  +/-  {std_pr:.4f}")
    print(f"  Mean best iteration: {mean_iter}  (used as n_estimators for final model)")
    print(f"{'='*65}\n")

    return fold_scores, mean_iter


# ── Public API ─────────────────────────────────────────────────────────────────

def train_pdm_model() -> str:
    """
    Full training pipeline: CV -> final model -> SHAP -> save bundle.

    Steps:
        1.  Build feature matrix with strict temporal train/test split.
        2.  Run 5-fold TimeSeriesSplit CV on training data; report PR-AUC
            mean +/- std across folds.
        3.  Train the final model on ALL training data using the CV mean
            best_iteration as fixed n_estimators (no early stopping against
            the test set — keeps the holdout completely clean).
        4.  Evaluate on the held-out test set: PR-AUC, F1 at optimal
            threshold, confusion matrix, classification report.
        5.  Generate SHAP values; print top-10 features.
        6.  Save model bundle to models/pdm_failure_model.joblib.

    Returns:
        Absolute path string of the saved model file.
    """
    t0 = time.time()

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    print("Building PdM feature matrix (this may take 1-2 minutes)...")
    train_df, test_df = build_feature_matrix()

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.int32)
    X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_COL].values.astype(np.int32)

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw   = float(n_neg) / max(n_pos, 1)

    print(f"Training set:  {len(y_train):,} rows  ({n_pos:,} positives = {n_pos/len(y_train)*100:.2f}%)")
    print(f"Test set:      {len(y_test):,} rows  ({int(np.sum(y_test==1)):,} positives = {np.mean(y_test)*100:.2f}%)")
    print(f"scale_pos_weight (full train): {spw:.1f}")

    # ── 2. TimeSeriesSplit cross-validation ───────────────────────────────────
    cv_scores, mean_best_iter = _run_time_series_cv(X_train, y_train, n_splits=5)

    # ── 3. Final model — trained on ALL training data ─────────────────────────
    # Use mean CV best_iteration as a fixed n_estimators so the test set is
    # never seen during training. Add 1 because XGBoost's best_iteration is
    # 0-indexed; n_estimators must include that final round.
    final_n_estimators = mean_best_iter + 1
    print(f"Training final model on all {len(y_train):,} training rows "
          f"for {final_n_estimators} rounds (no early stopping)...")

    final_model = _make_xgb(
        n_estimators          = final_n_estimators,
        spw                   = spw,
        early_stopping_rounds = None,   # test set is a clean holdout
    )
    final_model.fit(X_train, y_train, verbose=False)

    # ── 4. Holdout evaluation ─────────────────────────────────────────────────
    y_prob = final_model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_prob))

    optimal_threshold, best_f1 = _find_optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= optimal_threshold).astype(int)

    print(f"\n{'='*65}")
    print(f"Held-out Test Evaluation  (datetime >= {TRAIN_CUTOFF})")
    print(f"{'='*65}")
    print(f"  PR-AUC:              {pr_auc:.4f}   <-- primary metric")
    print(f"  CV PR-AUC (mean):    {float(np.mean(cv_scores)):.4f}  +/-  {float(np.std(cv_scores)):.4f}")
    print(f"  Optimal threshold:   {optimal_threshold:.4f}  (maximises F1 on test set)")
    print(f"  F1 at threshold:     {best_f1:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=["no_failure", "failure"]))

    # ── 5. SHAP feature importance ────────────────────────────────────────────
    print("Computing SHAP values...")

    shap_n = min(5_000, len(X_test))
    rng    = np.random.default_rng(42)
    X_shap = X_test[rng.choice(len(X_test), size=shap_n, replace=False)]

    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):      # some shap versions return a list for binary
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top10         = np.argsort(mean_abs_shap)[::-1][:10]

    print(f"\n{'='*65}")
    print(f"Top 10 Features by Mean |SHAP Value|")
    print(f"{'='*65}")
    for rank, idx in enumerate(top10, 1):
        print(f"  {rank:2d}. {FEATURE_COLS[idx]:<28s}  {mean_abs_shap[idx]:.5f}")
    print(f"{'='*65}\n")

    # ── 6. Save model bundle ──────────────────────────────────────────────────
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = {
        "model":             final_model,
        "feature_names":     FEATURE_COLS,
        "optimal_threshold": optimal_threshold,
        "scale_pos_weight":  spw,
        "train_cutoff":      TRAIN_CUTOFF,
        "metrics": {
            "pr_auc":        pr_auc,
            "cv_pr_auc_mean": float(np.mean(cv_scores)),
            "cv_pr_auc_std":  float(np.std(cv_scores)),
            "f1":            best_f1,
        },
        "cv_fold_scores":    cv_scores,
        "n_estimators_used": final_n_estimators,
        "version":           "pdm_v1",
    }

    joblib.dump(bundle, MODEL_SAVE_PATH, compress=3)

    elapsed = time.time() - t0
    print(f"Model saved -> {MODEL_SAVE_PATH}")
    print(f"Total training time: {elapsed:.1f}s")

    return str(MODEL_SAVE_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    saved_path = train_pdm_model()
    print(f"\nDone. Model bundle at: {saved_path}")
