"""
Online inference for the Azure PdM failure-prediction model.

Given a numeric machineID (1-100), this module:
  1. Builds the latest feature vector for that machine.
  2. Runs model.predict_proba() to get the failure probability.
  3. Computes SHAP values for the single prediction to explain the result.
  4. Returns a structured dict with risk tier and top-5 driving features.

The model bundle and SHAP explainer are loaded once and cached via
lru_cache so repeated inference calls are fast.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap

from backend.prediction.pdm_features import FEATURE_COLS, get_latest_features

logger = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────────────────

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_MODEL_PATH    = _PROJECT_ROOT / "models" / "pdm_failure_model.joblib"

# ── Risk tier thresholds ───────────────────────────────────────────────────────

_RISK_THRESHOLDS = {
    "Low":      "<0.25",
    "Medium":   "0.25-0.50",
    "High":     "0.50-0.75",
    "Critical": ">=0.75",
}


def _prob_to_risk_tier(prob: float) -> str:
    """Map a failure probability to a human-readable risk tier."""
    if prob < 0.25:
        return "Low"
    if prob < 0.50:
        return "Medium"
    if prob < 0.75:
        return "High"
    return "Critical"


# ── Cached model + explainer loading ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_pdm_bundle() -> dict:
    """
    Load models/pdm_failure_model.joblib from disk once and cache it.

    Returns the bundle dict saved by pdm_trainer.train_pdm_model().
    Logs a warning and returns None if the file does not yet exist (model
    has not been trained). Callers should handle None gracefully.
    """
    if not _MODEL_PATH.exists():
        logger.warning(
            "PdM model file not found — train first with: "
            "python -m backend.prediction.pdm_trainer",
            extra={"expected_path": str(_MODEL_PATH)},
        )
        return None  # type: ignore[return-value]

    bundle = joblib.load(_MODEL_PATH)
    logger.info(
        "PdM model loaded",
        extra={
            "path":    str(_MODEL_PATH),
            "version": bundle.get("version"),
            "pr_auc":  bundle.get("metrics", {}).get("pr_auc"),
        },
    )
    return bundle


@lru_cache(maxsize=1)
def _load_explainer() -> Optional[shap.TreeExplainer]:
    """
    Build and cache a SHAP TreeExplainer for the trained PdM model.

    Kept separate from _load_pdm_bundle so we can clear it independently
    after retraining without re-loading the raw bundle unnecessarily.
    """
    bundle = _load_pdm_bundle()
    if bundle is None:
        return None
    return shap.TreeExplainer(bundle["model"])


def reload_pdm_model() -> None:
    """
    Clear the lru_cache for both the model bundle and the SHAP explainer.

    Call this after retraining (e.g. from pipelines/train_pdm_model.py) so
    the next inference request picks up the freshly saved model file.
    """
    _load_pdm_bundle.cache_clear()
    _load_explainer.cache_clear()
    logger.info("PdM model cache cleared — next call will reload from disk")


# ── Public inference API ───────────────────────────────────────────────────────

def predict_failure(machine_id: int) -> Optional[dict]:
    """
    Predict failure probability for a single machine and explain the result.

    Args:
        machine_id: Integer machine identifier (1-100 in the PdM dataset).

    Returns:
        A dict with:
            machine_id          (int)
            failure_probability (float, 0.00-1.00, 2 decimal places)
            risk_tier           (str: "Low" | "Medium" | "High" | "Critical")
            risk_thresholds     (dict: tier → range string)
            top_features        (list of 5 dicts: name, shap_value, direction)
            horizon             ("24h")
            model_version       (str from bundle)
        Returns None if:
            - The model file does not exist yet.
            - machine_id is not found in the telemetry dataset.
    """
    bundle = _load_pdm_bundle()
    if bundle is None:
        logger.warning(
            "Cannot predict — PdM model not trained yet",
            extra={"machine_id": machine_id},
        )
        return None

    # ── Build feature vector for this machine ──────────────────────────────────
    try:
        feature_dict = get_latest_features(machine_id)
    except FileNotFoundError as exc:
        logger.error(
            "PdM telemetry data missing — cannot compute features",
            extra={"machine_id": machine_id, "error": str(exc)},
        )
        raise
    if feature_dict is None:
        logger.warning(
            "Cannot predict — machine not found in dataset",
            extra={"machine_id": machine_id},
        )
        return None

    feature_names: list[str] = bundle["feature_names"]
    X = np.array(
        [[feature_dict.get(col, 0.0) for col in feature_names]],
        dtype=np.float32,
    )

    # ── Failure probability ────────────────────────────────────────────────────
    model = bundle["model"]
    prob  = float(model.predict_proba(X)[0, 1])
    prob  = round(prob, 4)

    # ── SHAP explanation ───────────────────────────────────────────────────────
    explainer = _load_explainer()
    top_features: list[dict] = []

    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(X)

            # binary:logistic returns 2-D [1, n_features]; list form → take class-1 slice.
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            shap_row = shap_vals[0]  # shape (n_features,)

            # Rank by absolute SHAP value, keep top 5.
            top_indices = np.argsort(np.abs(shap_row))[::-1][:5]

            for idx in top_indices:
                sv = float(shap_row[idx])
                top_features.append({
                    "name":       feature_names[idx],
                    "shap_value": round(sv, 5),
                    "direction":  "increases_risk" if sv > 0 else "decreases_risk",
                })
        except Exception as exc:
            logger.warning(
                "SHAP computation failed — returning prediction without explanation",
                extra={"machine_id": machine_id, "error": str(exc)},
            )

    risk_tier = _prob_to_risk_tier(prob)

    logger.info(
        "PdM failure prediction complete",
        extra={
            "machine_id":          machine_id,
            "failure_probability": prob,
            "risk_tier":           risk_tier,
        },
    )

    return {
        "machine_id":          machine_id,
        "failure_probability": prob,
        "risk_tier":           risk_tier,
        "risk_thresholds":     _RISK_THRESHOLDS,
        "top_features":        top_features,
        "horizon":             "24h",
        "model_version":       bundle.get("version", "pdm_v1"),
    }
