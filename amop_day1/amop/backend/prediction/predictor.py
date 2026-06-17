"""
Prediction inference — loads the trained model bundle and runs single-record inference.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.prediction.features import extract_features, active_feature_names

logger = get_logger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when no trained model file exists yet."""


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    settings = get_settings()
    path = settings.model_path
    try:
        bundle = joblib.load(path)
        logger.info("Prediction model loaded", extra={"path": path, "trained_on": bundle.get("trained_on")})
        return bundle
    except FileNotFoundError:
        raise ModelNotTrainedError(
            f"Model file not found at '{path}'. "
            "Ingest data first via POST /api/v1/ingest to train the model."
        )


def reload_model() -> None:
    """Clear the cached bundle so the next call picks up a freshly trained file."""
    _load_bundle.cache_clear()


def predict(
    machine_id: str,
    issue_description: str,
    technician_notes: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> dict:
    """
    Run severity + ETA inference for a single work order.

    Returns:
        severity            str   "low" | "medium" | "high"
        severity_confidence float 0–1
        eta_hours           float estimated repair hours
        features_used       list  names of non-zero features
    """
    bundle = _load_bundle()  # raises ModelNotTrainedError if not trained

    vec = extract_features(
        issue=issue_description,
        notes=technician_notes,
        machine_id=machine_id,
        created_at=created_at or datetime.utcnow(),
    )
    X = np.array([vec], dtype=float)

    clf = bundle["severity_classifier"]
    reg = bundle["eta_regressor"]
    label_names: dict = bundle["label_names"]

    # Severity
    proba = clf.predict_proba(X)[0]
    severity_idx = int(np.argmax(proba))
    severity = label_names[severity_idx]
    confidence = float(proba[severity_idx])

    # ETA
    eta_raw = float(reg.predict(X)[0])
    eta_hours = max(0.25, round(eta_raw, 1))

    features_used = active_feature_names(vec)

    logger.info(
        "Prediction complete",
        extra={
            "machine_id": machine_id,
            "severity": severity,
            "confidence": round(confidence, 3),
            "eta_hours": eta_hours,
        },
    )

    return {
        "severity": severity,
        "severity_confidence": round(confidence, 4),
        "eta_hours": eta_hours,
        "features_used": features_used,
    }
