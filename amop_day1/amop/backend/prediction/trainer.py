"""
Train XGBoost severity classifier and ETA regressor on work order data.

Generates synthetic labels from issue text keywords (no manual annotation needed).
Run automatically after each ingest, or manually:
    python -m backend.prediction.trainer
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.prediction.features import (
    FEATURE_NAMES,
    assign_eta,
    assign_severity,
    extract_features,
)

logger = get_logger(__name__)

_LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
_LABEL_NAMES = {v: k for k, v in _LABEL_MAP.items()}


def _load_rows_from_sqlite() -> list[dict]:
    """Fetch all work orders from SQLite for training."""
    from backend.db.session import SessionLocal
    from backend.db.models import WorkOrder

    db = SessionLocal()
    try:
        rows = db.query(WorkOrder).all()
        return [
            {
                "machine_id": r.machine_id,
                "issue_description": r.issue_description,
                "technician_notes": r.technician_notes,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    finally:
        db.close()


_SYNTHETIC_HIGH = [
    ("Emergency: complete motor failure on main drive", None, "X"),
    ("Critical safety hazard: fire risk from electrical fault", "Sparks observed near panel", "X"),
    ("Urgent: severe structural damage to main frame", None, "X"),
    ("CRITICAL: machine seized, safety lockout required", "Production halted", "X"),
    ("Emergency stop activated: catastrophic bearing failure", None, "X"),
    ("Safety hazard: major hydraulic burst, fluid spray", "Evacuated area", "X"),
    ("Critical failure: explosion risk in pressure vessel", None, "X"),
    ("Immediate shutdown: complete conveyor collapse", "Load bearing failed", "X"),
    ("Emergency: severe electrical fault, risk of fire", None, "X"),
    ("Critical: unsafe vibration levels, imminent failure", "Structural crack visible", "X"),
]


def _build_dataset(rows: list[dict]):
    X, y_sev, y_eta = [], [], []

    for r in rows:
        issue = r["issue_description"]
        notes = r.get("technician_notes")
        severity = assign_severity(issue)
        vec = extract_features(
            issue=issue,
            notes=notes,
            machine_id=r.get("machine_id", ""),
            created_at=r.get("created_at"),
        )
        total_words = int(vec[2])
        eta = assign_eta(severity, total_words)

        X.append(vec)
        y_sev.append(_LABEL_MAP[severity])
        y_eta.append(eta)

    # Inject synthetic high-severity examples (ensures model can predict all 3 classes)
    for issue, notes, machine_id in _SYNTHETIC_HIGH:
        vec = extract_features(issue=issue, notes=notes, machine_id=machine_id)
        X.append(vec)
        y_sev.append(_LABEL_MAP["high"])
        y_eta.append(assign_eta("high", int(vec[2])))

    return np.array(X, dtype=float), np.array(y_sev, dtype=int), np.array(y_eta, dtype=float)


def train_models(rows: Optional[list[dict]] = None) -> str:
    """
    Train severity classifier + ETA regressor and save to disk.

    rows: pre-parsed work order dicts (used by ingest pipeline to avoid a
          second DB round-trip). If None, loads from SQLite.

    Returns the path to the saved model file.
    """
    settings = get_settings()
    model_path = settings.model_path

    if rows is None:
        logger.info("Loading work orders from SQLite for training")
        rows = _load_rows_from_sqlite()

    if len(rows) < 10:
        logger.warning(
            "Too few records to train a reliable model",
            extra={"count": len(rows)},
        )

    logger.info("Building training dataset", extra={"rows": len(rows)})
    X, y_sev, y_eta = _build_dataset(rows)

    # ── Severity classifier ───────────────────────────────────────────────────
    # Compute per-sample weights to balance all three classes
    unique_classes, class_counts = np.unique(y_sev, return_counts=True)
    class_weight = {cls: len(y_sev) / (len(unique_classes) * cnt)
                    for cls, cnt in zip(unique_classes, class_counts)}
    sample_weights = np.array([class_weight[c] for c in y_sev])

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    clf.fit(X, y_sev, sample_weight=sample_weights)
    logger.info("Severity classifier trained", extra={"classes": int(len(np.unique(y_sev)))})

    # ── ETA regressor ─────────────────────────────────────────────────────────
    reg = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        verbosity=0,
    )
    reg.fit(X, y_eta)
    logger.info("ETA regressor trained")

    # ── Save bundle ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    bundle = {
        "severity_classifier": clf,
        "eta_regressor": reg,
        "label_names": _LABEL_NAMES,
        "feature_names": FEATURE_NAMES,
        "trained_on": len(rows),
        "version": "1.0",
    }
    joblib.dump(bundle, model_path)
    logger.info("Model bundle saved", extra={"path": model_path, "trained_on": len(rows)})
    return model_path


if __name__ == "__main__":
    path = train_models()
    print(f"Models saved to {path}")
