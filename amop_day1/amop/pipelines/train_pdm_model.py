"""
End-to-end PdM model training pipeline.

Runs feature engineering, XGBoost training, evaluation, SHAP analysis,
and Qdrant profile building in a single command:

    python -m pipelines.train_pdm_model

This does NOT touch the existing keyword-based failure_model.joblib used by
the work-order prediction endpoint — it writes to pdm_failure_model.joblib.
"""

from __future__ import annotations

import logging
import time

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    overall_start = time.time()

    # ── Step 1: Train XGBoost on PdM telemetry ────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Loading PdM datasets from dataset_files/...")
    print("="*60)

    from backend.prediction.pdm_trainer import train_pdm_model

    model_path = train_pdm_model()

    print(f"\n[OK] Model saved to: {model_path}")

    # ── Step 2: Build machine profiles in Qdrant ──────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Building machine profiles in Qdrant pdm_profiles...")
    print("="*60)

    from backend.prediction.pdm_semantic_bridge import build_and_store_profiles

    count = build_and_store_profiles()
    print(f"\n[OK] PdM profiles stored in Qdrant 'pdm_profiles' collection  ({count} machines)")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"PdM pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  - XGBoost model:     {model_path}")
    print(f"  - Qdrant profiles:   pdm_profiles collection  ({count} points)")
    print(f"\nNext steps:")
    print(f"  - Inference:  from backend.prediction.pdm_predictor import predict_failure")
    print(f"  - Similarity: from backend.prediction.pdm_semantic_bridge import find_similar_pdm_machine")
    print()


if __name__ == "__main__":
    main()
