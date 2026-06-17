"""
ML prediction pipeline (Phase 5).

    features  : keyword-based feature extraction from work order text
    trainer   : XGBoost classifier (severity) + regressor (ETA) training
    predictor : inference — loads saved bundle, runs single-record prediction

Training is triggered automatically after each CSV ingest (pipelines/ingest.py).
Endpoint: POST /api/v1/predict
"""

from backend.prediction.predictor import predict, ModelNotTrainedError, reload_model

__all__ = ["predict", "ModelNotTrainedError", "reload_model"]
