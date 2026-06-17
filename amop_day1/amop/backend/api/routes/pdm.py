from fastapi import APIRouter, Depends, HTTPException

from backend.core.security import validate_api_key
from backend.schemas.work_order import PdmPredictionResult
from backend.prediction.pdm_predictor import predict_failure
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/pdm/predict/{machine_id}",
    response_model=PdmPredictionResult,
    dependencies=[Depends(validate_api_key)],
)
async def pdm_predict_endpoint(machine_id: int) -> PdmPredictionResult:
    """
    Predict failure probability for a PdM machine using sensor telemetry.

    machine_id must be a numeric ID in the range 1-100 (Azure PdM dataset).
    Returns 404 if the machine is not found in the telemetry dataset.
    Returns 503 if the PdM model has not been trained yet
    (run: python -m pipelines.train_pdm_model).
    """
    try:
        result = predict_failure(machine_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "PdM_telemetry.csv is missing from dataset_files/. "
                "Restore the file and restart the server."
            ),
        )

    if result is None:
        # predict_failure returns None for two reasons: model not trained, or
        # machine not in dataset. Distinguish by checking the model file.
        from pathlib import Path
        model_path = Path(__file__).resolve().parents[3] / "models" / "pdm_failure_model.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "PdM model has not been trained yet. "
                    "Run: python -m pipelines.train_pdm_model"
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=f"Machine {machine_id} not found in the PdM telemetry dataset (valid range: 1-100).",
        )

    return PdmPredictionResult(
        pdm_machine_id=result["machine_id"],
        failure_probability=result["failure_probability"],
        risk_tier=result["risk_tier"],
        top_features=result["top_features"],
        horizon=result["horizon"],
        match_basis="direct",
    )
