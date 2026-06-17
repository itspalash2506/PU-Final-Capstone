from fastapi import APIRouter, Depends, HTTPException

from backend.core.security import validate_api_key
from backend.schemas.work_order import PredictionRequest, PredictionResponse
from backend.prediction.predictor import predict, ModelNotTrainedError
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(validate_api_key)],
)
async def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """
    Predict failure severity and estimated repair time for a work order.

    The model is trained automatically after each CSV ingest.
    Returns 503 if no model has been trained yet.
    """
    try:
        result = predict(
            machine_id=request.machine_id,
            issue_description=request.issue_description,
            technician_notes=request.technician_notes,
        )
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction service error.")

    return PredictionResponse(**result)
