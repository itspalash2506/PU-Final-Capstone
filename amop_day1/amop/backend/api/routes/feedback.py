"""
POST /api/v1/feedback

Accepts a thumbs up (rating=1) or thumbs down (rating=-1) on a query response.

Side effects:
1. Writes feedback record to SQLite feedback table
2. Updates effectiveness_score in Qdrant payload for each retrieved document
   - rating=1:  score = min(score + 0.1, 1.0)
   - rating=-1: score = max(score - 0.1, 0.0)
   - Clamped to [0.0, 1.0] always

Why update Qdrant:
The reranker uses effectiveness_score as a signal (10% weight).
Documents that consistently get positive feedback float higher in future results.
Documents that consistently get negative feedback sink lower.
This closes the feedback loop without any retraining.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db, validate_api_key
from backend.schemas.work_order import FeedbackRequest, FeedbackResponse
from backend.db.repositories import FeedbackRepo, QueryLogRepo
from backend.clients.vector_store import get_qdrant_client
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key),
) -> FeedbackResponse:
    """
    Submit feedback on a query response.

    Requires the query_log_id from the original /query response.
    Updates Qdrant effectiveness_score for all retrieved documents
    associated with that query.
    """
    # 1. Verify the query log exists
    query_log = QueryLogRepo(db).get(body.query_log_id)
    if not query_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query log {body.query_log_id} not found",
        )

    # 2. Write feedback to SQLite
    feedback = FeedbackRepo(db).create(
        query_log_id=body.query_log_id,
        rating=body.rating,
        comment=body.comment,
    )

    # 3. Update effectiveness_score in Qdrant for each retrieved document
    doc_ids: list[str] = []
    if query_log.retrieved_doc_ids:
        try:
            doc_ids = json.loads(query_log.retrieved_doc_ids)
        except (json.JSONDecodeError, TypeError):
            doc_ids = []

    if doc_ids:
        _update_qdrant_effectiveness(doc_ids, body.rating)

    logger.info(
        "Feedback submitted",
        extra={
            "feedback_id": feedback.id,
            "query_log_id": body.query_log_id,
            "rating": body.rating,
            "docs_updated": len(doc_ids),
        },
    )

    return FeedbackResponse(
        id=feedback.id,
        message=f"Feedback recorded. Updated effectiveness score for {len(doc_ids)} documents.",
    )


def _update_qdrant_effectiveness(doc_ids: list[str], rating: int) -> None:
    """
    Update effectiveness_score in Qdrant payload for each document.

    Uses Qdrant's set_payload to update only the effectiveness_score field
    without touching any other payload fields.

    rating=1  → increment by 0.1, clamped to max 1.0
    rating=-1 → decrement by 0.1, clamped to min 0.0
    """
    settings = get_settings()
    client = get_qdrant_client()
    delta = 0.1 if rating == 1 else -0.1

    updated = 0
    failed = 0

    for doc_id in doc_ids:
        try:
            results = client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[doc_id],
                with_payload=True,
            )
            if not results:
                continue

            current_score = results[0].payload.get("effectiveness_score", 0.5)
            new_score = round(max(0.0, min(1.0, current_score + delta)), 3)

            client.set_payload(
                collection_name=settings.qdrant_collection,
                payload={"effectiveness_score": new_score},
                points=[doc_id],
            )
            updated += 1

        except Exception as e:
            logger.warning(
                "Failed to update Qdrant effectiveness_score",
                extra={"doc_id": doc_id, "error": str(e)},
            )
            failed += 1

    logger.info(
        "Qdrant effectiveness scores updated",
        extra={"updated": updated, "failed": failed, "delta": delta},
    )
