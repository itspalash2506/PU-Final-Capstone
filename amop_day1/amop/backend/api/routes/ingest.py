import asyncio
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from backend.core.security import validate_api_key
from backend.schemas.work_order import IngestResponse
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(validate_api_key)])
async def ingest_work_orders(file: UploadFile = File(..., description="MWO CSV file")) -> IngestResponse:
    """
    Upload a CSV of work orders (columns: mach, date_received, issue, info, tech).
    Parses rows, saves to SQLite, builds BM25 index, and upserts dense vectors to Qdrant.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content_bytes = await file.read()
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = content_bytes.decode("latin-1")

    logger.info("Ingest started", extra={"upload_filename": file.filename, "bytes": len(content_bytes)})

    # Run the blocking pipeline in a thread pool so we don't block the event loop
    from pipelines.ingest import run_ingest
    result = await asyncio.get_event_loop().run_in_executor(None, run_ingest, content)

    return IngestResponse(**result)
