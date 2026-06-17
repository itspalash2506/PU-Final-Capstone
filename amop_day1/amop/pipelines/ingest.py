"""
Work order ingestion pipeline.

Flow: CSV content → parse rows → SQLite bulk insert
                               → BM25 index (pickle)
                               → Qdrant dense vectors (upsert)

The three stages are independent and each returns a bool indicating success.
Call run_ingest() from the /ingest route or directly from the CLI.
"""

import csv
import hashlib
import json
import os
import pickle
import uuid
from datetime import datetime
from io import StringIO
from typing import Optional

from rank_bm25 import BM25Okapi
from qdrant_client.models import PointStruct

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.db.session import SessionLocal
from backend.db.models import WorkOrder
from backend.db.repositories import WorkOrderRepo
from backend.clients.embedding_client import encode
from backend.clients.vector_store import ensure_collection, upsert_points

logger = get_logger(__name__)

_FILLER_VALUES = {"n/a", "na", "not applicable", "none", "n.a.", "-", ""}
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y")
_EMBED_BATCH = 64


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> Optional[datetime]:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_techs(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Return (primary_tech, all_techs_json_string)."""
    if not raw or not raw.strip():
        return None, None
    parts = [t.strip() for t in raw.split(",") if t.strip()]
    return (parts[0], json.dumps(parts)) if parts else (None, None)


def _is_filler(text: str) -> bool:
    return text.strip().lower() in _FILLER_VALUES


def _row_id(machine_id: str, date_received: str, issue: str) -> str:
    """Deterministic UUID v5 from content — re-ingesting the same row produces the same ID."""
    key = f"{machine_id}|{date_received}|{issue}".lower().encode()
    return str(uuid.UUID(bytes=hashlib.sha256(key).digest()[:16], version=4))


def _doc_text(row: dict) -> str:
    """Combine machine_id + issue + notes into a single string for embedding/BM25.

    machine_id is prepended so both BM25 and the dense vector carry machine
    identity — without it, queries like 'failure on A56' score zero for 'A56'
    because the machine ID never appeared in the indexed text.
    """
    parts = [f"Machine {row['machine_id']}", row["issue_description"]]
    if row.get("technician_notes"):
        parts.append(row["technician_notes"])
    return " ".join(parts)


# ── Stage 1: Parse CSV ────────────────────────────────────────────────────────

def parse_csv(content: str) -> list[dict]:
    """
    Parse MWO CSV (columns: mach, date_received, issue, info, tech).
    Returns normalized row dicts ready for the three ingestion stages.
    Rows missing machine_id or issue are silently dropped.
    """
    reader = csv.DictReader(StringIO(content))
    rows: list[dict] = []

    for raw in reader:
        row = {k.strip().lower(): (v or "").strip() for k, v in raw.items()}

        machine_id = row.get("mach", "")
        issue = row.get("issue", "")
        if not machine_id or not issue:
            continue

        info = row.get("info", "")
        date_received = row.get("date_received", "")
        has_notes = not _is_filler(info)
        primary_tech, all_techs = _parse_techs(row.get("tech", ""))

        rows.append({
            "id": _row_id(machine_id, date_received, issue),
            "machine_id": machine_id,
            "issue_description": issue,
            "technician_notes": info if has_notes else None,
            "primary_tech": primary_tech,
            "all_techs": all_techs,
            "has_notes": has_notes,
            "created_at": _parse_date(row.get("date_received", "")) or datetime.utcnow(),
        })

    logger.info("CSV parsed", extra={"rows": len(rows)})
    return rows


# ── Stage 2: SQLite ───────────────────────────────────────────────────────────

def save_to_sqlite(rows: list[dict]) -> int:
    """Upsert parsed rows into work_orders, skipping duplicates. Returns count of new rows."""
    db = SessionLocal()
    try:
        work_orders = [WorkOrder(**r) for r in rows]
        return WorkOrderRepo(db).bulk_upsert(work_orders)
    finally:
        db.close()


# ── Stage 3a: BM25 index ──────────────────────────────────────────────────────

def build_bm25_index(rows: list[dict]) -> bool:
    """
    Tokenise issue+notes, fit BM25Okapi, and pickle the index to disk.
    The pickle contains both the BM25 object and the ordered list of doc IDs
    so the retriever can map BM25 result indices back to work order UUIDs.
    """
    settings = get_settings()
    index_path = settings.bm25_index_path
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)

    corpus = [_doc_text(r).lower().split() for r in rows]
    doc_ids = [r["id"] for r in rows]

    bm25 = BM25Okapi(corpus)

    with open(index_path, "wb") as f:
        pickle.dump({"index": bm25, "doc_ids": doc_ids}, f)

    logger.info("BM25 index saved", extra={"docs": len(corpus), "path": index_path})
    return True


# ── Stage 3b: Qdrant dense vectors ────────────────────────────────────────────

def embed_and_upsert(rows: list[dict]) -> bool:
    """
    Embed each work order and upsert into Qdrant in batches of _EMBED_BATCH.
    The Qdrant point ID matches the SQLite WorkOrder.id (UUID string).
    """
    ensure_collection()
    total = 0

    for i in range(0, len(rows), _EMBED_BATCH):
        batch = rows[i : i + _EMBED_BATCH]
        texts = [_doc_text(r) for r in batch]
        vectors = encode(texts)

        points = [
            PointStruct(
                id=r["id"],
                vector=v,
                payload={
                    "machine_id": r["machine_id"],
                    "issue_description": r["issue_description"],
                    "technician_notes": r.get("technician_notes"),
                    "primary_tech": r.get("primary_tech"),
                    "has_notes": r["has_notes"],
                    "created_at": r["created_at"].isoformat(),
                    "effectiveness_score": 0.5,
                },
            )
            for r, v in zip(batch, vectors)
        ]

        upsert_points(points)
        total += len(points)
        logger.info("Qdrant batch upserted", extra={"batch_start": i, "upserted": total})

    return True


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_ingest(content: str) -> dict:
    """
    Full pipeline: parse → SQLite → BM25 → Qdrant.
    Returns a result dict consumed by IngestResponse.
    """
    rows = parse_csv(content)
    if not rows:
        return {
            "inserted": 0,
            "bm25_indexed": False,
            "qdrant_indexed": False,
            "message": "No valid rows found in CSV.",
        }

    inserted = save_to_sqlite(rows)

    bm25_ok = False
    qdrant_ok = False

    try:
        bm25_ok = build_bm25_index(rows)
    except Exception as exc:
        logger.error("BM25 indexing failed", extra={"error": str(exc)}, exc_info=True)

    try:
        qdrant_ok = embed_and_upsert(rows)
    except Exception as exc:
        logger.error("Qdrant upsert failed", extra={"error": str(exc)}, exc_info=True)

    # ── Stage 4: Train ML prediction models ──────────────────────────────────
    try:
        from backend.prediction.trainer import train_models
        from backend.prediction.predictor import reload_model
        train_models(rows=rows)
        reload_model()  # clear lru_cache so next predict call uses fresh model
        logger.info("ML models retrained after ingest")
    except Exception as exc:
        logger.error("ML model training failed (non-fatal)", extra={"error": str(exc)}, exc_info=True)

    return {
        "inserted": inserted,
        "bm25_indexed": bm25_ok,
        "qdrant_indexed": qdrant_ok,
        "message": f"Ingested {inserted} new work orders ({len(rows) - inserted} duplicates skipped).",
    }
