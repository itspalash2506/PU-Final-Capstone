import json
import uuid
import asyncio
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.core.security import validate_api_key, sanitize_query
from backend.db.session import get_db
from backend.db.repositories import QueryLogRepo, AgentTraceRepo
from backend.schemas.work_order import QueryRequest, QueryResponse, SourceDoc
from backend.agents import run_agent_graph, stream_agent_graph
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(validate_api_key)])
async def query_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """
    Submit a maintenance question.  Runs through the multi-agent graph:
        Router → RAG (hybrid BM25 + Qdrant + LLM) → Summarizer
    For prediction queries: Router → Prediction → Summarizer
    For combined queries:   Router → RAG → Prediction → Summarizer

    All queries and agent traces are logged for tracing and feedback.
    """
    clean_query = sanitize_query(request.query)
    request_id = str(uuid.uuid4())

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_agent_graph(
            clean_query,
            top_k=request.top_k,
            machine_id=request.machine_id,
        ),
    )

    # Persist query log
    QueryLogRepo(db).create(
        request_id=request_id,
        query_text=clean_query,
        machine_id=request.machine_id,
        response_text=result["answer"],
        latency_ms=result["latency_ms"],
        retrieved_doc_ids=[s["id"] for s in result["sources"]],
    )

    # Persist per-agent traces
    if result.get("agent_traces"):
        trace_repo = AgentTraceRepo(db)
        for trace in result["agent_traces"]:
            trace_repo.log_agent(
                query_log_id=request_id,
                agent_name=trace["agent_name"],
                status=trace["status"],
                latency_ms=trace.get("latency_ms"),
                error_message=trace.get("error"),
            )

    sources = [
        SourceDoc(
            id=s["id"],
            machine_id=s["machine_id"],
            issue_description=s["issue_description"],
            technician_notes=s.get("technician_notes"),
            score=s["score"],
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        request_id=request_id,
        answer=result["answer"],
        sources=sources,
        latency_ms=result["latency_ms"],
        intent=result.get("intent"),
        agent_traces=result.get("agent_traces", []),
        routing_result=result.get("routing_result"),
        rca_result=result.get("rca_result"),
        router_method=result.get("router_method"),
        router_model=result.get("router_model"),
        alpha_machine_id=result.get("alpha_machine_id"),
        numeric_machine_id=result.get("numeric_machine_id"),
        id_assumption=result.get("id_assumption", False),
    )


@router.post("/query/stream", dependencies=[Depends(validate_api_key)])
async def query_stream_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    SSE streaming endpoint — emits per-agent progress events as the pipeline
    executes, then a final 'complete' event with the full answer and metadata.

    Clients consume this with fetch() + ReadableStream (not EventSource, since
    the request body is POST JSON).  Each event is a JSON object on a
    'data: ...' line followed by a blank line (standard SSE framing).
    """
    clean_query = sanitize_query(request.query)
    request_id = str(uuid.uuid4())

    async def _generate():
        yield f"data: {json.dumps({'type': 'connected', 'request_id': request_id})}\n\n"

        final_event: dict = {}
        try:
            async for event in stream_agent_graph(
                clean_query,
                top_k=request.top_k,
                machine_id=request.machine_id,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") == "complete":
                    final_event = event
        except Exception as exc:
            logger.error("SSE stream error", extra={"error": str(exc)}, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        # Persist to DB after stream completes
        if final_event:
            raw_sources = final_event.get("sources", [])
            QueryLogRepo(db).create(
                request_id=request_id,
                query_text=clean_query,
                machine_id=request.machine_id,
                response_text=final_event.get("answer", ""),
                latency_ms=final_event.get("latency_ms", 0),
                retrieved_doc_ids=[
                    s.get("id") for s in raw_sources if isinstance(s, dict)
                ],
            )
            for trace in final_event.get("agent_traces", []):
                AgentTraceRepo(db).log_agent(
                    query_log_id=request_id,
                    agent_name=trace["agent_name"],
                    status=trace["status"],
                    latency_ms=trace.get("latency_ms"),
                    error_message=trace.get("error"),
                )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
