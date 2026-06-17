"""
POST /api/v1/eval — Run DeepEval metrics against the live RAG pipeline.

Workflow:
  1. Accept EvalRequest with optional test_cases list.
     If test_cases is empty, fall back to evaluation/test_cases.SAMPLE_TEST_CASES.
  2. For each test case that has no actual_output, query the live RAG pipeline
     and fill it in (so metrics measure the real system, not a cached answer).
  3. Run harness.run_evaluation() — calls DeepEval metrics (faithfulness,
     answer relevancy, contextual precision, contextual recall).
  4. Persist per-case scores to EvaluationResult table.
  5. Return EvalResponse with aggregate scores and eval_run_id.
"""

import uuid
import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.security import validate_api_key, sanitize_query
from backend.db.session import get_db
from backend.db.models import EvaluationResult
from backend.db.repositories import EvalResultRepo
from backend.schemas.work_order import EvalRequest, EvalResponse
from backend.agents import run_agent_graph
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _fill_actual_outputs(test_cases: list[dict]) -> list[dict]:
    """
    For any test case that has no actual_output, call the RAG pipeline live.
    Modifies test_cases in-place and returns the list.
    """
    for tc in test_cases:
        if tc.get("actual_output"):
            continue
        try:
            query = sanitize_query(tc["input"])
            result = run_agent_graph(query=query, top_k=5)
            tc["actual_output"] = result.get("answer", "")

            # If test case has no retrieval_context, use the retrieved sources
            if not tc.get("retrieval_context"):
                sources = result.get("sources", [])
                tc["retrieval_context"] = [
                    f"Machine {s['machine_id']} | Issue: {s['issue_description']} | "
                    f"Notes: {s.get('technician_notes') or 'N/A'}"
                    for s in sources
                ]
        except Exception as exc:
            logger.warning(
                "Failed to get live answer for eval test case",
                extra={"case_id": tc.get("id"), "error": str(exc)},
            )
            tc["actual_output"] = tc.get("actual_output") or ""
    return test_cases


@router.post("/eval", response_model=EvalResponse, dependencies=[Depends(validate_api_key)])
async def eval_endpoint(
    request: EvalRequest,
    db: Session = Depends(get_db),
) -> EvalResponse:
    """
    Run DeepEval evaluation against the live RAG pipeline.

    - Accepts up to 20 custom test cases, or uses built-in sample cases.
    - Queries the RAG pipeline for any test case missing an `actual_output`.
    - Scores each case with: answer_relevancy, faithfulness,
      contextual_precision, contextual_recall.
    - Stores results in the evaluation_results table.
    """
    from evaluation.harness import run_evaluation
    from evaluation.test_cases import SAMPLE_TEST_CASES

    run_id = request.run_id or str(uuid.uuid4())

    # Use provided test cases or fall back to built-in samples
    raw_cases = request.test_cases if request.test_cases else list(SAMPLE_TEST_CASES)
    # Cap at 20 to avoid excessively long eval runs on the free LLM tier
    raw_cases = raw_cases[:20]

    # Fill actual_output via live RAG calls (run in executor to stay async-safe)
    filled_cases = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _fill_actual_outputs(raw_cases),
    )

    # Run DeepEval
    try:
        eval_output = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_evaluation(filled_cases, run_id=run_id),
        )
    except Exception as exc:
        logger.error("DeepEval run failed", extra={"run_id": run_id, "error": str(exc)}, exc_info=True)
        # Return partial result so the caller knows what happened
        return EvalResponse(
            eval_run_id=run_id,
            results={"error": -1.0},
            total_cases=len(filled_cases),
        )

    # Persist per-case metric scores to the database
    repo = EvalResultRepo(db)
    db_results: list[EvaluationResult] = []
    for case_info in eval_output.get("per_case", []):
        case_id = case_info.get("id", "unknown")
        for metric_name, score in case_info.get("scores", {}).items():
            db_results.append(
                EvaluationResult(
                    eval_run_id=run_id,
                    test_case_id=case_id,
                    metric_name=metric_name,
                    score=float(score),
                )
            )
    if db_results:
        repo.bulk_insert(db_results)

    logger.info(
        "Evaluation complete",
        extra={
            "run_id": run_id,
            "total_cases": eval_output["total_cases"],
            "passed": eval_output.get("passed", 0),
            "results": eval_output["results"],
        },
    )

    return EvalResponse(
        eval_run_id=run_id,
        results=eval_output["results"],
        total_cases=eval_output["total_cases"],
    )
