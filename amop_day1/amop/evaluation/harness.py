"""
DeepEval evaluation harness for AMOP RAG pipeline.

Runs up to four metrics against a set of test cases:
  - answer_relevancy     : Is the generated answer relevant to the question?
  - faithfulness         : Is every claim in the answer grounded in the retrieved context?
  - contextual_precision : Are the retrieved chunks ranked well (relevant chunks first)?
  - contextual_recall    : Does the retrieved context cover the expected ground-truth answer?

DeepEval uses an LLM judge internally. We configure it to use OpenRouter so we don't
need a separate OpenAI API key — the same credentials used by the AMOP backend are reused.

Usage (standalone):
    python -m evaluation.harness

Usage (programmatic):
    from evaluation.harness import run_evaluation
    results = run_evaluation(test_cases)
"""

from __future__ import annotations

import os
import uuid
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _configure_deepeval_openrouter() -> str:
    """
    Point DeepEval's internal LLM judge at OpenRouter via OpenAI-compatible env vars.
    Returns the model name to pass to metric constructors.
    """
    from backend.core.config import get_settings
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("OPENAI_API_BASE", settings.openrouter_base_url)
    return settings.openrouter_model


_EVAL_MODEL = _configure_deepeval_openrouter()

from deepeval.test_case import LLMTestCase  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_THRESHOLDS = {
    "answer_relevancy": 0.7,
    "faithfulness": 0.7,
    "contextual_precision": 0.6,
    "contextual_recall": 0.6,
}


def _build_metrics(thresholds: dict[str, float] | None = None) -> list[tuple[str, object]]:
    """
    Returns a list of (name, metric) pairs.
    Metric construction is guarded per-metric so a missing import for one
    metric does not prevent the others from running.
    """
    t = thresholds or _THRESHOLDS
    result: list[tuple[str, object]] = []

    def _try_add(name: str, cls_path: str, kwargs: dict):
        try:
            module_name, cls_name = cls_path.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name)
            try:
                metric = cls(threshold=kwargs["threshold"], model=_EVAL_MODEL)
            except TypeError:
                # Older deepeval — no model parameter
                metric = cls(threshold=kwargs["threshold"])
            result.append((name, metric))
        except (ImportError, AttributeError) as exc:
            logger.warning(f"Metric {name} unavailable: {exc}")

    _try_add("answer_relevancy",     "deepeval.metrics.AnswerRelevancyMetric",     {"threshold": t["answer_relevancy"]})
    _try_add("faithfulness",         "deepeval.metrics.FaithfulnessMetric",        {"threshold": t["faithfulness"]})
    _try_add("contextual_precision", "deepeval.metrics.ContextualPrecisionMetric", {"threshold": t["contextual_precision"]})
    _try_add("contextual_recall",    "deepeval.metrics.ContextualRecallMetric",    {"threshold": t["contextual_recall"]})

    return result


def _to_llm_test_case(tc: dict) -> LLMTestCase:
    return LLMTestCase(
        input=tc["input"],
        actual_output=tc.get("actual_output") or "",
        expected_output=tc.get("expected_output") or "",
        retrieval_context=tc.get("retrieval_context") or [],
    )


def _score_metric(metric, test_case: LLMTestCase) -> float:
    try:
        metric.measure(test_case)
        return float(metric.score) if metric.score is not None else 0.0
    except Exception as exc:
        logger.warning(
            "Metric measurement failed",
            extra={"metric": type(metric).__name__, "error": str(exc)},
        )
        return 0.0


def run_evaluation(
    test_cases: list[dict],
    run_id: Optional[str] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> dict:
    """
    Evaluate test cases with DeepEval.

    Returns:
        {
            "eval_run_id": str,
            "results": {metric_name: avg_score, ...},
            "per_case": [{"id": str, "scores": {metric_name: score}}],
            "total_cases": int,
            "passed": int,
        }
    """
    if not test_cases:
        return {
            "eval_run_id": run_id or str(uuid.uuid4()),
            "results": {},
            "per_case": [],
            "total_cases": 0,
            "passed": 0,
        }

    eval_run_id = run_id or str(uuid.uuid4())
    named_metrics = _build_metrics(thresholds)

    logger.info(
        "Starting DeepEval run",
        extra={
            "eval_run_id": eval_run_id,
            "num_cases": len(test_cases),
            "metrics": [n for n, _ in named_metrics],
        },
    )

    per_case: list[dict] = []
    aggregated: dict[str, list[float]] = {name: [] for name, _ in named_metrics}
    passed = 0

    for i, tc_raw in enumerate(test_cases):
        case_id = tc_raw.get("id", f"case_{i+1}")
        llm_tc = _to_llm_test_case(tc_raw)
        scores: dict[str, float] = {}
        case_passed = True

        for name, metric in named_metrics:
            score = _score_metric(metric, llm_tc)
            scores[name] = score
            aggregated[name].append(score)
            threshold = (thresholds or _THRESHOLDS).get(name, 0.5)
            if score < threshold:
                case_passed = False

        if case_passed:
            passed += 1

        per_case.append({"id": case_id, "scores": scores})
        logger.info("Test case scored", extra={"case_id": case_id, "scores": scores})

    results = {
        name: (sum(vals) / len(vals) if vals else 0.0)
        for name, vals in aggregated.items()
    }

    logger.info(
        "DeepEval run complete",
        extra={"eval_run_id": eval_run_id, "results": results, "passed": passed},
    )

    return {
        "eval_run_id": eval_run_id,
        "results": results,
        "per_case": per_case,
        "total_cases": len(test_cases),
        "passed": passed,
    }


if __name__ == "__main__":
    from evaluation.test_cases import SAMPLE_TEST_CASES
    output = run_evaluation(SAMPLE_TEST_CASES)
    print(f"\nEval run: {output['eval_run_id']}")
    print(f"Total cases: {output['total_cases']} | Passed: {output['passed']}")
    print("\nAggregate scores:")
    for metric, score in output["results"].items():
        print(f"  {metric:<30} {score:.4f}")
