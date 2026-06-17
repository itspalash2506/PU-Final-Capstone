import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from backend.db.models import WorkOrder, QueryLog, Feedback, EvaluationResult, AgentTrace
from backend.core.logging import get_logger

logger = get_logger(__name__)


class WorkOrderRepo:
    def __init__(self, db: Session):
        self.db = db

    def upsert(self, work_order: WorkOrder) -> None:
        """Insert or replace a work order record by id."""
        existing = self.db.get(WorkOrder, work_order.id)
        if existing:
            # Update fields
            existing.machine_id = work_order.machine_id
            existing.issue_description = work_order.issue_description
            existing.technician_notes = work_order.technician_notes
            existing.primary_tech = work_order.primary_tech
            existing.all_techs = work_order.all_techs
            existing.has_notes = work_order.has_notes
            existing.created_at = work_order.created_at
        else:
            self.db.add(work_order)
        self.db.commit()

    def bulk_insert(self, work_orders: list[WorkOrder]) -> None:
        """Batch insert for ingestion pipeline — much faster than individual inserts."""
        self.db.bulk_save_objects(work_orders)
        self.db.commit()
        logger.info("Bulk inserted work orders", extra={"count": len(work_orders)})

    def bulk_upsert(self, work_orders: list[WorkOrder]) -> int:
        """Insert only rows whose ID does not already exist. Returns count of new rows inserted."""
        if not work_orders:
            return 0
        ids = [wo.id for wo in work_orders]
        existing = {
            row[0]
            for row in self.db.query(WorkOrder.id).filter(WorkOrder.id.in_(ids)).all()
        }
        new_orders = [wo for wo in work_orders if wo.id not in existing]
        if new_orders:
            self.db.bulk_save_objects(new_orders)
            self.db.commit()
        skipped = len(work_orders) - len(new_orders)
        logger.info("Bulk upsert complete", extra={"new": len(new_orders), "skipped": skipped})
        return len(new_orders)

    def get_technician_history(self, machine_id: str) -> list[str]:
        """Return list of all technicians who worked on a given machine."""
        rows = (
            self.db.query(WorkOrder.all_techs)
            .filter(WorkOrder.machine_id == machine_id)
            .filter(WorkOrder.all_techs.isnot(None))
            .all()
        )
        techs: list[str] = []
        for (all_techs_json,) in rows:
            try:
                techs.extend(json.loads(all_techs_json))
            except (json.JSONDecodeError, TypeError):
                pass
        return techs


class QueryLogRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        request_id: str,
        query_text: str,
        machine_id: Optional[str],
        response_text: Optional[str],
        latency_ms: Optional[int],
        retrieved_doc_ids: Optional[list[str]],
    ) -> QueryLog:
        log = QueryLog(
            id=request_id,
            query_text=query_text,
            machine_id=machine_id,
            response_text=response_text,
            latency_ms=latency_ms,
            retrieved_doc_ids=json.dumps(retrieved_doc_ids or []),
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def get(self, request_id: str) -> Optional[QueryLog]:
        return self.db.get(QueryLog, request_id)


class FeedbackRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        query_log_id: str,
        rating: int,
        comment: Optional[str] = None,
    ) -> Feedback:
        feedback = Feedback(
            query_log_id=query_log_id,
            rating=rating,
            comment=comment,
        )
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback


class EvalResultRepo:
    def __init__(self, db: Session):
        self.db = db

    def bulk_insert(self, results: list[EvaluationResult]) -> None:
        self.db.bulk_save_objects(results)
        self.db.commit()

    def get_run_summary(self, eval_run_id: str) -> dict[str, float]:
        """Return average score per metric for an eval run."""
        rows = (
            self.db.query(EvaluationResult.metric_name, EvaluationResult.score)
            .filter(EvaluationResult.eval_run_id == eval_run_id)
            .all()
        )
        from collections import defaultdict
        buckets: dict[str, list[float]] = defaultdict(list)
        for metric, score in rows:
            buckets[metric].append(score)
        return {metric: sum(scores) / len(scores) for metric, scores in buckets.items()}

    def count_run(self, eval_run_id: str) -> int:
        return (
            self.db.query(EvaluationResult)
            .filter(EvaluationResult.eval_run_id == eval_run_id)
            .count()
        )


class AgentTraceRepo:
    def __init__(self, db: Session):
        self.db = db

    def log_agent(
        self,
        query_log_id: str,
        agent_name: str,
        status: str,
        latency_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> AgentTrace:
        trace = AgentTrace(
            query_log_id=query_log_id,
            agent_name=agent_name,
            status=status,
            latency_ms=latency_ms,
            error_message=error_message,
        )
        self.db.add(trace)
        self.db.commit()
        return trace
