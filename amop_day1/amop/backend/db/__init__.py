"""
Database layer: ORM models, repository DAOs, and session management.
"""

from backend.db.models import (
    Base,
    WorkOrder,
    QueryLog,
    Feedback,
    EvaluationResult,
    AgentTrace,
)
from backend.db.session import init_db, get_db, SessionLocal
from backend.db.repositories import (
    WorkOrderRepo,
    QueryLogRepo,
    FeedbackRepo,
    EvalResultRepo,
    AgentTraceRepo,
)

__all__ = [
    # ORM base + models
    "Base",
    "WorkOrder",
    "QueryLog",
    "Feedback",
    "EvaluationResult",
    "AgentTrace",
    # Session utilities
    "init_db",
    "get_db",
    "SessionLocal",
    # Repositories
    "WorkOrderRepo",
    "QueryLogRepo",
    "FeedbackRepo",
    "EvalResultRepo",
    "AgentTraceRepo",
]
