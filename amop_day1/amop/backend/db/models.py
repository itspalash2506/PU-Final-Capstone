import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    Text, DateTime, ForeignKey, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class WorkOrder(Base):
    """
    Relational copy of ingested MWO work orders.
    Actual dataset columns: mach, date_received, issue, info, tech
    Internal names used here for clarity.
    Cross-referenced to Qdrant via id (same UUID as Qdrant point ID).
    """
    __tablename__ = "work_orders"

    id = Column(String, primary_key=True, default=_uuid)
    machine_id = Column(String, nullable=False, index=True)       # from: mach
    issue_description = Column(Text, nullable=False)               # from: issue
    technician_notes = Column(Text, nullable=True)                 # from: info (null if filler)
    primary_tech = Column(String, nullable=True)                   # first value from: tech
    all_techs = Column(Text, nullable=True)                        # full: tech (JSON array string)
    has_notes = Column(Boolean, default=True, nullable=False)      # False for 4% filler rows
    created_at = Column(DateTime, nullable=False)                  # from: date_received


class QueryLog(Base):
    """
    Every user query with its full agent workflow result.
    request_id ties this to agent_traces for debugging.
    retrieved_doc_ids is a JSON array of Qdrant point UUIDs.
    """
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=_uuid)           # request_id returned in response
    query_text = Column(Text, nullable=False)
    machine_id = Column(String, nullable=True)                     # extracted by Equipment Agent
    response_text = Column(Text, nullable=True)                    # final Judge Agent response
    latency_ms = Column(Integer, nullable=True)
    retrieved_doc_ids = Column(Text, nullable=True)                # JSON: ["uuid1", "uuid2", ...]
    created_at = Column(DateTime, default=_now, nullable=False)

    # Relationships
    feedback = relationship("Feedback", back_populates="query_log", cascade="all, delete-orphan")
    traces = relationship("AgentTrace", back_populates="query_log", cascade="all, delete-orphan")


class Feedback(Base):
    """
    User thumbs up/down on a query response.
    On positive feedback: effectiveness_score incremented in Qdrant payload.
    On negative feedback: effectiveness_score decremented.
    """
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=_uuid)
    query_log_id = Column(String, ForeignKey("query_logs.id"), nullable=False)
    rating = Column(Integer, nullable=False)                        # 1 = positive, -1 = negative
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    query_log = relationship("QueryLog", back_populates="feedback")


class EvaluationResult(Base):
    """
    DeepEval metric score per test case per eval run.
    Each run produces N rows (one per metric per test case).
    """
    __tablename__ = "evaluation_results"

    id = Column(String, primary_key=True, default=_uuid)
    eval_run_id = Column(String, nullable=False, index=True)
    test_case_id = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)                   # faithfulness | relevance | correctness | hallucination
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)


class AgentTrace(Base):
    """
    Per-agent execution record for every query.
    Used for debugging latency and failure patterns.
    """
    __tablename__ = "agent_traces"

    id = Column(String, primary_key=True, default=_uuid)
    query_log_id = Column(String, ForeignKey("query_logs.id"), nullable=False)
    agent_name = Column(String, nullable=False)                    # equipment|retrieval|rca|prediction|routing|judge
    status = Column(String, nullable=False)                        # success | failed | skipped
    latency_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    query_log = relationship("QueryLog", back_populates="traces")

    __table_args__ = (
        Index("ix_agent_traces_query_log_id", "query_log_id"),
    )
