from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    inserted: int
    bm25_indexed: bool
    qdrant_indexed: bool
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    machine_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDoc(BaseModel):
    id: str
    machine_id: str
    issue_description: str
    technician_notes: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceDoc]
    latency_ms: int
    intent: Optional[str] = None  # which agent path was taken: rag | predict | both | rca
    agent_traces: list[dict] = Field(default_factory=list)
    routing_result: Optional[dict] = None
    rca_result: Optional[dict] = None
    router_method: Optional[str] = None
    router_model: Optional[str] = None
    alpha_machine_id: Optional[str] = None
    numeric_machine_id: Optional[int] = None
    id_assumption: bool = False


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_log_id: str
    rating: int = Field(..., ge=-1, le=1, description="1 = positive, -1 = negative")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    message: str


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    machine_id: str
    issue_description: str = Field(..., min_length=1, max_length=500)
    technician_notes: Optional[str] = None


class PdmFeature(BaseModel):
    name: str
    shap_value: float
    direction: str


class PdmPredictionResult(BaseModel):
    pdm_machine_id: int
    failure_probability: float
    risk_tier: str
    top_features: list[PdmFeature]
    horizon: str
    match_basis: str          # "direct" | "symptom_similarity"
    similarity_score: Optional[float] = None  # set when match_basis is symptom_similarity


class PredictionResponse(BaseModel):
    severity: str
    severity_confidence: float
    eta_hours: float
    features_used: list[str]


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    run_id: Optional[str] = None
    test_cases: list[dict] = Field(default_factory=list)


class EvalResponse(BaseModel):
    eval_run_id: str
    results: dict[str, float]
    total_cases: int
