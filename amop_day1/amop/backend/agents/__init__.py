"""
Multi-agent system — LangGraph-based orchestration.

Agents:
    RouterAgent     — LLM-based intent classification (rag | predict | both | rca)
                      with keyword fallback when LLM is unavailable
    RAGAgent        — hybrid BM25 + Qdrant retrieval with LLM generation
    PredictionAgent — XGBoost severity/ETA estimation
    RCAAgent        — Root Cause Analysis: RAG history + specialized LLM reasoning
    SummarizerAgent — consolidates multi-agent output into final response

Entry point: run_agent_graph(query, top_k, machine_id) → dict
"""

from backend.agents.graph import run_agent_graph, stream_agent_graph

__all__ = ["run_agent_graph", "stream_agent_graph"]
