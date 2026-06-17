import React, { useState } from 'react';
import './App.css';
import QueryInput from './components/QueryInput';
import AgentPipeline from './components/AgentPipeline';
import AnswerPanel from './components/AnswerPanel';
import PdmPanel from './components/PdmPanel';
import { useAgentStream } from './hooks/useAgentStream';

export default function App() {
  const { agentSteps, isStreaming, finalResult, streamingAnswer, error, submitQuery, reset } =
    useAgentStream();
  const [hasQueried, setHasQueried] = useState(false);

  const handleSubmit = (query, topK, machineId) => {
    setHasQueried(true);
    submitQuery(query, topK, machineId);
  };

  const handleReset = () => {
    setHasQueried(false);
    reset();
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-logo">
            <span className="logo-glyph">⚙</span>
            <div className="logo-text-group">
              <span className="logo-name">AMOP</span>
              <span className="logo-sub">Autonomous Maintenance Operations Platform</span>
            </div>
          </div>

          <div className="header-right">
            {hasQueried && !isStreaming && (
              <button className="new-query-btn" onClick={handleReset}>
                + New Query
              </button>
            )}
            <div className="header-status-pill">
              <span className={`status-dot ${isStreaming ? 'status-dot--active' : 'status-dot--idle'}`} />
              <span>{isStreaming ? 'Processing…' : 'Ready'}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="app-main">
        <QueryInput onSubmit={handleSubmit} isLoading={isStreaming} />

        {hasQueried && (
          <div className="results-area">
            <AgentPipeline steps={agentSteps} isStreaming={isStreaming} />

            {error && (
              <div className="error-banner">
                <span className="error-banner__icon">✗</span>
                <span>{error}</span>
              </div>
            )}

            {(streamingAnswer || finalResult) && (
              <AnswerPanel
                result={finalResult}
                streamingText={streamingAnswer}
                isStreaming={isStreaming}
              />
            )}
          </div>
        )}

        {!hasQueried && (
          <div className="landing-hint">
            <div className="landing-hint__grid">
              <FeatureCard
                icon="🔍"
                title="Hybrid Retrieval"
                desc="BM25 + dense vector search fused via Reciprocal Rank Fusion across 20 000+ work orders"
              />
              <FeatureCard
                icon="🤖"
                title="Multi-Agent Pipeline"
                desc="Equipment extraction → intent routing → RAG → RCA → technician routing — all visible in real time"
              />
              <FeatureCard
                icon="📊"
                title="Failure Prediction"
                desc="XGBoost severity & ETA prediction from issue text; sensor-based PdM with SHAP explanations"
              />
              <FeatureCard
                icon="🔬"
                title="Root Cause Analysis"
                desc="Structured RCA with probable cause, contributing factors, corrective & preventive actions"
              />
            </div>
          </div>
        )}

        <PdmPanel />
      </main>

      <footer className="app-footer">
        AMOP · Phase 6 · Powered by OpenRouter + LangGraph + Qdrant
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="feature-card">
      <span className="feature-card__icon">{icon}</span>
      <div className="feature-card__title">{title}</div>
      <div className="feature-card__desc">{desc}</div>
    </div>
  );
}
