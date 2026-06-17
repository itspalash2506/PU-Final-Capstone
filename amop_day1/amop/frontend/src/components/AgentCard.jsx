import React from 'react';

const AGENT_META = {
  equipment: { icon: '🔍', color: '#60a5fa' },
  router:    { icon: '🔀', color: '#a78bfa' },
  rag:       { icon: '📚', color: '#34d399' },
  rca:       { icon: '🔬', color: '#fb923c' },
  prediction:{ icon: '📊', color: '#f472b6' },
  routing:   { icon: '👷', color: '#fbbf24' },
  summarizer:{ icon: '✍️', color: '#22d3ee' },
};

export default function AgentCard({ step, isCurrent }) {
  const { agent, label, status, latency_ms, summary, data = {} } = step;
  const meta = AGENT_META[agent] || { icon: '⚙️', color: '#94a3b8' };

  return (
    <div className={`agent-card agent-card--${status} ${isCurrent ? 'agent-card--current' : ''}`}>
      <div className="agent-card__header">
        <span className="agent-card__icon">{meta.icon}</span>
        <div className="agent-card__title-group">
          <span className="agent-card__label">{label}</span>
          {latency_ms != null && status === 'complete' && (
            <span className="agent-card__latency">Agent time: {latency_ms}ms</span>
          )}
        </div>
        <StatusBadge status={status} isCurrent={isCurrent} />
      </div>

      {summary && (
        <div className="agent-card__summary">{summary}</div>
      )}

      {status === 'complete' && agent === 'router' && data.intent && (
        <div className="agent-card__tags">
          <span className="agent-card__tag" style={{ borderColor: meta.color, color: meta.color }}>
            {data.intent.toUpperCase()}
          </span>
          {data.method && (
            <span className="agent-card__tag agent-card__tag--muted">
              {data.method === 'llm' ? 'FAST MODEL' : data.method.replace('_', ' ').toUpperCase()}
            </span>
          )}
          {data.model && (
            <span className="agent-card__model" title={data.model}>
              {shortModelName(data.model)}
            </span>
          )}
        </div>
      )}

      {status === 'complete' && agent === 'prediction' && data.severity && (
        <SeverityPill severity={data.severity} />
      )}

      {isCurrent && status !== 'complete' && (
        <div className="agent-card__progress-bar">
          <div className="agent-card__progress-fill" style={{ background: meta.color }} />
        </div>
      )}
    </div>
  );
}

function shortModelName(model) {
  const name = String(model).split('/').pop() || String(model);
  return name.replace(':free', '');
}

function StatusBadge({ status, isCurrent }) {
  if (status === 'complete') {
    return <span className="status-badge status-badge--done">✓</span>;
  }
  if (status === 'error') {
    return <span className="status-badge status-badge--error">✗</span>;
  }
  if (status === 'skipped') {
    return <span className="status-badge status-badge--skipped">—</span>;
  }
  if (isCurrent) {
    return <span className="status-badge status-badge--running"><span className="pulse-dot" /></span>;
  }
  return <span className="status-badge status-badge--pending">○</span>;
}

function SeverityPill({ severity }) {
  const colors = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#22c55e',
  };
  const color = colors[severity?.toLowerCase()] || '#94a3b8';
  return (
    <div className="agent-card__tag" style={{ borderColor: color, color }}>
      {severity?.toUpperCase()} SEVERITY
    </div>
  );
}
