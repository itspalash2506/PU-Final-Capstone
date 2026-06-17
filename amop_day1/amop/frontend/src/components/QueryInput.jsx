import React, { useState } from 'react';

const EXAMPLE_QUERIES = [
  'What issues has machine A6 had with hydraulic pressure?',
  'Predict failure severity for machine B12 — grinding noise in spindle',
  'Root cause analysis for repeated bearing failures on A23',
  'What are the most common issues on machine C7?',
];

export default function QueryInput({ onSubmit, isLoading }) {
  const [query, setQuery] = useState('');
  const [machineId, setMachineId] = useState('');
  const [topK, setTopK] = useState(5);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed, topK, machineId.trim() || null);
  };

  const applyExample = (q) => {
    setQuery(q);
  };

  return (
    <div className="query-input-wrapper">
      <form className="query-form" onSubmit={handleSubmit}>
        <div className="query-main-row">
          <div className="query-textarea-wrap">
            <textarea
              className="query-textarea"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Describe the maintenance issue, ask about a machine, or request a failure prediction…"
              rows={3}
              maxLength={500}
              disabled={isLoading}
            />
            <span className="char-count">{query.length}/500</span>
          </div>
          <button
            type="submit"
            className={`submit-btn ${isLoading ? 'loading' : ''}`}
            disabled={!query.trim() || isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner" />
                Running
              </>
            ) : (
              <>
                <span className="submit-icon">▶</span>
                Run
              </>
            )}
          </button>
        </div>

        <div className="query-footer">
          <button
            type="button"
            className="advanced-toggle"
            onClick={() => setShowAdvanced(v => !v)}
          >
            {showAdvanced ? '▲' : '▼'} Options
          </button>

          {showAdvanced && (
            <div className="advanced-fields">
              <label className="field-label">
                Machine ID
                <input
                  className="field-input"
                  value={machineId}
                  onChange={e => setMachineId(e.target.value)}
                  placeholder="e.g. A6 (auto-detected if blank)"
                  disabled={isLoading}
                />
              </label>
              <label className="field-label">
                Top K results
                <input
                  className="field-input narrow"
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={e => setTopK(Number(e.target.value))}
                  disabled={isLoading}
                />
              </label>
            </div>
          )}

          <div className="example-queries">
            <span className="examples-label">Try:</span>
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                type="button"
                className="example-chip"
                onClick={() => applyExample(q)}
                disabled={isLoading}
              >
                {q.length > 50 ? q.slice(0, 50) + '…' : q}
              </button>
            ))}
          </div>
        </div>
      </form>
    </div>
  );
}
