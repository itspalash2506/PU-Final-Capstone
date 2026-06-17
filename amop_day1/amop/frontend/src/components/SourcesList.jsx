import React, { useState } from 'react';

export default function SourcesList({ sources }) {
  const [expanded, setExpanded] = useState(null);

  if (!sources || sources.length === 0) {
    return <p className="text-muted">No sources retrieved.</p>;
  }

  return (
    <div className="sources-list">
      {sources.map((src, i) => {
        const isOpen = expanded === i;
        const scorePercent = Math.round((src.score || 0) * 100);
        return (
          <div key={src.id || i} className="source-card">
            <div
              className="source-card__header"
              onClick={() => setExpanded(isOpen ? null : i)}
            >
              <div className="source-card__left">
                <span className="source-index">#{i + 1}</span>
                <div className="source-card__info">
                  <span className="source-machine">Machine {src.machine_id}</span>
                  <span className="source-issue">{src.issue_description}</span>
                </div>
              </div>
              <div className="source-card__right">
                <div className="relevance-bar">
                  <div
                    className="relevance-fill"
                    style={{ width: `${scorePercent}%` }}
                  />
                </div>
                <span className="relevance-label">{scorePercent}%</span>
                <span className="expand-icon">{isOpen ? '▲' : '▼'}</span>
              </div>
            </div>

            {isOpen && (
              <div className="source-card__body">
                {src.technician_notes ? (
                  <>
                    <div className="source-section-label">Technician Notes</div>
                    <div className="source-notes">{src.technician_notes}</div>
                  </>
                ) : (
                  <span className="text-muted">No technician notes on record.</span>
                )}
                <div className="source-id">ID: {src.id}</div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
