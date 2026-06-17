import React from 'react';

const URGENCY_COLORS = {
  immediate: '#ef4444',
  urgent:    '#f97316',
  scheduled: '#eab308',
  monitor:   '#22c55e',
};

export default function RoutingPanel({ routing }) {
  if (!routing) return null;

  const {
    recommended_technician,
    backup_technician,
    urgency,
    urgency_detail,
    assignment_basis,
    machine_id,
  } = routing;

  const urgencyColor = URGENCY_COLORS[urgency?.toLowerCase()] || '#94a3b8';

  return (
    <div className="routing-panel">
      <div className="routing-urgency" style={{ borderColor: urgencyColor }}>
        <span className="routing-urgency__label">Urgency</span>
        <span className="routing-urgency__value" style={{ color: urgencyColor }}>
          {urgency?.toUpperCase() || 'UNKNOWN'}
        </span>
        {urgency_detail && (
          <span className="routing-urgency__detail">{urgency_detail}</span>
        )}
      </div>

      <div className="routing-techs">
        <div className="routing-tech routing-tech--primary">
          <span className="routing-tech__icon">👷</span>
          <div>
            <div className="routing-tech__role">Recommended Technician</div>
            <div className="routing-tech__name">{recommended_technician || 'Unassigned'}</div>
          </div>
        </div>

        {backup_technician && (
          <div className="routing-tech routing-tech--backup">
            <span className="routing-tech__icon">👤</span>
            <div>
              <div className="routing-tech__role">Backup Technician</div>
              <div className="routing-tech__name">{backup_technician}</div>
            </div>
          </div>
        )}
      </div>

      {assignment_basis && (
        <div className="routing-basis">
          Assignment basis:{' '}
          <span className="routing-basis__value">
            {assignment_basis === 'machine_history' ? 'Machine history' : 'Document similarity'}
          </span>
          {machine_id && ` for machine ${machine_id}`}
        </div>
      )}
    </div>
  );
}
