import React, { useState } from 'react';
import { fetchPdmPrediction } from '../api/pdm';

const RISK_COLORS = {
  Low:      '#22c55e',
  Medium:   '#eab308',
  High:     '#f97316',
  Critical: '#ef4444',
};

function formatFeatureName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export default function PdmPanel() {
  const [machineId, setMachineId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const id = parseInt(machineId, 10);
    if (isNaN(id) || id < 1 || id > 100) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchPdmPrediction(id);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskColor = result ? (RISK_COLORS[result.risk_tier] || '#94a3b8') : null;
  const probPercent = result ? Math.round(result.failure_probability * 100) : 0;
  const maxShap = result
    ? Math.max(...result.top_features.map(f => Math.abs(f.shap_value)), 0.0001)
    : 1;

  return (
    <section className="pdm-section">
      <div className="section-header">
        <span className="section-title">PdM Sensor Prediction</span>
        <span className="pdm-hint-label">numeric IDs 1–100</span>
      </div>

      <form className="pdm-input-row" onSubmit={handleSubmit}>
        <input
          className="pdm-machine-input"
          type="number"
          min={1}
          max={100}
          value={machineId}
          onChange={e => setMachineId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit(e)}
          placeholder="Machine ID (1 – 100)"
          disabled={loading}
        />
        <button
          type="submit"
          className={`pdm-predict-btn ${loading ? 'loading' : ''}`}
          disabled={!machineId || loading}
        >
          {loading ? (
            <><span className="spinner" /> Fetching…</>
          ) : (
            '⚡ Predict'
          )}
        </button>
      </form>

      {error && (
        <div className="error-banner pdm-error">
          <span className="error-banner__icon">✗</span>
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="pdm-result">
          <div className="pdm-gauge-wrap">
            <div className="pdm-gauge-header">
              <span className="pdm-gauge-label">
                Failure Probability — next {result.horizon}
              </span>
              <span className="pdm-gauge-pct" style={{ color: riskColor }}>
                {probPercent}%
              </span>
            </div>
            <div className="pdm-gauge-track">
              <div
                className="pdm-gauge-fill"
                style={{ width: `${probPercent}%`, background: riskColor }}
              />
            </div>
          </div>

          <div className="pdm-meta-row">
            <div
              className="pdm-risk-badge"
              style={{ borderColor: riskColor, color: riskColor }}
            >
              {result.risk_tier.toUpperCase()} RISK
            </div>
            <div className="pdm-machine-tag">Machine #{result.pdm_machine_id}</div>
            <div className="pdm-basis-tag">{result.match_basis}</div>
          </div>

          {result.top_features.length > 0 && (
            <div className="pdm-features">
              <div className="pdm-features-title">Top Driving Factors (SHAP)</div>
              {result.top_features.map((f, i) => {
                const barPct = Math.round((Math.abs(f.shap_value) / maxShap) * 100);
                const isRisk = f.direction === 'increases_risk';
                const barColor = isRisk ? '#ef4444' : '#22c55e';
                return (
                  <div key={i} className="pdm-feature-row">
                    <span
                      className="pdm-feature-dir"
                      style={{ color: barColor }}
                    >
                      {isRisk ? '↑' : '↓'}
                    </span>
                    <span className="pdm-feature-name">
                      {formatFeatureName(f.name)}
                    </span>
                    <div className="pdm-feature-bar-wrap">
                      <div
                        className="pdm-feature-bar"
                        style={{ width: `${barPct}%`, background: barColor }}
                      />
                    </div>
                    <span className="pdm-feature-val">
                      {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
