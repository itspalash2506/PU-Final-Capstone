import React from 'react';

const CONFIDENCE_COLORS = {
  high: '#22c55e',
  medium: '#eab308',
  low: '#f97316',
};

const SECTION_META = {
  root_cause: { title: 'Root Cause', tone: 'primary' },
  contributing_factors: { title: 'Contributing Factors', tone: 'warning' },
  evidence: { title: 'Evidence', tone: 'info' },
  corrective_action: { title: 'Corrective Action', tone: 'action' },
  preventive_action: { title: 'Preventive Action', tone: 'success' },
};

const SECTION_LABELS = {
  'root cause': 'root_cause',
  'contributing factors': 'contributing_factors',
  evidence: 'evidence',
  'corrective action': 'corrective_action',
  'preventive action': 'preventive_action',
  confidence: 'confidence',
};

const SECTION_ORDER = [
  'root_cause',
  'contributing_factors',
  'evidence',
  'corrective_action',
  'preventive_action',
];

export default function RCAPanel({ rca }) {
  if (!rca) return null;

  const {
    probable_cause,
    contributing_factors,
    corrective_action,
    preventive_action,
    confidence,
    analysis,
    sources: rcaSources,
  } = rca;

  const parsedAnalysis = parseRCAAnalysis(analysis);
  const confColor = CONFIDENCE_COLORS[confidence?.toLowerCase()] || '#94a3b8';

  return (
    <div className="rca-panel">
      {confidence && (
        <div className="rca-confidence">
          <span>Confidence:</span>
          <span className="rca-confidence__value" style={{ color: confColor }}>
            {confidence.toUpperCase()}
          </span>
        </div>
      )}

      {probable_cause && (
        <RCASection title="Root Cause" content={probable_cause} tone="primary" />
      )}

      {contributing_factors && (
        <RCASection
          title="Contributing Factors"
          content={contributing_factors}
          tone="warning"
        />
      )}

      {corrective_action && (
        <RCASection
          title="Corrective Action"
          content={corrective_action}
          tone="action"
        />
      )}

      {preventive_action && (
        <RCASection
          title="Preventive Action"
          content={preventive_action}
          tone="success"
        />
      )}

      {!probable_cause && parsedAnalysis && (
        <RCAAnalysisView analysis={parsedAnalysis} sourcesCount={rcaSources?.length} />
      )}

      {!probable_cause && !parsedAnalysis && analysis && (
        <div className="rca-raw">{analysis}</div>
      )}
    </div>
  );
}

export function RCAAnswer({ text, children }) {
  const parsedAnalysis = parseRCAAnalysis(text);

  if (!parsedAnalysis) return children || null;

  return <RCAAnalysisView analysis={parsedAnalysis} />;
}

function RCAAnalysisView({ analysis, sourcesCount }) {
  const visibleSections = SECTION_ORDER
    .map(key => ({ key, meta: SECTION_META[key], section: analysis.sections[key] }))
    .filter(item => item.section?.length);

  return (
    <div className="rca-report">
      <div className="rca-report__header">
        <div>
          <div className="rca-report__eyebrow">Root Cause Analysis</div>
          <h3 className="rca-report__title">
            {typeof sourcesCount === 'number'
              ? `Based on ${sourcesCount} historical records`
              : analysis.subtitle || 'Maintenance finding summary'}
          </h3>
        </div>
        {analysis.confidence && (
          <div className={`rca-confidence-card rca-confidence-card--${analysis.confidence.level}`}>
            <span className="rca-confidence-card__label">Confidence</span>
            <span className="rca-confidence-card__value">{analysis.confidence.label}</span>
          </div>
        )}
      </div>

      <div className="rca-report__grid">
        {visibleSections.map(({ key, meta, section }) => (
          <RCASection
            key={key}
            title={meta.title}
            content={section}
            tone={meta.tone}
          />
        ))}
      </div>

      {analysis.confidence?.detail && (
        <div className="rca-confidence-note">
          {analysis.confidence.detail}
        </div>
      )}
    </div>
  );
}

function RCASection({ title, content, tone = 'default' }) {
  const lines = Array.isArray(content) ? normalizeLines(content) : splitContent(content);
  const listItems = lines.filter(line => line.type === 'bullet');
  const paragraphs = lines.filter(line => line.type !== 'bullet');

  return (
    <div className={`rca-section rca-section--${tone}`}>
      <div className="rca-section__title">
        <span>{title}</span>
      </div>
      <div className="rca-section__body">
        {paragraphs.map((line, index) => (
          <p key={`p-${index}`}>{line.text}</p>
        ))}
        {listItems.length > 0 && (
          <ul className="rca-section__list">
            {listItems.map((line, index) => (
              <li key={`li-${index}`}>{line.text}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function parseRCAAnalysis(text) {
  if (!text || typeof text !== 'string') return null;

  const lines = text.replace(/\r/g, '').split('\n');
  const sections = {};
  let currentKey = null;
  let subtitle = '';

  lines.forEach(rawLine => {
    const line = cleanMarkdown(rawLine).trim();
    if (!line) return;

    if (/^root cause analysis\b/i.test(line)) {
      const match = line.match(/\((.*?)\)/);
      subtitle = match ? match[1] : '';
      return;
    }

    const headingMatch = line.match(
      /^(Root Cause|Contributing Factors|Evidence|Corrective Action|Preventive Action|Confidence)\s*:\s*(.*)$/i
    );

    if (headingMatch) {
      const key = SECTION_LABELS[headingMatch[1].toLowerCase()];
      currentKey = key;
      sections[key] = headingMatch[2] ? [headingMatch[2]] : [];
      return;
    }

    if (currentKey) {
      sections[currentKey].push(line);
    }
  });

  const hasRCASections = SECTION_ORDER.some(key => sections[key]?.length);
  if (!hasRCASections) return null;

  const confidenceText = sections.confidence?.join(' ') || '';
  const confidenceMatch = confidenceText.match(/^(high|medium|low)\b/i);

  return {
    subtitle,
    sections,
    confidence: confidenceText
      ? {
          level: confidenceMatch?.[1]?.toLowerCase() || 'unknown',
          label: confidenceMatch?.[1] || 'Unknown',
          detail: confidenceText,
        }
      : null,
  };
}

function normalizeLines(lines) {
  return lines
    .map(line => cleanMarkdown(line).trim())
    .filter(Boolean)
    .map(line => {
      const bulletMatch = line.match(/^[-*]\s+(.*)$/);
      return {
        type: bulletMatch ? 'bullet' : 'paragraph',
        text: bulletMatch ? bulletMatch[1].trim() : line,
      };
    });
}

function splitContent(content) {
  return normalizeLines(String(content).split('\n'));
}

function cleanMarkdown(value) {
  return String(value)
    .replace(/\*\*/g, '')
    .replace(/\*/g, '')
    .replace(/[ \t]+/g, ' ')
    .trim();
}
