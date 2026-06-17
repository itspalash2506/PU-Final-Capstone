import React, { useState } from 'react';
import SourcesList from './SourcesList';
import RoutingPanel from './RoutingPanel';
import RCAPanel, { RCAAnswer } from './RCAPanel';

const TABS = [
  { id: 'answer',  label: 'Answer' },
  { id: 'sources', label: 'Sources' },
  { id: 'rca',     label: 'Root Cause' },
  { id: 'routing', label: 'Routing' },
];

export default function AnswerPanel({ result, streamingText = '', isStreaming = false }) {
  const answer = result?.answer || '';
  const sources = result?.sources || [];
  const routing_result = result?.routing_result;
  const rca_result = result?.rca_result;
  const intent = result?.intent;
  const alpha_machine_id = result?.alpha_machine_id;
  const numeric_machine_id = result?.numeric_machine_id;
  const id_assumption = result?.id_assumption;
  const latency_ms = result?.latency_ms;
  const agent_traces = result?.agent_traces || [];

  const availableTabs = TABS.filter(t => {
    if (t.id === 'rca') return !!rca_result;
    if (t.id === 'routing') return !!routing_result;
    if (t.id === 'sources') return sources.length > 0;
    return true;
  });

  const [activeTab, setActiveTab] = useState('answer');

  const totalAgentMs = agent_traces.reduce((s, t) => s + (t.latency_ms || 0), 0);

  // Streaming mode — show live answer preview before the final result arrives
  if (!result && streamingText) {
    return (
      <section className="answer-section">
        <div className="answer-meta-bar">
          <div className="answer-meta-chips">
            <span className="meta-chip meta-chip--intent" style={{ opacity: 0.5 }}>ANALYZING…</span>
          </div>
        </div>
        <div className="answer-body">
          <div className="answer-text answer-streaming-preview">
            <FormattedAnswer text={streamingText} />
            {isStreaming && <span className="agent-card__cursor">▊</span>}
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="answer-section">
      <div className="answer-meta-bar">
        <div className="answer-meta-chips">
          {intent && (
            <span className="meta-chip meta-chip--intent">{intent.toUpperCase()}</span>
          )}
          {alpha_machine_id && (
            <span className="meta-chip meta-chip--machine">
              Machine {alpha_machine_id}
              {numeric_machine_id && ` (#${numeric_machine_id})`}
              {id_assumption && ' ≈'}
            </span>
          )}
          <span className="meta-chip" title="Full request wall-clock time">
            {latency_ms}ms pipeline total
          </span>
          {totalAgentMs > 0 && (
            <span className="meta-chip" title="Sum of completed agent timings">
              {totalAgentMs}ms agent time
            </span>
          )}
          <span className="meta-chip">{sources.length} sources</span>
        </div>
        <div className="answer-tabs">
          {availableTabs.map(tab => (
            <button
              key={tab.id}
              className={`answer-tab ${activeTab === tab.id ? 'answer-tab--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
              {tab.id === 'sources' && sources.length > 0 && (
                <span className="tab-count">{sources.length}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="answer-body">
        {activeTab === 'answer' && (
          <div className="answer-text">
            {answer ? (
              <RCAAnswer text={answer}>
                <FormattedAnswer text={answer} />
              </RCAAnswer>
            ) : <p className="text-muted">No answer generated.</p>}
          </div>
        )}

        {activeTab === 'sources' && (
          <SourcesList sources={sources} />
        )}

        {activeTab === 'rca' && rca_result && (
          <RCAPanel rca={rca_result} />
        )}

        {activeTab === 'routing' && routing_result && (
          <RoutingPanel routing={routing_result} />
        )}
      </div>
    </section>
  );
}

function FormattedAnswer({ text }) {
  const blocks = buildAnswerBlocks(text);

  return (
    <div className="answer-report">
      {blocks.map((block, index) => {
        if (block.type === 'heading') {
          return (
            <div key={index} className="answer-report__heading">
              {renderInline(block.text)}
            </div>
          );
        }

        if (block.type === 'list') {
          return (
            <ul key={index} className="answer-report__list">
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }

        if (block.type === 'field') {
          return (
            <div key={index} className={`answer-report__field answer-report__field--${block.tone}`}>
              <div className="answer-report__field-label">{block.label}</div>
              <div className="answer-report__field-value">{renderInline(block.value)}</div>
            </div>
          );
        }

        return (
          <p key={index} className="answer-report__paragraph">
            {renderInline(block.text)}
          </p>
        );
      })}
    </div>
  );
}

function buildAnswerBlocks(text) {
  const blocks = [];
  let pendingList = [];

  const flushList = () => {
    if (pendingList.length) {
      blocks.push({ type: 'list', items: pendingList });
      pendingList = [];
    }
  };

  String(text)
    .replace(/\r/g, '')
    .split('\n')
    .forEach(rawLine => {
      const line = rawLine.trim();

      if (!line) {
        flushList();
        return;
      }

      const bulletMatch = line.match(/^[-*]\s+(.*)$/);
      if (bulletMatch) {
        pendingList.push(bulletMatch[1].trim());
        return;
      }

      flushList();

      const headingMatch = line.match(/^#{1,3}\s+(.+)$/)
        || line.match(/^\*\*(.+?)\*\*:?\s*$/);
      if (headingMatch && headingMatch[1].length <= 80) {
        blocks.push({ type: 'heading', text: cleanMarkdown(headingMatch[1]) });
        return;
      }

      const fieldMatch = line.match(/^\*\*(.+?)\*\*:\s*(.+)$/)
        || line.match(/^([A-Z][A-Za-z ]{2,35}):\s+(.+)$/);
      if (fieldMatch) {
        const label = cleanMarkdown(fieldMatch[1]);
        const value = cleanMarkdown(fieldMatch[2]);
        blocks.push({
          type: 'field',
          label,
          value,
          tone: fieldTone(label),
        });
        return;
      }

      blocks.push({ type: 'paragraph', text: cleanMarkdown(line) });
    });

  flushList();
  return blocks;
}

function fieldTone(label) {
  const normalized = label.toLowerCase();
  if (normalized.includes('recommended')) return 'primary';
  if (normalized.includes('backup')) return 'secondary';
  if (normalized.includes('urgent') || normalized.includes('risk')) return 'warning';
  if (normalized.includes('confidence')) return 'success';
  return 'default';
}

function renderInline(text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);

  return parts.map((part, index) => {
    const boldMatch = part.match(/^\*\*([^*]+)\*\*$/);
    if (boldMatch) {
      return <strong key={index}>{boldMatch[1]}</strong>;
    }
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

function cleanMarkdown(value) {
  return String(value)
    .replace(/^#+\s*/, '')
    .replace(/\*\*/g, '')
    .trim();
}
