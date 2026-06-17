import React, { useMemo } from 'react';
import AgentCard from './AgentCard';

// Every possible agent node in graph order
const FULL_PIPELINE = [
  'equipment',
  'router',
  'rag',
  'rca',
  'prediction',
  'routing',
  'summarizer',
];

const AGENT_LABELS = {
  equipment: 'Equipment Agent',
  router: 'Intent Router',
  rag: 'RAG Agent',
  rca: 'RCA Agent',
  prediction: 'Prediction Agent',
  routing: 'Routing Agent',
  summarizer: 'Summarizer',
};

// Which nodes are active for each intent
const ACTIVE_FOR_INTENT = {
  rag:     new Set(['equipment', 'router', 'rag', 'routing', 'summarizer']),
  predict: new Set(['equipment', 'router', 'prediction', 'summarizer']),
  both:    new Set(['equipment', 'router', 'rag', 'rca', 'prediction', 'routing', 'summarizer']),
  rca:     new Set(['equipment', 'router', 'rag', 'rca', 'summarizer']),
};

export default function AgentPipeline({ steps, isStreaming }) {
  const detectedIntent = useMemo(() => {
    const routerStep = steps.find(s => s.agent === 'router');
    return routerStep?.data?.intent || null;
  }, [steps]);

  const activeSet = detectedIntent ? ACTIVE_FOR_INTENT[detectedIntent] : null;

  const displaySteps = useMemo(() => {
    const completedMap = Object.fromEntries(steps.map(s => [s.agent, s]));
    const completedAgents = steps.map(s => s.agent);
    const lastCompleted = completedAgents[completedAgents.length - 1];

    // Find which node in FULL_PIPELINE is "currently running"
    // It's the next active node after the last completed one
    let currentAgent = null;
    if (isStreaming) {
      const lastCompletedIdx = FULL_PIPELINE.indexOf(lastCompleted);
      // Walk forward until we find an active node
      for (let i = lastCompletedIdx + 1; i < FULL_PIPELINE.length; i++) {
        const candidate = FULL_PIPELINE[i];
        if (!activeSet || activeSet.has(candidate)) {
          currentAgent = candidate;
          break;
        }
      }
    }

    return FULL_PIPELINE.map(agentName => {
      // Already completed
      if (completedMap[agentName]) {
        return { ...completedMap[agentName], _isCurrent: false, _skipped: false };
      }

      // Intent known and this agent is not on the path
      const isSkipped = activeSet && !activeSet.has(agentName);
      if (isSkipped) {
        return {
          agent: agentName,
          label: AGENT_LABELS[agentName] || agentName,
          status: 'skipped',
          latency_ms: null,
          summary: 'Not required for this query',
          data: {},
          _isCurrent: false,
          _skipped: true,
        };
      }

      // Currently running
      if (agentName === currentAgent) {
        return {
          agent: agentName,
          label: AGENT_LABELS[agentName] || agentName,
          status: 'running',
          latency_ms: null,
          summary: '',
          data: {},
          _isCurrent: true,
          _skipped: false,
        };
      }

      // Waiting to run
      return {
        agent: agentName,
        label: AGENT_LABELS[agentName] || agentName,
        status: 'pending',
        latency_ms: null,
        summary: '',
        data: {},
        _isCurrent: false,
        _skipped: false,
      };
    });
  }, [steps, isStreaming, activeSet]);

  const completedCount = steps.length;
  const activeCount = activeSet ? activeSet.size : FULL_PIPELINE.length;

  return (
    <section className="pipeline-section">
      <div className="section-header">
        <span className="section-title">Agent Pipeline</span>
        <span className="pipeline-counter">
          {completedCount}/{activeCount} agents
        </span>
        {isStreaming && <span className="pipeline-status-badge">Processing</span>}
        {!isStreaming && steps.length > 0 && (
          <span className="pipeline-status-badge pipeline-status-badge--done">Complete</span>
        )}
      </div>

      <div className="pipeline-track">
        {displaySteps.map((step, i) => (
          <React.Fragment key={step.agent}>
            <AgentCard step={step} isCurrent={step._isCurrent} />
            {i < displaySteps.length - 1 && (
              <div
                className={`pipeline-arrow
                  ${step.status === 'complete' && !displaySteps[i + 1]._skipped ? 'pipeline-arrow--active' : ''}
                  ${step._skipped || displaySteps[i + 1]._skipped ? 'pipeline-arrow--skipped' : ''}`
                }
              >
                →
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {detectedIntent && (
        <div className="pipeline-intent-legend">
          Intent path:&nbsp;
          <span className="pipeline-intent-name">{detectedIntent.toUpperCase()}</span>
          &nbsp;—&nbsp;
          {FULL_PIPELINE.filter(a => activeSet && !activeSet.has(a)).length > 0
            ? `${FULL_PIPELINE.filter(a => activeSet && !activeSet.has(a)).length} agent(s) not required`
            : 'all agents active'}
        </div>
      )}
    </section>
  );
}
