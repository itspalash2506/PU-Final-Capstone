import { useState, useCallback } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'amop-dev-secret';

export function useAgentStream() {
  const [agentSteps, setAgentSteps] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [error, setError] = useState(null);
  const [requestId, setRequestId] = useState(null);

  const reset = useCallback(() => {
    setAgentSteps([]);
    setFinalResult(null);
    setStreamingAnswer('');
    setError(null);
    setRequestId(null);
  }, []);

  const submitQuery = useCallback(async (query, topK = 5, machineId = null) => {
    setIsStreaming(true);
    reset();

    const handleEvent = (event) => {
      switch (event.type) {
        case 'connected':
          setRequestId(event.request_id);
          break;

        case 'agent_complete':
          setAgentSteps(prev => {
            const idx = prev.findIndex(s => s.agent === event.agent);
            if (idx >= 0) {
              const next = [...prev];
              // Preserve streamingText accumulated during token events
              next[idx] = { ...next[idx], ...event, status: 'complete' };
              return next;
            }
            return [...prev, { ...event, status: 'complete' }];
          });
          break;

        case 'token':
          // Stream tokens into the answer area, not the agent card
          setStreamingAnswer(prev => prev + event.token);
          break;

        case 'complete':
          setFinalResult(event);
          setAgentSteps(prev => prev.map(s => ({ ...s, status: 'complete' })));
          break;

        case 'error':
          setError(event.message || 'Unknown pipeline error');
          break;

        default:
          break;
      }
    };

    try {
      const body = { query, top_k: topK };
      if (machineId) body.machine_id = machineId;

      const response = await fetch(`${API_URL}/api/v1/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const json = await response.json();
          detail = json.detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep the possibly-incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            handleEvent(JSON.parse(raw));
          } catch (parseErr) {
            console.warn('SSE parse error:', parseErr, raw);
          }
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsStreaming(false);
    }
  }, [reset]);

  return { agentSteps, isStreaming, finalResult, streamingAnswer, error, requestId, submitQuery, reset };
}
