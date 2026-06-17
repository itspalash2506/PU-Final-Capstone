const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'amop-dev-secret';

export async function fetchPdmPrediction(machineId) {
  const res = await fetch(`${API_URL}/api/v1/pdm/predict/${machineId}`, {
    headers: { 'X-API-Key': API_KEY },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
