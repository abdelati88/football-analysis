/* Talking to the server.
 *
 * Every path is relative, so the page works the same served from localhost
 * during development and from a domain in production. The previous version
 * hard-coded an absolute host, which meant a local copy silently wrote its
 * matches to the live server.
 */

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
    ...options,
  });

  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  const payload = isJson ? await response.json().catch(() => null) : null;

  if (!response.ok) {
    // The server explains itself in `error`; fall back to the status only when
    // it has not, so the user never sees a bare number.
    const failure = new Error(payload?.error || `Request failed (${response.status})`);
    // Some refusals are not dead ends. A 409 on starting an analysis carries
    // the job already running, which is exactly what the caller should attach
    // to instead of giving up — so the rest of the response travels with the
    // message rather than being thrown away with it.
    failure.status = response.status;
    failure.payload = payload;
    throw failure;
  }
  return payload;
}

export const api = {
  listMatches: () => request('/api/matches'),
  getMatch: (id) => request(`/api/matches/${id}`),
  saveMatch: (body) => request('/api/matches', { method: 'POST', body: JSON.stringify(body) }),
  updateMatch: (id, body) => request(`/api/matches/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteMatch: (id) => request(`/api/matches/${id}`, { method: 'DELETE' }),
  csvUrl: (id) => `/api/matches/${id}/export.csv`,
  jsonUrl: (id) => `/api/matches/${id}/export.json`,

  engineStatus: () => request('/api/analysis/status'),
  listJobs: () => request('/api/analysis'),
  startAnalysis: (formData) => request('/api/analysis', { method: 'POST', body: formData }),
  jobStatus: (id, withResult = false) =>
    request(`/api/analysis/${id}${withResult ? '?include_result=1' : ''}`),
  cancelJob: (id) => request(`/api/analysis/${id}/cancel`, { method: 'POST' }),
  deleteJob: (id) => request(`/api/analysis/${id}`, { method: 'DELETE' }),
  saveJobAsMatch: (id, body) =>
    request(`/api/analysis/${id}/save`, { method: 'POST', body: JSON.stringify(body) }),
  jobVideoUrl: (id) => `/api/analysis/${id}/video`,
};
