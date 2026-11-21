// Thin client for the backend contract. Never throws: every outcome becomes { ok, ... }.

const runtime = (typeof window !== "undefined" && window.__APP_CONFIG__) || {};

export const API_BASE_URL = (
  runtime.API_BASE_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");

export function newSessionId() {
  const rand =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : Math.random().toString(16).slice(2, 14);
  return `s-${rand}`;
}

function describeError(status, body) {
  const detail = body && body.detail;
  if (status === 503) {
    return "All providers are unavailable right now — every API key in the pool failed. Please try again in a moment.";
  }
  if (status === 422) {
    if (Array.isArray(detail)) {
      const parts = detail.map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${d.msg}` : d.msg;
      });
      return `Invalid request — ${parts.join("; ")}`;
    }
    return `Invalid request — ${detail || "check your message"}`;
  }
  if (typeof detail === "string") return `${detail} (HTTP ${status})`;
  return `Unexpected server response (HTTP ${status}).`;
}

async function readJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

export async function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const body = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) body.system_prompt = systemPrompt.trim();
  const headers = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;

  let res;
  try {
    res = await fetch(`${API_BASE_URL}/chat`, { method: "POST", headers, body: JSON.stringify(body) });
  } catch {
    return { ok: false, status: 0, error: `Could not reach the server at ${API_BASE_URL}. Check your connection and try again.` };
  }
  const data = await readJson(res);
  if (res.ok && data && typeof data.reply === "string" && data.metadata) {
    return { ok: true, status: res.status, data };
  }
  return { ok: false, status: res.status, error: describeError(res.status, data) };
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    const data = await readJson(res);
    return { ok: res.ok, data };
  } catch {
    return { ok: false, data: null };
  }
}
