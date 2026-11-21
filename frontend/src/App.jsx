import { useEffect, useRef, useState } from "react";
import { API_BASE_URL, fetchHealth, newSessionId, sendChat } from "./api.js";

const SIMULATIONS = [
  { value: "", label: "Off — normal request" },
  { value: "key_1=429", label: "key_1 = 429 (rate limit)" },
  { value: "key_1=500", label: "key_1 = 500 (server error)" },
  { value: "key_1=503", label: "key_1 = 503 (unavailable)" },
  { value: "*=429", label: "* = 429 (every key down)" },
];

function MetaItem({ label, testId, children, title }) {
  return (
    <div className="meta-item" title={title}>
      <span className="meta-label">{label}</span>
      <span className="meta-value" data-testid={testId}>
        {children}
      </span>
    </div>
  );
}

function AssistantMessage({ msg }) {
  const m = msg.metadata;
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="row row-assistant">
      <div className="bubble bubble-assistant" data-testid="assistant-message">
        <div className="reply-text">{msg.content}</div>
        <div className="meta">
          <div className={`source-badge ${cached ? "is-cache" : "is-live"}`}>
            <span className="dot" aria-hidden="true" />
            <span data-testid="meta-source">{m.resolution_source}</span>
          </div>
          <MetaItem label="model" testId="meta-model">{m.model}</MetaItem>
          <MetaItem label="latency" testId="meta-latency">{`${m.latency_ms} ms`}</MetaItem>
          <MetaItem label="in" testId="meta-input-tokens" title="Input (prompt) tokens for this request">
            {m.input_tokens}
          </MetaItem>
          <MetaItem label="out" testId="meta-output-tokens" title="Output (completion) tokens for this reply">
            {m.output_tokens}
          </MetaItem>
          <MetaItem label="key" testId="meta-key-id">{m.key_id ?? "—"}</MetaItem>
          <MetaItem label="fallbacks" testId="meta-fallback-attempts">{m.fallback_attempts}</MetaItem>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => newSessionId()); // new session on every page load
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [pending, setPending] = useState(false);
  const [health, setHealth] = useState(null);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  async function send() {
    const text = input.trim();
    if (!text || pending) return;
    setInput(""); // chat-input is empty after sending
    setMessages((prev) => [...prev, { id: crypto.randomUUID?.() ?? Date.now(), role: "user", content: text }]);
    setPending(true);
    const result = await sendChat({ sessionId, message: text, systemPrompt, simulateFailure: simulate });
    setPending(false);
    setMessages((prev) => [
      ...prev,
      result.ok
        ? { id: `${Date.now()}-a`, role: "assistant", content: result.data.reply, metadata: result.data.metadata }
        : { id: `${Date.now()}-e`, role: "error", content: result.error, status: result.status },
    ]);
    inputRef.current?.focus();
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  }

  function startNewChat() {
    setSessionId(newSessionId());
    setMessages([]);
    inputRef.current?.focus();
  }

  const totals = messages.filter((m) => m.role === "assistant");
  const cacheHits = totals.filter((m) => m.metadata.resolution_source === "LLM_CACHE").length;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">◆</div>
          <div>
            <div className="brand-title">Resilient Chat</div>
            <div className="brand-sub">LiteLLM · key failover · cache</div>
          </div>
        </div>

        <section className="panel">
          <div className="panel-head">
            <h2>Session</h2>
            <button type="button" className="ghost" onClick={startNewChat}>
              New chat
            </button>
          </div>
          <code className="session-id" data-testid="session-id">{sessionId}</code>
          <div className="stats">
            <span>{totals.length} replies</span>
            <span>{cacheHits} from cache</span>
          </div>
        </section>

        <section className="panel">
          <h2>
            <label htmlFor="system-prompt">System prompt</label>
          </h2>
          <textarea
            id="system-prompt"
            data-testid="system-prompt-input"
            className="system-prompt"
            placeholder="Optional — e.g. “Answer like a patient teacher.” Applies to this session."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={4}
          />
        </section>

        <section className="panel">
          <h2>
            <label htmlFor="simulate">Failover test</label>
          </h2>
          <select id="simulate" data-testid="simulate-failure" value={simulate} onChange={(e) => setSimulate(e.target.value)}>
            {SIMULATIONS.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
          <p className="hint">
            Sends <code>X-Simulate-Failure</code> with the next messages so you can watch the backend fail over.
          </p>
        </section>

        <section className="panel health">
          <h2>Backend</h2>
          {health === null ? (
            <p className="hint">Checking…</p>
          ) : health.ok ? (
            <p className="hint ok">
              ● Online · {health.data.keys_configured} keys · cache {health.data.cache_enabled ? "on" : "off"}
            </p>
          ) : (
            <p className="hint bad">● Unreachable</p>
          )}
          <p className="hint mono">{API_BASE_URL}</p>
        </section>
      </aside>

      <main className="chat">
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty">
              <h1>Ask anything.</h1>
              <p>
                Every reply shows where it came from — a fresh <b>LIVE_INFERENCE</b> or an <b>LLM_CACHE</b> hit —
                with the model, latency and this request's token usage.
              </p>
            </div>
          )}

          {messages.map((msg) => {
            if (msg.role === "user") {
              return (
                <div key={msg.id} className="row row-user">
                  <div className="bubble bubble-user" data-testid="user-message">
                    {msg.content}
                  </div>
                </div>
              );
            }
            if (msg.role === "error") {
              return (
                <div key={msg.id} className="row row-assistant">
                  <div className="bubble bubble-error" role="alert" data-testid="error-message">
                    <strong>{msg.status ? `Error ${msg.status}` : "Connection error"}</strong>
                    <span>{msg.content}</span>
                  </div>
                </div>
              );
            }
            return <AssistantMessage key={msg.id} msg={msg} />;
          })}

          {pending && (
            <div className="row row-assistant">
              <div className="bubble bubble-pending" data-testid="pending">
                <span className="typing" aria-label="Waiting for reply">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
        >
          {simulate && <div className="sim-flag">Simulating: {simulate}</div>}
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            className="chat-input"
            placeholder="Type a message — Enter to send, Shift+Enter for a new line"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            autoFocus
          />
          {input.trim().length > 4000 && (
            <div className="char-warn">{input.trim().length} / 4000 characters — the server will reject this</div>
          )}
          <button type="submit" data-testid="send-button" className="send" disabled={pending || !input.trim()}>
            {pending ? "Sending…" : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
