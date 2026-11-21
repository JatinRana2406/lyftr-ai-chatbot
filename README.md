# Resilient LLM Chatbot — key failover + response caching

A full-stack chatbot: **React (Vite)** frontend, **FastAPI + LiteLLM** backend, **SQLite** session storage.
Each reply shows where it came from (`LIVE_INFERENCE` or `LLM_CACHE`), the model, latency and that
request's input/output tokens. When an API key is rate-limited, out of quota or the provider returns 5xx,
the backend fails over to the next key automatically; when every key fails the user gets a clean
`503 {"detail": "all providers unavailable"}` shown as a readable message in the chat.

```
/backend            FastAPI app (app/), tests (tests/), multi-stage Dockerfile
/frontend           React app (src/), multi-stage Dockerfile (nginx runtime), vercel.json
docker-compose.yml  backend :8000 + frontend :3000 + named volume at /data
Makefile            make up / make down / make test / make logs / make clean
.env.example        every variable, placeholder values only
render.yaml         Render blueprint for the backend
```

## 1. Live URLs

| Tier | URL |
| --- | --- |
| Frontend | `https://lyftr-ai-chatbot-hagz.vercel.app/` |
| Backend | `https://lyftr-ai-chatbot-e33z.onrender.com` (try `/health`) |
| Repository | `https://github.com/JatinRana2406/lyftr-ai-chatbot` |
| Demo video | `https://drive.google.com/file/d/1_6Rg2Lx6eLOnXc4jiknm--_ek__xsJW_/view?usp=drive_link` |

**Deploying (≈10 minutes)**

1. **Backend on Render** — New → Blueprint → pick this repo (uses `render.yaml`), or New → Web Service →
   Docker, root directory `backend`. Set `LLM_API_KEYS` (two keys, comma-separated), `LLM_MODEL`,
   `ALLOWED_ORIGINS` (your Vercel URL, no trailing slash) and `ENABLE_FAILURE_SIMULATION=true`.
   Check `https://<backend>/health`.
2. **Frontend on Vercel** — New Project → this repo → root directory `frontend` (framework: Vite).
   Add env var `VITE_API_BASE_URL=https://<backend>` and deploy.
3. Put the Vercel URL into the backend's `ALLOWED_ORIGINS` and redeploy the backend.

> Render's free tier sleeps after inactivity (the first request can take ~50 s) and its filesystem is
> wiped on redeploy. History survives container restarts locally via the Docker volume; on Render, attach
> a disk at `/data` (paid) for the same guarantee.

## 2. Run locally: with Docker and without

**With Docker** (backend on http://localhost:8000, frontend on http://localhost:3000)

```bash
cp .env.example .env        # then put two real keys in LLM_API_KEYS
make up                     # docker compose up -d --build
make logs                   # follow logs; failover attempts are logged per key
make down                   # stop (the SQLite volume `chat-data` is kept)
make test                   # backend test suite in a throwaway python:3.12 container
make clean                  # stop and delete the volume
```

No keys yet? Set `LLM_MOCK=true` in `.env`: LiteLLM returns mock replies, while caching, failover,
SQLite history and the UI all behave exactly as with a real provider.

**Without Docker**

```bash
# backend (Python 3.11+)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEYS="key-one,key-two" ENABLE_FAILURE_SIMULATION=true \
       DATABASE_URL=sqlite:///./data/app.db ALLOWED_ORIGINS=http://localhost:3000
uvicorn app.main:app --reload --port 8000
python -m pytest -q          # tests

# frontend (Node 18+), in another terminal
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev    # http://localhost:3000
```

**Getting two keys:** two free Google AI Studio keys (from two Google accounts/projects) with
`LLM_MODEL=gemini/gemini-2.0-flash`, or mix providers with `LLM_KEY_MODELS`, e.g.
`LLM_API_KEYS=<gemini-key>,<groq-key>` and `LLM_KEY_MODELS=gemini/gemini-2.0-flash,groq/llama-3.1-8b-instant`.
Any LiteLLM model string works; use one your key has access to.

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_API_KEYS` | — (required) | Comma-separated key pool, ≥2 keys. Ids are `key_1`, `key_2`, … in this order |
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | LiteLLM model for every key |
| `LLM_KEY_MODELS` | empty | Optional per-key models, same order as the keys |
| `DATABASE_URL` | `sqlite:////data/app.db` (Docker) | SQLite location |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `ENABLE_FAILURE_SIMULATION` | `false` (compose: `true`) | Enables `X-Simulate-Failure` |
| `CACHE_ENABLED` | `true` | LiteLLM response cache on/off |
| `CACHE_TYPE` / `CACHE_DIR` | `local` (compose: `disk`) / `/data/llm-cache` | In-memory or disk cache |
| `DEFAULT_SYSTEM_PROMPT` | "You are a helpful, concise assistant." | Used when a session sets none |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-key timeout before failing over |
| `LLM_MAX_OUTPUT_TOKENS`, `LLM_TEMPERATURE` | `1024`, `0.3` | Generation settings |
| `LLM_MOCK` | `false` | LiteLLM mock replies, local development only |
| `API_BASE_URL` (frontend container) | `http://localhost:8000` | Backend URL written into `config.js` at start |
| `VITE_API_BASE_URL` (frontend build) | `http://localhost:8000` | Build-time backend URL (Vercel) |

## 3. Design decisions: key fallback, caching, SQLite session storage

### API contract

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | `200 {"status":"ok","keys_configured":2,"cache_enabled":true}` once the DB answers `SELECT 1` and the key pool is loaded; otherwise `503` with a `detail` |
| `POST /chat` | `{session_id, message, system_prompt?}` → `{session_id, reply, metadata{resolution_source, model, latency_ms, input_tokens, output_tokens, key_id, fallback_attempts}}` |
| `GET /sessions/{id}/messages` | History oldest-first without the system prompt; `metadata` on assistant messages only; `404 {"detail":"session not found"}` |

Every error is JSON with `detail`: request validation → 422 (custom handler that keeps FastAPI's shape
but strips non-serialisable internals), all keys down → 503, unknown session/route → 404, and a
catch-all handler turns any unexpected bug into `500 {"detail":"internal server error"}` — never a
stack trace or HTML page. Validation is done by Pydantic: `session_id` 1–128 chars, `message`
1–4000 chars **after trimming**, optional `system_prompt`.

### Key fallback (`backend/app/gateway.py`)

- The pool is built from `LLM_API_KEYS`; key *n* gets id `key_n`. Only ids ever leave the server (logs,
  responses, attempt trail) — the key strings never do.
- For each request the gateway tries keys **in order**, one `litellm.acompletion(...)` call per key with
  that key's `api_key` and `num_retries=0`, so a bad key is abandoned immediately instead of retried.
- A key counts as failed when LiteLLM raises (LiteLLM maps provider errors into typed exceptions with a
  `status_code`: 429 rate limit, quota/`RESOURCE_EXHAUSTED`, 5xx, timeouts, connection errors, a
  rejected key) **or** returns an empty reply. The gateway logs the reason and moves on; when no key is
  left it raises `AllProvidersUnavailable` → `503 {"detail":"all providers unavailable"}`.
- `fallback_attempts` = number of keys that failed before the one that answered.
- **Why an explicit ordered loop instead of LiteLLM's `Router` fallbacks?** The contract needs the exact
  `key_id` that answered, an exact `fallback_attempts` count, and failure injection that applies to
  *one request only*. Router keeps cross-request cooldown state (a simulated 429 would bench `key_1`
  for later, unrelated requests) and doesn't expose the answering key reliably. The loop is ~30 lines,
  deterministic and unit-tested, while LiteLLM still does the provider calls, error normalisation and
  caching.
- **`X-Simulate-Failure`** (`backend/app/simulation.py`): parsed only when
  `ENABLE_FAILURE_SIMULATION=true`. Format `key_1=429,key_2=500` or `*=429`; statuses 429/500/502/503;
  a malformed header is a 422. A targeted key raises a simulated provider error *in place of its call*,
  so it flows through exactly the same failover path as a real error. Simulated requests skip the
  cache **lookup** (they still store their answer) so `key_1=429` always demonstrates a real live
  answer from `key_2` rather than a cache hit.

### Caching

- LiteLLM's built-in cache: `litellm.cache = Cache(type="local")` (in-memory) or `Cache(type="disk")`
  (persists under `/data/llm-cache`, which is on the Docker volume), selected by `CACHE_TYPE`.
- **Cache key = model + the full message list** (system prompt + history + new message), via LiteLLM's
  documented custom-key hook (`cache.get_cache_key = ...`, SHA-256 of a canonical JSON). It deliberately
  ignores the API key, so a reply cached via `key_2` is reused even when `key_1` is tried first.
  Consequences: the same first message in a new session (same default system prompt) → `LLM_CACHE`;
  a different message, a different system prompt or the same text later in a conversation (different
  history) → never from cache.
- Hits are detected from LiteLLM's `response._hidden_params["cache_hit"]`. On a hit the reply text and
  token usage are the ones stored with the cached response, `key_id` is `null`, `fallback_attempts` is 0,
  and latency is a few ms (no network).

### SQLite session storage (`backend/app/db.py`)

- Standard-library `sqlite3`, file at `DATABASE_URL` (`sqlite:////data/app.db` on the named volume, so
  history survives `make down && make up`). WAL mode; schema created with `CREATE TABLE IF NOT EXISTS`
  at startup.
- Tables: `messages` exactly as specified (plus a `CHECK` on `role` and an index on
  `(session_id, id)`), and `sessions(session_id, system_prompt, created_at, updated_at)` which stores a
  session's system-prompt override, so the client only ever sends the new message.
- A turn is written **after** the LLM succeeds, in one transaction (session upsert + user row + assistant
  row with its metadata JSON). A 422 or 503 therefore never leaves anything in history — not even an empty
  session, which is why `GET /sessions/{id}/messages` 404s after a failed first request.
- Multi-turn: on each `/chat` the backend loads the session's history from SQLite and sends
  `[system] + history + [new user message]` to the model.

### Tokens and latency

- `input_tokens`/`output_tokens` come from the provider's `usage` for that single call (prompt =
  system + history + new message; completion = this reply). If a provider omits usage, LiteLLM's
  `token_counter` is used. They are never summed across the conversation.
- `latency_ms` is measured on the server with `time.perf_counter()` from the start of the request handler
  to the moment the reply is ready, as an integer.

### Frontend

- A new `session_id` (`s-` + random hex) is generated on every page load; "New chat" starts another.
- Required `data-testid`s: `chat-input`, `send-button`, `assistant-message`, `meta-source`, `meta-model`,
  `meta-latency` (e.g. `812 ms`), `meta-input-tokens`, `meta-output-tokens`, `system-prompt-input`.
  The input clears on send; Enter sends, Shift+Enter adds a line.
- Errors (422, 503, network) render as a readable red message in the chat (`data-testid="error-message"`),
  never a blank screen.
- The backend URL is read at runtime from `config.js` (written from `API_BASE_URL` by the container
  entrypoint), falling back to the build-time `VITE_API_BASE_URL` — one image works for any backend.

### Docker

- Backend: `python:3.12-slim` build stage creates a virtualenv; runtime stage copies only the venv and
  `app/`, runs as a non-root user, declares `/data` as a volume, honours `$PORT`, has a `/health`
  healthcheck.
- Frontend: `node:22-alpine` build stage runs `vite build`; runtime is `nginx:1.27-alpine` serving the
  static files on port 3000 (no Node in the final image).

## 4. Verification: how to see failover, cache hits and per-message token usage

Set `B=http://localhost:8000` (or the deployed backend URL).

```bash
# health
curl -s $B/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}

# live call, then the same first message in a NEW session -> cache hit
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-a1","message":"What is the capital of France?"}'
# ... "resolution_source":"LIVE_INFERENCE", "key_id":"key_1", "latency_ms":812 ...
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-b2","message":"What is the capital of France?"}'
# ... same reply, "resolution_source":"LLM_CACHE", "key_id":null, "fallback_attempts":0, latency a few ms ...

# multi-turn memory + history from SQLite
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-c3","message":"My name is Asha."}'
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-c3","message":"What is my name?"}'
curl -s $B/sessions/s-c3/messages
curl -s $B/sessions/nope/messages        # 404 {"detail":"session not found"}

# failover on demand
curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-d4","message":"Name a planet."}'
# 200, "key_id":"key_2", "fallback_attempts":1
curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=500' \
  -d '{"session_id":"s-d4","message":"Name another."}'
# 200, "key_id":"key_2"
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"s-e5","message":"Anyone there?"}'
# HTTP 503 {"detail":"all providers unavailable"}; s-e5 has no history (404)

# validation
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-f6","message":"   "}'   # 422
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-f6",'                 # 422 malformed JSON
```

**In the UI:** each assistant bubble shows a `LIVE_INFERENCE`/`LLM_CACHE` badge, model, latency, `in`/`out`
tokens for that request, `key` and `fallbacks`. Reload the page (new session) and send the same first
message to see `LLM_CACHE`. Use the **Failover test** dropdown in the sidebar to send
`X-Simulate-Failure` (`key_1=429` → answered by `key_2` with `fallbacks 1`; `*=429` → red 503 message).
Backend logs show every failed attempt: `key key_1 failed (rate_limited, status=429, simulated=True); trying next key`.

**Tests** (`make test` or `cd backend && python -m pytest -q`):

- `tests/test_api.py` — the HTTP contract end to end with a fake provider: response shape, cache hit/miss
  rules (incl. history and system prompt in the key), per-request tokens, multi-turn + isolated
  sessions, history endpoint + 404, persistence across app restarts, every 422 case, failover for
  429/500/502/503 (simulated and real provider errors), 503 with nothing stored, simulation switch off,
  JSON-only 500, CORS.
- `tests/test_gateway.py` — failover order across 3 keys, attempt trail never contains keys, per-key
  models, cache-key rules, header parser, SQLite schema and round trip.
- `tests/test_litellm_integration.py` — runs against the real LiteLLM library (mock responses, no
  network) to prove the custom cache key, hit detection and identical reply/usage on a hit.

## 5. Setup Used

- Editor: `<your editor, e.g. VS Code>`
- AI tools: Claude (Anthropic) for planning, code generation, tests and documentation; all code reviewed
  and run locally before submission.

## 6. Bonus features (if any)

- **Failover test panel in the UI** — sends `X-Simulate-Failure` from the browser, so failover can be
  shown in the demo without curl.
- **`key_id` and `fallback_attempts` shown on every reply**, plus a live backend-health indicator.
- **Per-key models (`LLM_KEY_MODELS`)** — the pool can span providers (e.g. Gemini + Groq), so a full
  provider outage is survivable, not just a single key's quota.
- **Session system prompt** (`system-prompt-input`) stored per session in SQLite.
- **Disk cache option** on the Docker volume, so cached answers survive restarts.
- **Empty replies count as failures** and trigger failover, so `reply` is never empty.
- **`LLM_MOCK=true`** — run the full stack without real keys for local development.
- **Runtime-configurable frontend image** — the backend URL comes from an env var at container start.
