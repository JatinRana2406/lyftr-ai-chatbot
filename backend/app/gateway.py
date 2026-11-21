"""LiteLLM gateway: ordered key pool with automatic failover + LiteLLM response cache.

How a request flows
-------------------
1. Keys are tried in pool order (key_1, key_2, ...). Each attempt is one
   `litellm.acompletion` call with that key's `api_key` and `num_retries=0`,
   so a failing key is abandoned immediately instead of being retried.
2. A key "fails" when LiteLLM raises (it maps provider errors to typed
   exceptions carrying `status_code`: 429 rate limit / quota, 5xx, timeouts,
   connection errors, rejected keys) or returns an empty reply. The next key is
   then tried. If every key fails, `AllProvidersUnavailable` is raised and the
   API answers 503.
3. Caching is LiteLLM's built-in cache (`litellm.cache = Cache(...)`, in-memory
   or disk). Its key is overridden to be exactly `model + full message list`,
   so it does not depend on which API key made the call. LiteLLM marks hits in
   `response._hidden_params["cache_hit"]`; we report LLM_CACHE for those.
4. `X-Simulate-Failure` makes chosen keys behave as if their provider returned
   the given status, for this request only. A simulated request bypasses the
   cache *lookup* so failover is always really exercised (it still stores its
   answer), which is what makes `key_1=429 -> key_id=key_2` deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .config import KeySlot, Settings
from .simulation import simulated_status_for

logger = logging.getLogger("gateway")

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"

CompletionFn = Callable[..., Awaitable[Any]]


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""

    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        super().__init__("all providers unavailable")
        self.attempts = attempts


class SimulatedProviderError(Exception):
    """Raised in place of a provider call when X-Simulate-Failure targets a key."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"simulated provider failure (HTTP {status_code})")
        self.status_code = status_code


class EmptyReplyError(Exception):
    status_code = 502


@dataclass
class LLMResult:
    reply: str
    model: str
    resolution_source: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int
    attempts: list[dict[str, Any]] = field(default_factory=list)


def cache_key_for(model: str, messages: list[dict[str, Any]]) -> str:
    """Cache key = model + the full message list sent to the LLM (system + history + new message)."""
    canonical = json.dumps(
        {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "chat:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def litellm_cache_key(*args: Any, **kwargs: Any) -> str:
    """Drop-in replacement for `litellm.Cache.get_cache_key` (documented custom-key hook)."""
    return cache_key_for(str(kwargs.get("model", "")), list(kwargs.get("messages") or []))


def error_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status of a provider error (LiteLLM exceptions expose `status_code`)."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_failure(exc: BaseException) -> str:
    """Human-readable reason, used for logs and the attempt trail (never contains a key)."""
    status = error_status(exc)
    text = str(exc).lower()
    name = type(exc).__name__
    if status == 429 or "ratelimit" in name.lower() or "rate limit" in text:
        return "rate_limited"
    if "quota" in text or "resource_exhausted" in text or "insufficient_quota" in text:
        return "quota_exhausted"
    if status is not None and status >= 500:
        return "server_error"
    if "timeout" in name.lower() or "timed out" in text:
        return "timeout"
    if "connection" in name.lower():
        return "connection_error"
    if status in (401, 403) or "authentication" in name.lower():
        return "key_rejected"
    if isinstance(exc, EmptyReplyError):
        return "empty_reply"
    return "provider_error"


def _read(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_reply(response: Any) -> str:
    choices = _read(response, "choices") or []
    if not choices:
        return ""
    message = _read(choices[0], "message")
    content = _read(message, "content") or ""
    return content if isinstance(content, str) else str(content)


def _is_cache_hit(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict) and hidden.get("cache_hit"):
        return True
    return bool(getattr(response, "cache_hit", False))


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        completion_fn: CompletionFn | None = None,
        token_counter: Callable[..., int] | None = None,
    ) -> None:
        self.settings = settings
        self.keys: list[KeySlot] = settings.key_pool
        # With an injected completion function (tests), the caller provides the cache behaviour.
        self.cache_enabled = settings.cache_enabled if completion_fn is not None else False
        self._completion = completion_fn
        self._token_counter = token_counter
        if completion_fn is None:
            self._setup_litellm()

    # ---- LiteLLM wiring -----------------------------------------------------

    def _setup_litellm(self) -> None:
        import litellm

        try:
            from litellm.caching.caching import Cache
        except ImportError:  # older LiteLLM layout
            from litellm.caching import Cache  # type: ignore[no-redef]

        litellm.telemetry = False
        litellm.suppress_debug_info = True
        litellm.drop_params = True  # ignore params a provider does not support instead of erroring
        litellm.num_retries = 0  # failover is ours: never retry the same key

        if self.settings.cache_enabled:
            cache = Cache(type="local")
            cache.get_cache_key = litellm_cache_key  # cache key = model + full messages
            litellm.cache = cache
            self.cache_enabled = True
        else:
            litellm.cache = None

        self._completion = litellm.acompletion
        self._token_counter = litellm.token_counter

    # ---- public API ---------------------------------------------------------

    async def complete(
        self, messages: list[dict[str, str]], simulate: dict[str, int] | None = None
    ) -> LLMResult:
        simulate = simulate or {}
        bypass_cache_lookup = bool(simulate)
        attempts: list[dict[str, Any]] = []
        failures = 0

        for slot in self.keys:
            started = time.perf_counter()
            try:
                simulated = simulated_status_for(simulate, slot.key_id)
                if simulated is not None:
                    raise SimulatedProviderError(simulated)
                response = await self._call(slot, messages, bypass_cache_lookup)
                reply = _extract_reply(response).strip()
                if not reply:
                    raise EmptyReplyError("provider returned an empty reply")
            except Exception as exc:  # noqa: BLE001 - any provider failure moves on to the next key
                failures += 1
                attempt = {
                    "key_id": slot.key_id,
                    "model": slot.model,
                    "status": error_status(exc),
                    "reason": classify_failure(exc),
                    "simulated": isinstance(exc, SimulatedProviderError),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
                attempts.append(attempt)
                logger.warning(
                    "key %s failed (%s, status=%s, simulated=%s); %s",
                    slot.key_id,
                    attempt["reason"],
                    attempt["status"],
                    attempt["simulated"],
                    "trying next key" if failures < len(self.keys) else "no keys left",
                )
                continue

            cache_hit = self.cache_enabled and _is_cache_hit(response)
            input_tokens, output_tokens = self._usage(response, slot.model, messages, reply)
            attempts.append({"key_id": slot.key_id, "model": slot.model, "status": 200, "cache_hit": cache_hit})
            return LLMResult(
                reply=reply,
                model=slot.model,  # the LiteLLM model name we routed to, e.g. gemini/gemini-2.0-flash
                resolution_source=LLM_CACHE if cache_hit else LIVE_INFERENCE,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                key_id=None if cache_hit else slot.key_id,
                fallback_attempts=0 if cache_hit else failures,
                attempts=attempts,
            )

        raise AllProvidersUnavailable(attempts)

    # ---- internals ----------------------------------------------------------

    async def _call(self, slot: KeySlot, messages: list[dict[str, str]], bypass_cache_lookup: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": slot.model,
            "messages": messages,
            "api_key": slot.api_key,
            "timeout": self.settings.request_timeout_s,
            "num_retries": 0,
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
        }
        if self.cache_enabled:
            kwargs["caching"] = True
            if bypass_cache_lookup:
                kwargs["cache"] = {"no-cache": True}  # skip lookup, still store the fresh answer
        if self.settings.mock_llm:
            # Local development without real keys: LiteLLM's own mock, still goes through its cache.
            kwargs["mock_response"] = f"(mock reply from {slot.key_id}) You said: {messages[-1]['content']}"
        return await self._completion(**kwargs)

    def _usage(self, response: Any, model: str, messages: list[dict[str, str]], reply: str) -> tuple[int, int]:
        """Tokens of THIS request only, as reported by the provider (or stored with the cached response)."""
        usage = _read(response, "usage")
        prompt = _read(usage, "prompt_tokens")
        completion = _read(usage, "completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int) and (prompt or completion):
            return prompt, completion
        # Provider sent no usage: count locally with LiteLLM's tokenizer helpers.
        if self._token_counter is not None:
            try:
                prompt = int(self._token_counter(model=model, messages=messages))
                completion = int(self._token_counter(model=model, text=reply))
                return prompt, completion
            except Exception:  # noqa: BLE001
                pass
        return int(prompt or 0), int(completion or 0)
