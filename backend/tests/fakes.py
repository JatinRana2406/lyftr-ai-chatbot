"""A stand-in for `litellm.acompletion` that behaves like LiteLLM with its cache enabled.

It honours the same kwargs the gateway sends (`api_key`, `caching`, `cache={"no-cache": True}`),
caches by the gateway's own cache key, flags hits in `_hidden_params["cache_hit"]`, and can be
told to make a given API key raise an error with a given HTTP status.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.config import Settings
from app.gateway import cache_key_for


class FakeProviderError(Exception):
    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(message)
        self.status_code = status_code


def make_settings(**overrides: Any) -> Settings:
    base = dict(
        api_keys=("sk-test-one", "sk-test-two"),
        key_models=(),
        default_model="gemini/gemini-2.0-flash",
        database_url="sqlite:///:memory:",
        allowed_origins=("http://localhost:3000",),
        enable_failure_simulation=True,
        cache_enabled=True,
        cache_type="local",
        cache_dir="/tmp/unused",
        default_system_prompt="You are a test assistant.",
        request_timeout_s=5.0,
        max_output_tokens=256,
        temperature=0.0,
        mock_llm=False,
    )
    base.update(overrides)
    return Settings(**base)


def _count(text: str) -> int:
    return max(1, len(text.split()))


class FakeCompletion:
    def __init__(self, live_delay_s: float = 0.02) -> None:
        self.cache: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []  # live provider calls only
        self.failing_keys: dict[str, int] = {}  # api_key -> status
        self.empty_reply_keys: set[str] = set()
        self.live_delay_s = live_delay_s

    async def __call__(self, **kwargs: Any) -> Any:
        model, messages = kwargs["model"], kwargs["messages"]
        key = cache_key_for(model, messages)
        cache_ctl = kwargs.get("cache") or {}
        if kwargs.get("caching") and not cache_ctl.get("no-cache") and key in self.cache:
            stored = self.cache[key]
            return SimpleNamespace(
                choices=stored.choices, usage=stored.usage, model=stored.model, _hidden_params={"cache_hit": True}
            )

        api_key = kwargs["api_key"]
        if api_key in self.failing_keys:
            raise FakeProviderError(self.failing_keys[api_key], "Rate limit / quota exceeded")
        await asyncio.sleep(self.live_delay_s)
        self.calls.append(kwargs)

        last = messages[-1]["content"]
        content = "" if api_key in self.empty_reply_keys else f"Echo[{len(messages)}]: {last}"
        prompt_tokens = sum(_count(m["content"]) + 4 for m in messages)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=_count(content)),
            model=model,
            _hidden_params={"cache_hit": None},
        )
        if kwargs.get("caching") and content:
            self.cache[key] = response
        return response
