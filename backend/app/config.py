"""Application settings, read once from environment variables.

Every setting has a safe default except LLM_API_KEYS, which must hold at
least two comma-separated keys for the failover pool to be considered loaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class KeySlot:
    """One entry of the key pool. `key_id` is public; `api_key` never leaves the server."""

    key_id: str
    api_key: str = field(repr=False)
    model: str


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    key_models: tuple[str, ...]
    default_model: str
    database_url: str
    allowed_origins: tuple[str, ...]
    enable_failure_simulation: bool
    cache_enabled: bool
    cache_type: str
    cache_dir: str
    default_system_prompt: str
    request_timeout_s: float
    max_output_tokens: int
    temperature: float
    mock_llm: bool

    @property
    def key_pool(self) -> list[KeySlot]:
        """key_1, key_2, ... in LLM_API_KEYS order. A key may get its own model via LLM_KEY_MODELS."""
        slots = []
        for index, api_key in enumerate(self.api_keys):
            model = self.key_models[index] if index < len(self.key_models) else self.default_model
            slots.append(KeySlot(key_id=f"key_{index + 1}", api_key=api_key, model=model))
        return slots


def load_settings() -> Settings:
    env = os.environ
    return Settings(
        api_keys=tuple(_split_csv(env.get("LLM_API_KEYS"))),
        key_models=tuple(_split_csv(env.get("LLM_KEY_MODELS"))),
        default_model=env.get("LLM_MODEL", "").strip() or DEFAULT_MODEL,
        database_url=env.get("DATABASE_URL", "").strip() or "sqlite:///./data/app.db",
        allowed_origins=tuple(
            origin.rstrip("/") for origin in _split_csv(env.get("ALLOWED_ORIGINS", "http://localhost:3000"))
        ),
        enable_failure_simulation=_as_bool(env.get("ENABLE_FAILURE_SIMULATION"), False),
        cache_enabled=_as_bool(env.get("CACHE_ENABLED"), True),
        cache_type=(env.get("CACHE_TYPE", "local").strip().lower() or "local"),
        cache_dir=env.get("CACHE_DIR", "").strip() or "./data/llm-cache",
        default_system_prompt=env.get("DEFAULT_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT,
        request_timeout_s=_as_float(env.get("LLM_TIMEOUT_SECONDS"), 30.0),
        max_output_tokens=_as_int(env.get("LLM_MAX_OUTPUT_TOKENS"), 1024),
        temperature=_as_float(env.get("LLM_TEMPERATURE"), 0.3),
        mock_llm=_as_bool(env.get("LLM_MOCK"), False),
    )
