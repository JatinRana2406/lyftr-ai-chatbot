"""Unit tests for the failover gateway, the simulation parser and SQLite storage.

These need only the standard library + pytest (no FastAPI, no LiteLLM).
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import Database, sqlite_path_from_url
from app.gateway import (
    LIVE_INFERENCE,
    LLM_CACHE,
    AllProvidersUnavailable,
    LLMGateway,
    cache_key_for,
    classify_failure,
    litellm_cache_key,
)
from app.simulation import InvalidSimulationHeader, parse_simulation_header, simulated_status_for
from tests.fakes import FakeCompletion, FakeProviderError, make_settings

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def run(coro):
    return asyncio.run(coro)


# ---- simulation header ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("key_1=429", {"key_1": 429}),
        ("key_1=429,key_2=500", {"key_1": 429, "key_2": 500}),
        ("*=429", {"*": 429}),
        (" key_2 = 502 , key_1=503 ", {"key_2": 502, "key_1": 503}),
        ("KEY_1=429", {"key_1": 429}),
        ("", {}),
        (None, {}),
    ],
)
def test_parse_simulation_header(raw, expected):
    assert parse_simulation_header(raw) == expected


@pytest.mark.parametrize("raw", ["key_1", "key_1=200", "key_0=429", "foo=429", "key_1=x", "=429"])
def test_parse_simulation_header_rejects_bad_input(raw):
    with pytest.raises(InvalidSimulationHeader):
        parse_simulation_header(raw)


def test_explicit_key_wins_over_wildcard():
    plan = {"*": 429, "key_2": 500}
    assert simulated_status_for(plan, "key_1") == 429
    assert simulated_status_for(plan, "key_2") == 500
    assert simulated_status_for({}, "key_1") is None


# ---- gateway failover ----------------------------------------------------------------


def gateway(fake, **overrides):
    return LLMGateway(make_settings(**overrides), completion_fn=fake)


def test_first_key_answers():
    fake = FakeCompletion(live_delay_s=0)
    result = run(gateway(fake).complete(MSGS))
    assert (result.key_id, result.fallback_attempts, result.resolution_source) == ("key_1", 0, LIVE_INFERENCE)
    assert fake.calls[0]["api_key"] == "sk-test-one"
    assert fake.calls[0]["num_retries"] == 0


def test_cache_hit_reports_cache_and_null_key():
    fake = FakeCompletion(live_delay_s=0)
    gw = gateway(fake)
    live = run(gw.complete(MSGS))
    hit = run(gw.complete(MSGS))
    assert hit.resolution_source == LLM_CACHE
    assert hit.key_id is None and hit.fallback_attempts == 0
    assert hit.reply == live.reply
    assert (hit.input_tokens, hit.output_tokens) == (live.input_tokens, live.output_tokens)


def test_three_keys_fail_over_in_order():
    fake = FakeCompletion(live_delay_s=0)
    gw = gateway(fake, api_keys=("a", "b", "c"))
    result = run(gw.complete(MSGS, simulate={"key_1": 429, "key_2": 503}))
    assert (result.key_id, result.fallback_attempts) == ("key_3", 2)
    assert [a["key_id"] for a in result.attempts] == ["key_1", "key_2", "key_3"]


def test_all_failed_raises_with_attempt_trail():
    fake = FakeCompletion(live_delay_s=0)
    fake.failing_keys = {"sk-test-one": 429, "sk-test-two": 500}
    with pytest.raises(AllProvidersUnavailable) as info:
        run(gateway(fake).complete(MSGS))
    assert [a["status"] for a in info.value.attempts] == [429, 500]
    assert "sk-test" not in repr(info.value.attempts)


def test_per_key_models():
    fake = FakeCompletion(live_delay_s=0)
    gw = gateway(fake, key_models=("gemini/gemini-2.0-flash", "groq/llama-3.1-8b-instant"))
    result = run(gw.complete(MSGS, simulate={"key_1": 500}))
    assert result.model == "groq/llama-3.1-8b-instant"
    assert fake.calls[-1]["model"] == "groq/llama-3.1-8b-instant"


def test_simulated_request_skips_cache_lookup_but_still_answers():
    fake = FakeCompletion(live_delay_s=0)
    gw = gateway(fake)
    run(gw.complete(MSGS))
    result = run(gw.complete(MSGS, simulate={"key_1": 429}))
    assert result.resolution_source == LIVE_INFERENCE and result.key_id == "key_2"
    assert fake.calls[-1]["cache"] == {"no-cache": True}


def test_cache_key_is_model_plus_messages():
    assert cache_key_for("m", MSGS) == cache_key_for("m", [dict(m) for m in MSGS])
    assert cache_key_for("m", MSGS) != cache_key_for("other", MSGS)
    assert cache_key_for("m", MSGS) != cache_key_for("m", MSGS[:1] + [{"role": "user", "content": "hi!"}])
    # independent of api_key and other call kwargs
    assert litellm_cache_key(model="m", messages=MSGS, api_key="x", temperature=1) == litellm_cache_key(
        model="m", messages=MSGS, api_key="y"
    )


@pytest.mark.parametrize(
    "exc,reason",
    [
        (FakeProviderError(429), "rate_limited"),
        (FakeProviderError(400, "RESOURCE_EXHAUSTED: quota"), "quota_exhausted"),
        (FakeProviderError(503), "server_error"),
        (FakeProviderError(401), "key_rejected"),
    ],
)
def test_classify_failure(exc, reason):
    assert classify_failure(exc) == reason


def test_usage_falls_back_to_token_counter():
    async def no_usage(**kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)

    gw = LLMGateway(make_settings(), completion_fn=no_usage, token_counter=lambda **kw: 7)
    result = run(gw.complete(MSGS))
    assert (result.input_tokens, result.output_tokens) == (7, 7)


# ---- storage ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,path",
    [
        ("sqlite:////data/app.db", "/data/app.db"),
        ("sqlite:///./data/app.db", "./data/app.db"),
        ("sqlite:///:memory:", ":memory:"),
        ("/tmp/x.db", "/tmp/x.db"),
    ],
)
def test_sqlite_path_from_url(url, path):
    assert sqlite_path_from_url(url) == path


def test_database_round_trip(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/nested/dir/app.db")  # parent dirs are created
    assert db.ping()
    assert not db.session_exists("s1")
    db.save_exchange("s1", "q1", "a1", {"resolution_source": "LIVE_INFERENCE"}, "custom prompt")
    db.save_exchange("s1", "q2", "a2", {"resolution_source": "LLM_CACHE"}, None)
    assert db.session_exists("s1")
    assert db.get_system_prompt("s1") == "custom prompt"  # a None override keeps the stored prompt
    history = db.get_history("s1")
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
    ]
    assert "metadata" not in history[0]
    assert history[3]["metadata"] == {"resolution_source": "LLM_CACHE"}
    assert history[0]["created_at"].endswith("+00:00")


def test_schema_matches_minimal_data_model(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/app.db")
    cols = {row["name"]: row for row in db._conn.execute("PRAGMA table_info(messages)")}
    assert list(cols) == ["id", "session_id", "role", "content", "metadata", "created_at"]
    assert cols["session_id"]["notnull"] == 1 and cols["metadata"]["notnull"] == 0
