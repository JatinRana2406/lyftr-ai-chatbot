"""Runs the gateway against the REAL LiteLLM library (no network: LiteLLM's mock_response).

Proves the parts that depend on LiteLLM itself: our custom cache key is used by
LiteLLM's built-in cache, hits are flagged, and the identical reply + stored usage
come back on a hit.
"""

from __future__ import annotations

import asyncio

import pytest

litellm = pytest.importorskip("litellm")

from app.gateway import LIVE_INFERENCE, LLM_CACHE, LLMGateway  # noqa: E402
from tests.fakes import make_settings  # noqa: E402


@pytest.fixture
def gw():
    gateway = LLMGateway(make_settings(mock_llm=True))
    yield gateway
    litellm.cache = None


def test_litellm_cache_hit_and_miss(gw):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "integration ping"}]
    first = asyncio.run(gw.complete(msgs))
    second = asyncio.run(gw.complete(msgs))
    other = asyncio.run(gw.complete(msgs[:1] + [{"role": "user", "content": "different"}]))

    assert first.resolution_source == LIVE_INFERENCE and first.key_id == "key_1"
    assert second.resolution_source == LLM_CACHE and second.key_id is None
    assert second.reply == first.reply
    assert (second.input_tokens, second.output_tokens) == (first.input_tokens, first.output_tokens)
    assert other.resolution_source == LIVE_INFERENCE


def test_litellm_simulated_failover(gw):
    msgs = [{"role": "user", "content": "failover via litellm"}]
    result = asyncio.run(gw.complete(msgs, simulate={"key_1": 429}))
    assert result.key_id == "key_2" and result.fallback_attempts == 1
    assert "key_2" in result.reply  # the mock reply names the key that produced it
