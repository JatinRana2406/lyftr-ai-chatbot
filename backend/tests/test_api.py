"""End-to-end tests of the HTTP contract (sections 3 and 4 of the assignment)."""

from __future__ import annotations

import uuid

import pytest


def sid() -> str:
    return f"s-{uuid.uuid4().hex[:8]}"


def chat(client, message, session_id=None, headers=None, **extra):
    body = {"session_id": session_id or sid(), "message": message, **extra}
    return client.post("/chat", json=body, headers=headers or {})


# ---- /health ------------------------------------------------------------------


def test_health_reports_keys_and_cache(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


# ---- /chat: response shape ------------------------------------------------------


def test_chat_response_matches_contract(client):
    session = sid()
    res = chat(client, "What is the capital of France?", session)
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == session
    assert body["reply"]
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "gemini/gemini-2.0-flash"
    assert isinstance(meta["latency_ms"], int)
    assert isinstance(meta["input_tokens"], int) and meta["input_tokens"] > 0
    assert isinstance(meta["output_tokens"], int) and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


def test_api_key_never_leaks(client):
    text = chat(client, "hello").text + client.get("/health").text
    assert "sk-test-one" not in text and "sk-test-two" not in text


# ---- caching --------------------------------------------------------------------


def test_same_first_message_in_new_session_hits_cache(client, fake):
    first = chat(client, "Tell me a fact about owls.").json()
    second = chat(client, "Tell me a fact about owls.").json()
    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]
    assert second["metadata"]["latency_ms"] <= first["metadata"]["latency_ms"]
    assert len(fake.calls) == 1


def test_different_message_never_from_cache(client):
    chat(client, "Message A")
    res = chat(client, "Message B").json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_history(client):
    """Same text as a second turn has a different message list, so it must not hit the cache."""
    chat(client, "Ping")
    session = sid()
    chat(client, "Something else", session)
    res = chat(client, "Ping", session).json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_system_prompt(client):
    chat(client, "Same text", system_prompt="You are a pirate.")
    res = chat(client, "Same text", system_prompt="You are a poet.").json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_disabled(make_client):
    client = make_client(cache_enabled=False)
    assert client.get("/health").json()["cache_enabled"] is False
    chat(client, "repeat me")
    assert chat(client, "repeat me").json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ---- tokens -----------------------------------------------------------------------


def test_tokens_are_per_request_not_running_totals(client):
    a = chat(client, "one two three four").json()["metadata"]
    b = chat(client, "five six seven eight").json()["metadata"]
    assert a["input_tokens"] == b["input_tokens"]  # equal-length first messages, new sessions

    session = sid()
    t1 = chat(client, "alpha beta", session).json()["metadata"]
    t2 = chat(client, "gamma delta", session).json()["metadata"]
    # the second turn's prompt includes history (system + history + new message) ...
    assert t2["input_tokens"] > t1["input_tokens"]
    # ... but output tokens are for that reply only (equal-length replies -> equal counts, no running total)
    assert t2["output_tokens"] == t1["output_tokens"]


# ---- multi-turn + history -------------------------------------------------------------


def test_multi_turn_history_sent_to_llm_and_stored(client, fake):
    session = sid()
    chat(client, "My name is Asha.", session)
    chat(client, "What is my name?", session)
    sent = fake.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "My name is Asha."

    res = client.get(f"/sessions/{session}/messages")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == session
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "My name is Asha."
    assert "metadata" not in msgs[0]
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert isinstance(msgs[1]["metadata"]["input_tokens"], int)
    assert all(m["role"] != "system" for m in msgs)


def test_sessions_are_isolated(client, fake):
    a, b = sid(), sid()
    chat(client, "secret for A", a)
    chat(client, "hello from B", b)
    assert all("secret for A" not in m["content"] for m in fake.calls[-1]["messages"])
    assert len(client.get(f"/sessions/{b}/messages").json()["messages"]) == 2


def test_unknown_session_is_404(client):
    res = client.get("/sessions/does-not-exist/messages")
    assert res.status_code == 404
    assert res.json() == {"detail": "session not found"}


def test_system_prompt_override_is_used_and_remembered(client, fake):
    session = sid()
    chat(client, "hi", session, system_prompt="You only answer in French.")
    assert fake.calls[-1]["messages"][0] == {"role": "system", "content": "You only answer in French."}
    chat(client, "again", session)  # no override sent: the session keeps its prompt
    assert fake.calls[-1]["messages"][0]["content"] == "You only answer in French."
    history = client.get(f"/sessions/{session}/messages").json()["messages"]
    assert all("French" not in m["content"] for m in history)


def test_history_survives_app_restart(make_client, tmp_path):
    session = sid()
    chat(make_client(), "persist me", session)
    fresh = make_client()  # new app + new connection on the same SQLite file
    assert fresh.get(f"/sessions/{session}/messages").json()["messages"][0]["content"] == "persist me"


# ---- validation (422) -------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "s-1"},
        {"message": "hi"},
        {"session_id": "s-1", "message": ""},
        {"session_id": "s-1", "message": "   \n\t "},
        {"session_id": "", "message": "hi"},
        {"session_id": "x" * 129, "message": "hi"},
        {"session_id": "s-1", "message": "a" * 4001},
        {"session_id": 123, "message": "hi"},
        {"session_id": "s-1", "message": None},
    ],
)
def test_invalid_bodies_are_422_json(client, body):
    res = client.post("/chat", json=body)
    assert res.status_code == 422
    assert "detail" in res.json()


def test_malformed_json_is_422(client):
    res = client.post("/chat", content=b'{"session_id": "s-1", "message": ', headers={"Content-Type": "application/json"})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_limits_are_inclusive_and_measured_after_trim(client):
    assert chat(client, "  " + "a" * 4000 + "  ", "x" * 128).status_code == 200


def test_failed_request_not_added_to_history(client):
    session = sid()
    chat(client, "kept", session)
    client.post("/chat", json={"session_id": session, "message": ""})
    chat(client, "dropped", session, headers={"X-Simulate-Failure": "*=429"})
    msgs = client.get(f"/sessions/{session}/messages").json()["messages"]
    assert [m["content"] for m in msgs if m["role"] == "user"] == ["kept"]


def test_failed_first_request_creates_no_session(client):
    session = sid()
    assert chat(client, "x", session, headers={"X-Simulate-Failure": "*=503"}).status_code == 503
    assert client.get(f"/sessions/{session}/messages").status_code == 404


# ---- failover via X-Simulate-Failure (section 4) ----------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_key_1_failure_falls_back_to_key_2(client, status):
    res = chat(client, f"failover {status}", headers={"X-Simulate-Failure": f"key_1={status}"})
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert "error" not in res.json()["reply"].lower()


def test_all_keys_simulated_down_is_503(client):
    res = chat(client, "nothing works", headers={"X-Simulate-Failure": "*=429"})
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}


def test_both_keys_listed_is_503(client):
    res = chat(client, "both", headers={"X-Simulate-Failure": "key_1=429,key_2=500"})
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}


def test_simulation_bypasses_cache_so_failover_is_visible(client):
    chat(client, "cached question")
    res = chat(client, "cached question", headers={"X-Simulate-Failure": "key_1=429"}).json()
    assert res["metadata"]["key_id"] == "key_2"
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_simulation_is_for_one_request_only(client):
    chat(client, "first", headers={"X-Simulate-Failure": "key_1=429"})
    assert chat(client, "second").json()["metadata"]["key_id"] == "key_1"


def test_simulation_ignored_when_disabled(make_client):
    client = make_client(enable_failure_simulation=False)
    res = chat(client, "no sim", headers={"X-Simulate-Failure": "*=429"})
    assert res.status_code == 200
    assert res.json()["metadata"]["key_id"] == "key_1"


@pytest.mark.parametrize("header", ["key_1", "key_1=418", "bogus=429", "key_1=abc"])
def test_invalid_simulation_header_is_422(client, header):
    res = chat(client, "bad header", headers={"X-Simulate-Failure": header})
    assert res.status_code == 422
    assert "detail" in res.json()


# ---- real (non-simulated) provider failures ----------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
def test_real_provider_error_triggers_failover(client, fake, status):
    fake.failing_keys["sk-test-one"] = status
    meta = chat(client, f"real {status}").json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1


def test_real_all_keys_down_is_503(client, fake):
    fake.failing_keys.update({"sk-test-one": 429, "sk-test-two": 500})
    res = chat(client, "down")
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}


def test_empty_reply_counts_as_failure(client, fake):
    fake.empty_reply_keys.add("sk-test-one")
    res = chat(client, "say something").json()
    assert res["reply"]
    assert res["metadata"]["key_id"] == "key_2"


def test_unexpected_crash_is_json_500_without_stack_trace(client, fake):
    client.app.state.gateway.complete = None  # break the gateway entirely -> TypeError inside the handler
    res = chat(client, "crash")
    assert res.status_code == 500
    assert res.json() == {"detail": "internal server error"}
    assert "Traceback" not in res.text and "secret" not in res.text


def test_unknown_route_is_json(client):
    res = client.get("/nope")
    assert res.status_code == 404
    assert "detail" in res.json()


def test_cors_allows_frontend_origin(client):
    res = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"
