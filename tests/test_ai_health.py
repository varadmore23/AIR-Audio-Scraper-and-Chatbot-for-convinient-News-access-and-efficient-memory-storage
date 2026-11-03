"""AI connection check: error classification (mocked SDK exceptions), status states, key never exposed,
context budget/metadata, and the one-retry rule for unusable answers. The real API is checked separately
(see README: python -c "import ai_health; ...") - every provider response here is a labelled mock."""

import pytest
from fastapi.testclient import TestClient

import ai_health
import config
import rag_engine

FAKE_KEY = "AIzaTESTKEYabcdefghijklmnopqrstuvwxyz0"


class ResourceExhausted(Exception): ...
class NotFound(Exception): ...
class PermissionDenied(Exception): ...
class DeadlineExceeded(Exception): ...
class ServiceUnavailable(Exception): ...


@pytest.mark.parametrize("exc,state", [
    (ResourceExhausted("429 You exceeded your current quota"), "quota_exceeded"),
    (NotFound("404 models/x is not found for API version v1beta"), "model_unavailable"),
    (PermissionDenied("403 API key not valid. Please pass a valid API key."), "auth_failed"),
    (Exception("400 API key not valid"), "auth_failed"),
    (DeadlineExceeded("504 Deadline Exceeded"), "timeout"),
    (ServiceUnavailable("503 failed to connect to all addresses"), "network_error"),
    (ConnectionError("getaddrinfo failed"), "network_error"),
])
def test_classify_error_mocked(exc, state):
    assert ai_health.classify_error(exc) == state


def test_scrub_removes_keys(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", FAKE_KEY)
    out = ai_health._scrub(f"bad request ?key={FAKE_KEY} and {FAKE_KEY}")
    assert FAKE_KEY not in out and "[redacted]" in out


@pytest.fixture
def client(corpus, monkeypatch):
    import chatbot_api
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    monkeypatch.setattr(ai_health, "_last_run", 0.0)
    with TestClient(chatbot_api.app) as c:
        yield c


def test_status_not_configured(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    d = client.get("/ai/status").json()
    assert d["state"] == "not_configured" and d["configured"] is False
    # no request is attempted without a key
    assert client.post("/ai/check").json()["state"] == "not_configured"


def test_key_set_is_unverified_until_checked(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", FAKE_KEY)
    d = client.get("/ai/status")
    assert d.json()["state"] == "unverified" and FAKE_KEY not in d.text


def _mock_probe(results):
    def probe(model, timeout=20.0):  # MOCK provider response, not a real API call
        return {"model": model, "latency_ms": 5.0, **results[model]}
    return probe


def test_check_primary_quota_uses_explicit_fallback(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", FAKE_KEY)
    models = rag_engine.gemini_models()
    res = {m: {"state": "connected", "ok": True, "detail": "reply 'OK'"} for m in models}
    res[models[0]] = {"state": "quota_exceeded", "ok": False, "detail": f"ResourceExhausted: 429 key={FAKE_KEY}"}
    monkeypatch.setattr(ai_health, "probe_model", _mock_probe(res))
    monkeypatch.setattr(rag_engine, "_model_cooldown", {})
    r = client.post("/ai/check")
    d = r.json()
    assert d["state"] == "connected" and d["serving_model"] == models[1]
    assert "fallback" in d["message"] and FAKE_KEY not in r.text
    assert rag_engine._model_cooldown.get(models[0])  # primary skipped by chat for its cooldown
    assert d["last_failure"]["model"] == models[0]
    # rate limit: a second check right away does not call the provider again
    monkeypatch.setattr(ai_health, "probe_model", lambda *a, **k: pytest.fail("probe called twice"))
    assert "note" in client.post("/ai/check").json()


def test_check_auth_failure_stops_after_first_model(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", FAKE_KEY)
    calls = []
    def probe(model, timeout=20.0):  # MOCK
        calls.append(model)
        return {"model": model, "state": "auth_failed", "ok": False, "latency_ms": 3.0, "detail": "API key not valid"}
    monkeypatch.setattr(ai_health, "probe_model", probe)
    d = client.post("/ai/check").json()
    assert d["state"] == "auth_failed" and len(calls) == 1


def test_generation_failing_after_good_check(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", FAKE_KEY)
    res = {m: {"state": "connected", "ok": True, "detail": "ok"} for m in rag_engine.gemini_models()}
    monkeypatch.setattr(ai_health, "probe_model", _mock_probe(res))
    client.post("/ai/check")
    import time
    time.sleep(1.1)  # checked_at has 1 s resolution
    ai_health.record_generation("m", False, error=DeadlineExceeded("deadline"))
    assert client.get("/ai/status").json()["state"] == "generation_failing"


def test_check_is_local_only(corpus, monkeypatch):
    import chatbot_api
    monkeypatch.setattr(config, "GEMINI_API_KEY", FAKE_KEY)
    with TestClient(chatbot_api.app, client=("203.0.113.9", 5000)) as c:
        assert c.post("/ai/check").status_code == 403


def _src(i, words, kind="article"):
    return {"id": f"S{i}", "kind": kind, "headline": f"Headline {i}", "source_name": "NewsOnAIR", "date": "2026-10-28",
            "time": "09:00", "genre": "national", "genre_name": "Politics & Governance", "matched_sections": ["Body"],
            "context": " ".join(f"w{i}x{j}" for j in range(words)), "text": f"matched passage {i}",
            "source_url": f"https://www.newsonair.gov.in/item-{i}/", "broadcast_type": "Morning News",
            "context_audio_start": 12.0, "context_audio_end": 40.0, "demo": False}


def test_context_has_metadata_and_explicit_budget(monkeypatch):
    monkeypatch.setattr(config, "RAG_CONTEXT_MAX_WORDS", 1000)
    monkeypatch.setattr(config, "RAG_CONTEXT_MAX_WORDS_PER_SOURCE", 600)
    stats = {}
    ctx = rag_engine._build_context([_src(1, 900), _src(2, 900), _src(3, 50, kind="broadcast")], stats)
    assert "url: https://www.newsonair.gov.in/item-1/" in ctx and "genre: Politics & Governance" in ctx
    assert "programme: Morning News" in ctx and "audio 00:12-00:40" in ctx
    assert "[... 300 more words of this source not shown]" in ctx      # per-source cap, marked
    assert "[context budget reached; matched passage only]" in ctx     # total budget, marked
    assert "[S3]" in ctx and stats["words_sent"] <= 1000 + 100 and stats["words_cut"] > 0


def test_answer_problem_detects_headings_only():
    srcs = [_src(1, 10)]
    assert rag_engine._answer_problem("### Topic\n\n### Sources\n- [S1] x, 2026-10-10", srcs)
    assert rag_engine._answer_problem("A claim with no citation at all, written in full.", srcs) == "no citations"
    assert rag_engine._answer_problem("The minister announced the scheme for farmers in four states "
                                      "on Tuesday morning [S1].", srcs) is None


def test_unusable_answer_retried_once_then_extractive(corpus, monkeypatch):
    calls = []
    def gen(context, q, history=None, guidance=""):  # MOCK generation: always headings only
        calls.append(guidance)
        return "### Heading\n\n### Sources\n- [S1] x"
    monkeypatch.setattr(rag_engine, "_generate_with_gemini", gen)
    r = rag_engine.answer_query("Tell me about the semiconductor story")
    assert len(calls) == 2 and "rejected" in calls[1] and r["mode"] == "extractive"
