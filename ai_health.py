"""
ai_health.py - real connection checks for the configured generation provider (Google Gemini API).

A configured key is not treated as "connected". check_connection() sends one minimal generation request
(a 5-token reply) to each configured model, classifies the outcome, measures latency and stores the result
in SQLite (table ai_health_checks). status() reports configuration, the latest successful and failed
checks and the latest chat-generation outcome without making any API call. Secrets never leave the
server: errors are reduced to a class and a short provider message with any key-like text removed.

States
  not_configured   GEMINI_API_KEY is empty
  unverified       key set, no check has run yet
  connected        a minimal generation request succeeded
  auth_failed      key rejected (invalid / permission denied)
  model_unavailable  model id not found or not served
  quota_exceeded   429 / resource exhausted
  network_error    DNS / connection failure
  timeout          request exceeded the timeout
  invalid_response the provider answered without usable text
  generation_failing  the connection check passes but recent chat generations failed
  sdk_missing      google-generativeai is not installed
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
from typing import Optional

import config

logger = logging.getLogger("ai_health")

PROVIDER = "Google Gemini API (generativelanguage.googleapis.com) via google-generativeai"
MIN_INTERVAL_S = 30.0          # a manual check is refused if one ran this recently
_lock = threading.Lock()
_last_run = 0.0

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ai_health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TEXT NOT NULL,
    kind TEXT NOT NULL,            -- check | generation
    model TEXT,
    state TEXT NOT NULL,
    ok INTEGER NOT NULL,
    latency_ms REAL,
    detail TEXT
)
"""


def _db():
    import vector_store
    conn = vector_store._get_db()
    conn.execute(CREATE_SQL)
    return conn


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scrub(text: str) -> str:
    """Remove anything that looks like a credential before a message is stored or returned."""
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[redacted]", text or "")
    text = re.sub(r"(key=)[^&\s]+", r"\1[redacted]", text)
    if config.GEMINI_API_KEY:
        text = text.replace(config.GEMINI_API_KEY, "[redacted]")
    return text[:240]


def classify_error(exc: BaseException) -> str:
    """Map a provider/SDK exception to a connection state."""
    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()
    if "resourceexhausted" in name or "429" in text or "quota" in text or "rate limit" in text:
        return "quota_exceeded"
    if "deadline" in name or "timeout" in name or "timed out" in text or "504" in text:
        return "timeout"
    if ("unauthenticated" in name or "permissiondenied" in name or "api key not valid" in text
            or "api_key_invalid" in text or "401" in text or "403" in text):
        return "auth_failed"
    if "notfound" in name or "404" in text or "not found" in text or "is not supported" in text:
        return "model_unavailable"
    if ("serviceunavailable" in name or "connection" in text or "getaddrinfo" in text
            or "name resolution" in text or "unreachable" in text or "503" in text or "retryerror" in name):
        return "network_error"
    if "invalidargument" in name or "400" in text:
        return "auth_failed" if "key" in text else "invalid_response"
    return "error"


def _record(kind: str, model: Optional[str], state: str, ok: bool, latency_ms: Optional[float], detail: str):
    try:
        conn = _db()
        try:
            conn.execute("INSERT INTO ai_health_checks (checked_at, kind, model, state, ok, latency_ms, detail) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", (_now(), kind, model, state, int(ok), latency_ms, _scrub(detail)))
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # health bookkeeping must never break answering
        logger.warning("Could not record AI health result: %s", exc)


def record_generation(model: Optional[str], ok: bool, error: Optional[BaseException] = None,
                      latency_ms: Optional[float] = None) -> None:
    """Called by the RAG engine after each generation attempt (success or the final failure)."""
    state = "connected" if ok else (classify_error(error) if error else "invalid_response")
    _record("generation", model, state, ok, latency_ms, "" if ok else f"{type(error).__name__ if error else ''}: "
            f"{error or 'empty response'}")


def probe_model(model: str, timeout: float = 20.0) -> dict:
    """One minimal real generation request. Returns {model, state, ok, latency_ms, detail}."""
    try:
        import google.generativeai as genai
    except ImportError:
        return {"model": model, "state": "sdk_missing", "ok": False, "latency_ms": None,
                "detail": "pip install google-generativeai"}
    from google.api_core import retry as gretry
    genai.configure(api_key=config.GEMINI_API_KEY)
    t0 = time.perf_counter()
    try:
        resp = genai.GenerativeModel(model).generate_content(
            "Reply with the single word OK.", generation_config={"max_output_tokens": 5, "temperature": 0},
            request_options={"timeout": timeout, "retry": gretry.Retry(predicate=lambda e: False, timeout=timeout)})
        ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            text = (resp.text or "").strip()
        except Exception as exc:  # blocked / no candidates -> no .text
            return {"model": model, "state": "invalid_response", "ok": False, "latency_ms": ms,
                    "detail": _scrub(f"{type(exc).__name__}: {exc}")}
        if not text:
            return {"model": model, "state": "invalid_response", "ok": False, "latency_ms": ms,
                    "detail": "empty text in response"}
        return {"model": model, "state": "connected", "ok": True, "latency_ms": ms, "detail": f"reply {text[:12]!r}"}
    except Exception as exc:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"model": model, "state": classify_error(exc), "ok": False, "latency_ms": ms,
                "detail": _scrub(f"{type(exc).__name__}: {exc}")}


def check_connection(force: bool = False) -> dict:
    """Run a real check of every configured model (rate-limited) and return the full status."""
    global _last_run
    if not config.GEMINI_API_KEY:
        return status()
    with _lock:
        if not force and time.monotonic() - _last_run < MIN_INTERVAL_S:
            st = status()
            st["note"] = f"A check ran less than {int(MIN_INTERVAL_S)} s ago; showing its result."
            return st
        _last_run = time.monotonic()
        import rag_engine
        results = []
        for m in rag_engine.gemini_models():
            r = probe_model(m)
            r["detail"] = _scrub(r.get("detail") or "")
            results.append(r)
            _record("check", m, r["state"], r["ok"], r["latency_ms"], r["detail"])
            if r["state"] in ("quota_exceeded", "model_unavailable"):
                # the chat path skips this model for its cooldown instead of waiting on it per question
                rag_engine._model_cooldown[m] = time.monotonic() + config.GEMINI_COOLDOWN_SECONDS
            if r["state"] in ("auth_failed", "network_error", "sdk_missing"):
                break  # the same for every model; do not spend more requests
    st = status()
    st["results"] = results
    return st


def _latest(conn, kind: str, ok: Optional[bool] = None, model: Optional[str] = None):
    q = "SELECT * FROM ai_health_checks WHERE kind = ?"
    args: list = [kind]
    if ok is not None:
        q += " AND ok = ?"
        args.append(int(ok))
    if model:
        q += " AND model = ?"
        args.append(model)
    r = conn.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
    return dict(r) if r else None


def status() -> dict:
    """Configuration + latest recorded outcomes. Makes no API call."""
    import rag_engine
    models = rag_engine.gemini_models()
    out = {"provider": PROVIDER, "configured_model": config.GEMINI_MODEL, "fallback_models": models[1:],
           "api_key_env": "GEMINI_API_KEY", "configured": bool(config.GEMINI_API_KEY),
           "fine_tuning": {"supported": False,
                           "reason": "No model on this account lists createTunedModel and the tuned-models endpoint "
                                     "returns 501 (checked with list_models / list_tuned_models)."}}
    if not out["configured"]:
        out.update(state="not_configured", message="Set GEMINI_API_KEY in the environment or in .env "
                                                   "(project root), then restart the server.")
        return out
    try:
        conn = _db()
        try:
            last_ok = _latest(conn, "check", True)
            last_fail = _latest(conn, "check", False)
            per_model = {m: _latest(conn, "check", model=m) for m in models}
            last_gen = _latest(conn, "generation")
            last_gen_ok = _latest(conn, "generation", True)
        finally:
            conn.close()
    except Exception as exc:
        out.update(state="unverified", message=f"Health records unavailable: {exc}")
        return out
    out.update(last_success=last_ok, last_failure=last_fail, models=per_model, last_generation=last_gen,
               last_generation_success=last_gen_ok, active_model=rag_engine._last_model_used or None)
    usable = [m for m, r in per_model.items() if r and r["ok"]]
    if not any(per_model.values()):
        out.update(state="unverified", message="Key configured; no connection check has run yet.")
    elif usable:
        newest_check = max(r["checked_at"] for r in per_model.values() if r)
        if last_gen and not last_gen["ok"] and last_gen["checked_at"] > newest_check:
            out.update(state="generation_failing",
                       message=f"Connection check passed, but the latest chat generation failed ({last_gen['state']}).")
        else:
            primary = per_model.get(models[0])
            fb = primary is None or not primary["ok"]
            out.update(state="connected", serving_model=usable[0],
                       message=(f"Connected. {models[0]} is {primary['state'].replace('_', ' ')}; answers use the "
                                f"fallback {usable[0]}." if fb and primary else f"Connected via {usable[0]}."))
    else:
        worst = next(r for r in per_model.values() if r)
        out.update(state=worst["state"], message=f"No configured model answered: {worst['detail']}")
    return out
