"""
chatbot_api.py - FastAPI server for VaniStream (API + web UI).

Endpoints
  GET  /                    home page (static/home.html): project, pipeline, source, dataset
  GET  /newsroom            newsroom app (static/index.html): stories, broadcasts, transcripts, RAG chat
  GET  /static/...          shared assets (theme.css, theme.js)
  GET  /articles?genre=&q=&date_from=&date_to=&limit=&offset=   one page of article previews (no body)
  GET  /articles/{id}       one full article + related articles in the same genre
  GET  /broadcasts?mode=day|type&type=&genre=&language=&date_from=&date_to=&q=&offset=
                            broadcasts grouped by date, paginated by date (no transcripts)
  GET  /broadcasts/facets   broadcast types, genres and languages present, with counts
  GET  /genres              taxonomy (id, name) with counts of indexed articles / broadcasts / chunks
  GET  /coverage            configured reporting window vs actual coverage of indexed records
  GET  /analytics           analytics dashboard page; /analytics/* its data (summary, timeseries,
                            distributions, jobs, rag) computed from the database
  POST /chat                grounded RAG answer with [S#] citations
  GET  /timeline?q=         relevant segments grouped by date/time
  GET  /sources             list of ingested broadcasts (archive + live)
  GET  /transcript?source_file=...   all segments of one broadcast
  GET  /segment/{faiss_id}  one segment with its broadcast context
  GET  /audio/{faiss_id}    the real audio file behind a segment (404 if absent)
  GET  /ingest/status       real dataset counts (DB), ingestion jobs, running job, attribution
  POST /ingest/run          start a NewsOnAIR ingestion job (local + X-Admin-Token only; capped batch)
  POST /ingest/preview      validate a date range + broadcast types and discover what would be ingested (admin)
  GET  /ingest/preview/{id} discovery progress / result of a preview
  POST /ingest/confirm      start the job for a ready preview (admin, confirm=true)
  POST /ingest/retry        retry every failed document (admin)
  POST /ingest/cancel/{id}  request cancellation of a running job (admin)
  GET  /health              server, generation, models, live-stream and corpus health
  GET  /stats, /dedup       corpus statistics and dedup report
  POST /stream/start|stop, GET /stream/status   live ingestion control

Run:
  uvicorn chatbot_api:app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import config  # sets up logging first
from tracing import trace

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import analytics
import news_articles
import newsonair_ingest
import rag_engine
import taxonomy
import vector_store
from live_stream_ingest import live_manager

logger = logging.getLogger("api")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


_sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    genre: Optional[str] = None   # limit retrieval to one genre (newsroom scope selector)


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict]
    filters: dict
    session_id: str
    mode: str
    insufficient_context: bool
    trace_id: str
    cited: list[str] = []
    question_type: Optional[str] = None
    timings: dict = {}


class StreamStartRequest(BaseModel):
    url: Optional[str] = None   # default: LIVE_STREAM_URL
    language: Optional[str] = None
    chunk_seconds: Optional[int] = None
    overlap_seconds: Optional[int] = None
    fast: bool = True
    transcriber: Optional[str] = None
    model: Optional[str] = None
    source_name: Optional[str] = None
    programme: Optional[str] = None   # what is on air (a broadcast type name); empty = Other / Unclassified
    verify: bool = True               # check that the URL serves audio before starting


class StreamCheckRequest(BaseModel):
    url: Optional[str] = None         # empty = the configured LIVE_STREAM_URL


class IngestRunRequest(BaseModel):
    articles: int = 20
    broadcasts: int = 2


class IngestPreviewRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    types: list[str] = []
    include_articles: bool = False
    reprocess: bool = False      # include bulletins that are already indexed


class IngestConfirmRequest(BaseModel):
    preview_id: str
    confirm: bool = False


class IngestRetryRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class StreamStopRequest(BaseModel):
    stream_id: Optional[str] = None


def _startup():
    with trace("SYS"):
        logger.info("VaniStream API starting | %s", " | ".join(f"{k}={v}" for k, v in config.describe().items()))
        try:
            c = vector_store.check_consistency()
            lo, hi = vector_store.get_date_range()
            logger.info("Corpus: %d segments, %d vectors, dates %s..%s", c["sqlite_rows"], c["faiss_vectors"], lo, hi)
        except Exception as exc:
            logger.error("Store check failed: %s", exc)
        rag_engine.get_generation_mode()
        # Load the embedding model now (in the background) so the first question does not pay for it.
        import threading

        def _warm():
            try:
                t0 = time.perf_counter()
                import nlp_pipeline
                nlp_pipeline.embed_sentences(["warm up"])
                vector_store._load_index()
                logger.info("Models warmed up in %.1fs", time.perf_counter() - t0)
            except Exception as exc:
                logger.warning("Model warm-up failed: %s", exc)
        threading.Thread(target=_warm, daemon=True, name="warmup").start()
        if config.LIVE_AUTOSTART:
            try:
                live_manager.start()
            except Exception as exc:
                logger.error("LIVE_AUTOSTART failed: %s", exc)


def _shutdown():
    if live_manager.running:
        live_manager.stop()


@asynccontextmanager
async def lifespan(_app):
    _startup()
    yield
    _shutdown()


app = FastAPI(title="VaniStream - AIR News Intelligence",
              description="Grounded RAG over All India Radio broadcasts (archive + live)",
              version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path not in ("/stream/status", "/health", "/", "/ingest/status") and not path.startswith(("/static", "/audio")):
        logger.debug("%s %s -> %d (%.0f ms)", request.method, path, response.status_code,
                     (time.perf_counter() - t0) * 1000)
    return response


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Answer a question from the corpus. Runs in a worker thread (blocking model calls)."""
    sid = req.session_id or str(uuid.uuid4())
    history = _sessions.setdefault(sid, [])[-6:]
    with trace("REQ") as tid:
        try:
            result = rag_engine.answer_query(req.message, history=history, genre=req.genre or None)
        except Exception as exc:
            logger.exception("Chat request failed: %s", exc)
            return JSONResponse(status_code=500, content={"error": f"Internal error ({tid}): {exc}", "trace_id": tid})
    _sessions[sid] = (_sessions[sid] + [{"role": "user", "content": req.message},
                                       {"role": "model", "content": result["answer"]}])[-20:]
    return ChatResponse(reply=result["answer"], sources=result["sources"], filters=result["filters_applied"],
                        session_id=sid, mode=result["mode"], insufficient_context=result["insufficient_context"],
                        trace_id=result["trace_id"], cited=result.get("cited", []),
                        question_type=result.get("question_type"), timings=result.get("timings", {}))


@app.get("/timeline")
def timeline(q: str = "", date_from: Optional[str] = None, date_to: Optional[str] = None,
             genre: Optional[str] = None):
    if not q.strip() and not genre:
        return {"groups": [], "filters_applied": {}}
    with trace("REQ"):
        return rag_engine.timeline_query(q, date_from=date_from, date_to=date_to, genre=genre)


# ---------------------------------------------------------------------------
# Corpus browsing
# ---------------------------------------------------------------------------

@app.get("/articles")
def articles(genre: Optional[str] = None, q: Optional[str] = None, category: Optional[str] = None,
             date_from: Optional[str] = None, date_to: Optional[str] = None,
             limit: int = Query(24, ge=1, le=100), offset: int = Query(0, ge=0)):
    """One page of article previews (headline, metadata, teaser), newest first, filtered in SQL."""
    items, total = vector_store.list_articles_page(genre, category, q, date_from, date_to, limit, offset)
    facets = vector_store.article_facets()
    nxt = offset + len(items)
    sample = items or vector_store.list_articles_page(limit=1)[0]
    return {"articles": [_article_preview(a) for a in items], "total": total, "offset": offset,
            "next_offset": nxt if nxt < total else None,
            "genres": facets["genres"], "categories": facets["categories"], "date_range": facets["date_range"],
            "disclaimer": news_articles.disclaimer_for(sample)}


def _article_preview(a: dict) -> dict:
    p = news_articles.preview(a)
    p["genre_id"] = a.get("genre_id")
    p["genre_name"] = taxonomy.genre_name(a.get("genre_id")) if a.get("genre_id") else a.get("genre")
    p["genre_tags"] = [{"id": t, "name": taxonomy.genre_name(t)} for t in taxonomy.decode_tags(a.get("genre_tags"))]
    return p


@app.get("/articles/{article_id}")
def article(article_id: str):
    a = vector_store.get_article_data(article_id)
    if a is None:
        return JSONResponse(status_code=404, content={"error": f"Article {article_id} not found"})
    a.setdefault("hour", int((a.get("time") or "00:00").replace(":", "")))
    a.setdefault("word_count", len(news_articles.full_text(a).split()))
    a["genre_name"] = taxonomy.genre_name(a.get("genre_id")) if a.get("genre_id") else a.get("genre")
    a["genre_tag_names"] = [taxonomy.genre_name(t) for t in taxonomy.decode_tags(a.get("genre_tags"))]
    return {"article": a, "related": [_article_preview(r) for r in vector_store.related_articles(a)],
            "disclaimer": news_articles.disclaimer_for([a])}


@app.get("/genres")
def genres():
    return {"genres": [{"id": g, **v} for g, v in taxonomy.genre_counts().items()],
            "taxonomy": [{"id": g, "name": n} for g, n in taxonomy.GENRES.items()]}


# ---------------------------------------------------------------------------
# Broadcast browser
# ---------------------------------------------------------------------------

def _split(v: Optional[str]) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def _bad_date(label: str, d: Optional[str]) -> Optional[JSONResponse]:
    if d:
        try:
            analytics.dt.date.fromisoformat(d)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": f"{label} must be YYYY-MM-DD"})
    return None


@app.get("/broadcasts")
def broadcasts(mode: str = "day", type: Optional[str] = None, genre: Optional[str] = None,
               language: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None,
               q: Optional[str] = None, days: int = Query(7, ge=1, le=60), offset: int = Query(0, ge=0),
               order: str = "desc", limit: int = Query(25, ge=1, le=100)):
    """Broadcasts without transcripts.

    mode=all : one list, newest first, paginated by row (`limit`/`offset`)
    mode=day : grouped by broadcast date, paginated by date (`days`/`offset`); `type` optional
    mode=type: like day, for the given broadcast type(s) (required)
    """
    try:
        types = taxonomy.broadcast_type_ids(_split(type))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if mode == "type" and not types:
        return JSONResponse(status_code=400, content={"error": "mode=type needs a broadcast type"})
    bad = _bad_date("date_from", date_from) or _bad_date("date_to", date_to)
    if bad:
        return bad
    if date_from and date_to and date_from > date_to:
        return JSONResponse(status_code=400, content={"error": "date_from is after date_to"})
    if mode == "all":
        res = vector_store.list_broadcasts_flat(types, genre, language, date_from, date_to, q, limit, offset, order)
    else:
        res = vector_store.list_broadcast_days(types, genre, language, date_from, date_to, q, days, offset, order)
    res["mode"] = mode
    return res


@app.get("/broadcasts/facets")
def broadcasts_facets():
    return vector_store.broadcast_facets()


@app.get("/broadcasts/detail")
def broadcast_detail(source_file: str):
    b = vector_store.get_broadcast(source_file)
    if not b:
        return JSONResponse(status_code=404, content={"error": f"No broadcast {source_file}"})
    b["has_audio"] = vector_store.resolve_audio_path(b["source_file"])[1]
    return b


# ---------------------------------------------------------------------------
# Coverage + analytics dashboard (read-only, computed from the database)
# ---------------------------------------------------------------------------

def _filters(date_from=None, date_to=None, content="all", types=None, genre=None, language=None, status=None,
             interval="day"):
    return analytics.Filters(date_from, date_to, content, _split(types), genre, language, status, interval)


def _analytics(fn):
    try:
        return fn()
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/coverage")
def coverage(content: str = "all", types: Optional[str] = None, genre: Optional[str] = None,
             language: Optional[str] = None):
    return _analytics(lambda: analytics.cached("coverage", _filters(None, None, content, types, genre, language),
                                               analytics.coverage))


@app.get("/analytics/summary")
def analytics_summary(date_from: Optional[str] = None, date_to: Optional[str] = None, content: str = "all",
                      types: Optional[str] = None, genre: Optional[str] = None, language: Optional[str] = None,
                      status: Optional[str] = None):
    return _analytics(lambda: analytics.summary(_filters(date_from, date_to, content, types, genre, language, status)))


@app.get("/analytics/timeseries")
def analytics_timeseries(date_from: Optional[str] = None, date_to: Optional[str] = None, content: str = "all",
                         types: Optional[str] = None, genre: Optional[str] = None, language: Optional[str] = None,
                         interval: str = "day"):
    return _analytics(lambda: analytics.cached("ts", _filters(date_from, date_to, content, types, genre, language,
                                                              None, interval), analytics.indexed_over_time))


@app.get("/analytics/distributions")
def analytics_distributions(date_from: Optional[str] = None, date_to: Optional[str] = None, content: str = "all",
                            types: Optional[str] = None, genre: Optional[str] = None,
                            language: Optional[str] = None, metric: str = "documents"):
    return _analytics(lambda: analytics.cached("dist", _filters(date_from, date_to, content, types, genre, language),
                                               analytics.distributions, metric))


@app.get("/analytics/jobs")
def analytics_jobs(date_from: Optional[str] = None, date_to: Optional[str] = None, status: Optional[str] = None,
                   interval: str = "day"):
    def go():
        f = _filters(date_from, date_to, status=status, interval=interval)
        return {"history": analytics.cached("jobs", f, analytics.job_history, ttl=10),
                "funnel": analytics.cached("funnel", f, analytics.funnel, ttl=10)}
    return _analytics(go)


@app.get("/analytics/rag")
def analytics_rag(date_from: Optional[str] = None, date_to: Optional[str] = None, genre: Optional[str] = None,
                  interval: str = "day"):
    return _analytics(lambda: analytics.cached("rag", _filters(date_from, date_to, genre=genre, interval=interval),
                                               analytics.rag_series))


# ---------------------------------------------------------------------------
# NewsOnAIR ingestion (dashboard)
# ---------------------------------------------------------------------------

@app.get("/ingest/status")
def ingest_status():
    """Actual counts from the database plus job history. Read-only and cheap."""
    running = vector_store.running_jobs()
    return {
        "dataset": vector_store.get_dataset_stats(),
        "coverage": analytics.cached("coverage", analytics.Filters(), analytics.coverage),
        "jobs": vector_store.list_jobs(8),
        "running": newsonair_ingest.is_running() or bool(running),
        "current_job": newsonair_ingest.current_job(),
        "running_jobs": running,
        "broadcast_types": [{"id": t, "name": n} for t, (n, _) in taxonomy.BROADCAST_TYPES.items() if t != "other"],
        "range_limits": {"max_days": config.INGEST_MAX_RANGE_DAYS, "earliest": config.INGEST_EARLIEST_DATE,
                         "today": config.now_local().date().isoformat(), "timezone": config.TIMEZONE,
                         "max_broadcasts": config.INGEST_MAX_JOB_BROADCASTS,
                         "max_articles": config.INGEST_MAX_JOB_ARTICLES},
        "trigger_enabled": bool(config.INGEST_ADMIN_TOKEN),
        "limits": {"articles": config.INGEST_MAX_ARTICLES, "broadcasts": config.INGEST_MAX_BROADCASTS},
        "source": {"name": "NewsOnAIR", "url": config.NEWSONAIR_BASE},
        "attribution": news_articles.REAL_ATTRIBUTION,
    }


def _admin_denied(request: Request) -> Optional[JSONResponse]:
    """None when the request may control ingestion (token configured, local client, matching header)."""
    if not config.INGEST_ADMIN_TOKEN:
        return JSONResponse(status_code=403, content={"error": "Ingestion trigger is disabled (set INGEST_ADMIN_TOKEN "
                                                               "in .env, or run: python newsonair_ingest.py)"})
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return JSONResponse(status_code=403, content={"error": "Ingestion can only be started from the server machine"})
    if request.headers.get("x-admin-token", "") != config.INGEST_ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Invalid admin token"})
    return None


@app.post("/ingest/preview")
def ingest_preview(request: Request, req: IngestPreviewRequest):
    """Validate a date range + broadcast types, then discover matching items in the background.

    Admin-only: discovery reads the publisher's archive (rate-limited, sequential requests).
    """
    errors, _ = newsonair_ingest.validate_range(req.date_from, req.date_to, req.types, req.include_articles)
    if errors:
        return JSONResponse(status_code=400, content={"error": "Invalid request", "errors": errors})
    denied = _admin_denied(request)
    if denied:
        return denied
    pid = newsonair_ingest.preview(req.date_from, req.date_to, req.types, req.include_articles, req.reprocess)
    return {"preview_id": pid, "status": "discovering"}


@app.get("/ingest/preview/{preview_id}")
def ingest_preview_status(preview_id: str):
    p = newsonair_ingest.get_preview(preview_id)
    if not p:
        return JSONResponse(status_code=404, content={"error": "Unknown or expired preview"})
    return p


@app.post("/ingest/confirm")
def ingest_confirm(request: Request, req: IngestConfirmRequest):
    """Start the job for a ready preview. Requires confirm=true and the admin token."""
    denied = _admin_denied(request)
    if denied:
        return denied
    rec = newsonair_ingest._previews.get(req.preview_id)
    if not rec:
        return JSONResponse(status_code=404, content={"error": "Unknown or expired preview; run the preview again"})
    if rec["status"] != "ready":
        return JSONResponse(status_code=409, content={"error": f"Preview is {rec['status']}", "detail": rec.get("error")})
    if not req.confirm:
        return JSONResponse(status_code=400, content={"error": "Confirmation required (confirm=true)"})
    if rec.get("blocking"):
        return JSONResponse(status_code=400, content={"error": "; ".join(rec["blocking"])})
    if rec.get("nothing_to_do"):
        return JSONResponse(status_code=400, content={"error": "Nothing new to ingest in this range"})
    try:
        job_id = newsonair_ingest.run_in_background(
            date_from=rec["date_from"], date_to=rec["date_to"], types=rec["types"],
            items=rec.get("_items", []) if rec["types"] else None, broadcasts=None if rec["types"] else 0,
            include_articles=rec["include_articles"], articles=None if rec["include_articles"] else 0,
            force=rec["include_indexed"], trigger="dashboard")
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    newsonair_ingest._previews.pop(req.preview_id, None)
    logger.info("Ingestion job %s confirmed from the dashboard (preview %s)", job_id, req.preview_id)
    return {"job_id": job_id, "status": "started"}


@app.post("/ingest/retry")
def ingest_retry(request: Request, req: Optional[IngestRetryRequest] = None):
    denied = _admin_denied(request)
    if denied:
        return denied
    req = req or IngestRetryRequest()
    try:
        job_id = newsonair_ingest.run_in_background(retry_failed=True, date_from=req.date_from, date_to=req.date_to,
                                                    trigger="dashboard")
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    return {"job_id": job_id, "status": "started"}


@app.post("/ingest/cancel/{job_id}")
def ingest_cancel(request: Request, job_id: str):
    denied = _admin_denied(request)
    if denied:
        return denied
    if not newsonair_ingest.cancel(job_id):
        return JSONResponse(status_code=404, content={"error": f"No running job {job_id}"})
    return {"job_id": job_id, "status": "cancel requested"}


@app.post("/ingest/run")
def ingest_run(request: Request, req: Optional[IngestRunRequest] = None):
    """Start an incremental ingestion job.

    Locked down: disabled unless INGEST_ADMIN_TOKEN is set, accepted only from the
    local machine with the matching X-Admin-Token header, batch size capped, one
    job at a time. Visitors of a deployed site cannot start scraping jobs.
    """
    req = req or IngestRunRequest()
    denied = _admin_denied(request)
    if denied:
        return denied
    if newsonair_ingest.is_running():
        return JSONResponse(status_code=409, content={"error": "An ingestion job is already running",
                                                      "job": newsonair_ingest.current_job()})
    articles = max(0, min(req.articles, config.INGEST_MAX_ARTICLES))
    broadcasts = max(0, min(req.broadcasts, config.INGEST_MAX_BROADCASTS))
    try:
        job_id = newsonair_ingest.run_in_background(articles=articles, broadcasts=broadcasts, trigger="dashboard")
    except RuntimeError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    logger.info("Ingestion job %s started from the dashboard (articles=%d, broadcasts=%d)", job_id, articles, broadcasts)
    return {"job_id": job_id, "status": "started", "articles": articles, "broadcasts": broadcasts}


@app.get("/sources")
def sources(limit: int = Query(200, le=2000)):
    return {"sources": vector_store.list_sources(limit)}


@app.get("/transcript")
def transcript(source_file: str):
    segs = vector_store.get_transcript(source_file)
    if not segs:
        return JSONResponse(status_code=404, content={"error": f"No transcript indexed for {source_file}"})
    first = segs[0]
    return {"source_file": source_file, "source_name": first["source_name"], "date": first["date"],
            "ingest_mode": first.get("ingest_mode"), "has_audio": first["has_audio"],
            "audio_faiss_id": first["faiss_id"], "segments": segs,
            "broadcast": vector_store.get_broadcast(source_file)}


@app.get("/segment/{faiss_id}")
def segment(faiss_id: int):
    rows = vector_store.get_segments([faiss_id])
    if faiss_id not in rows:
        return JSONResponse(status_code=404, content={"error": f"Segment {faiss_id} not found"})
    return rows[faiss_id]


@app.get("/stats")
def stats():
    return vector_store.get_stats()


@app.get("/dedup")
def dedup():
    if not os.path.isfile("dedup_report.json"):
        return {"error": "no report"}
    try:
        with open("dedup_report.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@app.get("/ai/status")
def ai_status():
    """Generation provider configuration and the last recorded check/generation outcomes. No API call;
    never includes the key."""
    import ai_health
    return ai_health.status()


@app.post("/ai/check")
def ai_check(request: Request):
    """Send one minimal real generation request per configured model and record the result.
    Server machine only (and the admin token when one is configured); at most one check per 30 s."""
    import ai_health
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return JSONResponse(status_code=403, content={"error": "The connection check can only be run from the server machine"})
    if config.INGEST_ADMIN_TOKEN and request.headers.get("x-admin-token", "") != config.INGEST_ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Admin token required"})
    return ai_health.check_connection()


@app.get("/health")
def health():
    """Is the server up, is the stream connected, is audio arriving, are segments being stored?"""
    today = config.now_local().strftime("%Y-%m-%d")
    store = vector_store.get_live_stats(today)
    stream = live_manager.get_status()
    lo, hi = vector_store.get_date_range()
    status = "healthy"
    if stream["running"] and not stream["stream_connected"]:
        status = "degraded (stream reconnecting)"
    return {
        "status": status,
        "server_time": config.now_local().isoformat(timespec="seconds"),
        "generation_mode": rag_engine.get_generation_mode(),
        "ai_connection": __import__("ai_health").status()["state"],  # from the last real check; see /ai/status
        "models": config.describe(),
        "stream_running": stream["running"],
        "stream_connected": stream["stream_connected"],
        "last_audio_received": stream["last_audio_received"],
        "last_transcription": stream["last_transcription"],
        "segments_today": store["segments_today"],
        "live_segments_today": store["live_segments_today"],
        "last_segment_id": store["last_segment_id"],
        "last_indexed_at": store["last_indexed_at"],
        "corpus_date_range": {"from": lo, "to": hi},
        "stream": stream,
        "stream_control": {"requires_token": bool(config.INGEST_ADMIN_TOKEN)},
    }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@app.api_route("/audio/{faiss_id:int}", methods=["GET", "HEAD"])
def get_audio(faiss_id: int):
    """Serve the real recording for a segment. Never returns placeholder audio."""
    source_file = vector_store.get_source_of(faiss_id)
    if source_file is None:
        return JSONResponse(status_code=404, content={"error": f"Segment {faiss_id} not found"})
    audio_relpath, exists = vector_store.resolve_audio_path(source_file)
    audio_dir = os.path.abspath(os.path.join(config.OUTPUT_DIR, "audio"))
    target_abs = os.path.abspath(audio_relpath)
    try:
        if os.path.commonpath([audio_dir, target_abs]) != audio_dir:
            return JSONResponse(status_code=403, content={"error": "Access denied: path escapes audio directory"})
    except ValueError:
        return JSONResponse(status_code=403, content={"error": "Access denied: path escapes audio directory"})
    if not exists:
        return JSONResponse(status_code=404, content={"error": f"Audio file not found on disk for segment {faiss_id}"})
    with open(target_abs, "rb") as f:
        header = f.read(12)
    mime = "audio/wav" if header[:4] == b"RIFF" and header[8:12] == b"WAVE" else "audio/mpeg"
    return FileResponse(target_abs, media_type=mime, filename=os.path.basename(target_abs))


# ---------------------------------------------------------------------------
# Live stream control
# ---------------------------------------------------------------------------

def _stream_denied(request: Request) -> Optional[JSONResponse]:
    """Live ingestion runs ASR continuously: only the server machine may start it (and, when an admin
    token is configured, only with that token)."""
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return JSONResponse(status_code=403, content={"error": "Live ingestion can only be controlled from the server machine"})
    if config.INGEST_ADMIN_TOKEN and request.headers.get("x-admin-token", "") != config.INGEST_ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Admin token required to control live ingestion"})
    return None


@app.post("/stream/check")
def stream_check(request: Request, req: Optional[StreamCheckRequest] = None):
    """Validate a stream URL (or the configured one) without starting anything."""
    denied = _stream_denied(request)
    if denied:
        return denied
    from live_stream_ingest import probe_stream, public_url
    url = ((req.url if req else None) or "").strip()
    source = "manual" if url else "configured"
    url = url or config.LIVE_STREAM_URL
    if not url:
        return {"ok": False, "source": None, "url": None,
                "detail": "No stream URL entered and LIVE_STREAM_URL is not configured"}
    res = probe_stream(url)
    return {**res, "source": source, "url": public_url(url)}


@app.post("/stream/start")
def stream_start(request: Request, req: Optional[StreamStartRequest] = None):
    denied = _stream_denied(request)
    if denied:
        return denied
    req = req or StreamStartRequest()
    if req.programme:
        try:
            ids = taxonomy.broadcast_type_ids([req.programme])
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        req.programme = None if ids == ["other"] else taxonomy.broadcast_type_name(ids[0])
    if live_manager.running:
        return JSONResponse(status_code=400, content={"error": f"A stream is already active with ID '{live_manager.stream_id}'. Stop it first."})
    try:
        sid = live_manager.start(url=req.url, language=req.language, chunk_seconds=req.chunk_seconds,
                                 overlap_seconds=req.overlap_seconds, fast=req.fast, transcriber=req.transcriber,
                                 model=req.model, source_name=req.source_name, verify=req.verify,
                                 programme=req.programme)
        return {"stream_id": sid, "status": "started"}
    except (ValueError, RuntimeError) as err:
        return JSONResponse(status_code=400, content={"error": str(err)})
    except Exception as err:
        logger.error("Failed to start stream: %s", err)
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.post("/stream/stop")
def stream_stop(request: Request, req: Optional[StreamStopRequest] = None):
    denied = _stream_denied(request)
    if denied:
        return denied
    try:
        return live_manager.stop(stream_id=req.stream_id if req else None)
    except ValueError as err:
        return JSONResponse(status_code=400, content={"error": str(err)})


@app.get("/stream/status")
def stream_status():
    return live_manager.get_status()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "home.html"), media_type="text/html")


@app.get("/newsroom")
def newsroom():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")


@app.get("/analytics")
def analytics_page():
    return FileResponse(os.path.join(STATIC_DIR, "analytics.html"), media_type="text/html")
