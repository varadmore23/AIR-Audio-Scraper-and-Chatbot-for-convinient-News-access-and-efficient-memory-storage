"""
config.py - Central configuration for VaniStream.

Every tunable value lives here and can be overridden with an environment
variable or a line in a local `.env` file (KEY=value, no quotes needed).
Nothing else in the codebase should read os.environ directly for settings.
"""

from __future__ import annotations

import datetime as dt
import os

_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_dotenv(path: str = _ENV_FILE) -> None:
    """Minimal .env loader (no extra dependency). Real env vars win."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


# --- Paths -----------------------------------------------------------------
OUTPUT_DIR = _env("VANI_OUTPUT_DIR", "output")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
FAISS_INDEX_PATH = _env("VANI_FAISS_INDEX", "faiss_index.bin")
SQLITE_DB_PATH = _env("VANI_SQLITE_DB", "news_store.db")

# --- Time ------------------------------------------------------------------
# AIR broadcasts on IST; "yesterday" / "7 PM" are interpreted in this zone.
TIMEZONE = _env("VANI_TIMEZONE", "Asia/Kolkata")
# "today"  -> relative dates are anchored on the real current date (live mode)
# "corpus" -> anchored on the newest date in the corpus (archive-only demos)
DATE_ANCHOR = _env("VANI_DATE_ANCHOR", "today").lower()


def get_tz() -> dt.tzinfo:
    """Return the configured timezone (falls back to the machine's local zone)."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(TIMEZONE)
    except Exception:  # tzdata missing on Windows, bad name, ...
        return dt.datetime.now().astimezone().tzinfo


def now_local() -> dt.datetime:
    """Timezone-aware current time in the configured zone."""
    return dt.datetime.now(get_tz())


# --- Speech recognition ----------------------------------------------------
ASR_BACKEND = _env("VANI_ASR_BACKEND", "faster-whisper")   # faster-whisper | whisper
ASR_MODEL = _env("VANI_ASR_MODEL", "small")                 # tiny|base|small|medium|large-v3|distil-large-v3
ASR_DEVICE = _env("VANI_ASR_DEVICE", "cpu")                 # cpu | cuda
ASR_COMPUTE_TYPE = _env("VANI_ASR_COMPUTE_TYPE", "int8")    # int8 on CPU, float16 on GPU
ASR_LANGUAGE = _env("VANI_ASR_LANGUAGE", "en")              # ISO code or "auto"
ASR_BEAM_SIZE = _env_int("VANI_ASR_BEAM_SIZE", 5)

# --- Live stream -----------------------------------------------------------
LIVE_STREAM_URL = _env("LIVE_STREAM_URL", "").strip()
LIVE_SOURCE_NAME = _env("LIVE_SOURCE_NAME", "AIR Live")
LIVE_CHUNK_SECONDS = _env_int("LIVE_CHUNK_SECONDS", 30)
LIVE_OVERLAP_SECONDS = _env_int("LIVE_OVERLAP_SECONDS", 5)
LIVE_RETRY_BASE_SECONDS = _env_float("LIVE_RETRY_BASE_SECONDS", 5.0)
LIVE_RETRY_MAX_SECONDS = _env_float("LIVE_RETRY_MAX_SECONDS", 300.0)
# 0 = retry forever (overnight runs). A positive number caps reconnect attempts.
LIVE_MAX_RETRIES = _env_int("LIVE_MAX_RETRIES", 0)
# Start live ingestion automatically when the API server starts (overnight runs).
LIVE_AUTOSTART = _env("LIVE_AUTOSTART", "0").lower() in ("1", "true", "yes")
# If ffmpeg produces no audio for this many seconds, the connection is recycled.
LIVE_READ_TIMEOUT_SECONDS = _env_float("LIVE_READ_TIMEOUT_SECONDS", 60.0)

# --- NLP -------------------------------------------------------------------
EMBEDDING_MODEL = _env("VANI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = _env_int("VANI_EMBEDDING_DIM", 384)
MMR_THRESHOLD = _env_float("VANI_MMR_THRESHOLD", 0.85)
# Live dedup only compares against the last N hours of stored segments, so a
# story repeated at 09:00/09:30/10:00 is merged, while a genuinely new
# development next week is not blocked by last week's wording.
LIVE_DEDUP_WINDOW_HOURS = _env_int("LIVE_DEDUP_WINDOW_HOURS", 24)

# --- Retrieval / generation -----------------------------------------------
RAG_TOP_K = _env_int("VANI_RAG_TOP_K", 6)
RAG_CANDIDATES = _env_int("VANI_RAG_CANDIDATES", 30)
RAG_MIN_DENSE_SCORE = _env_float("VANI_RAG_MIN_DENSE_SCORE", 0.25)
# The best match of a topical question must reach this cosine, or the question is treated as
# not covered. With all-MiniLM-L6-v2, off-topic questions top out around 0.26-0.35 on the
# sample corpus while answerable ones score 0.55+. 0 disables the check.
RAG_MIN_TOP_SCORE = _env_float("VANI_RAG_MIN_TOP_SCORE", 0.40)
# Below this best-match cosine, a topic word that appears in no indexed passage means "not covered".
RAG_STRONG_MATCH = _env_float("VANI_RAG_STRONG_MATCH", 0.55)
RAG_CONTEXT_NEIGHBORS = _env_int("VANI_RAG_CONTEXT_NEIGHBORS", 1)
# Context budget sent to the model (words). Articles longer than the per-source cap are cut with an
# explicit "[... N more words not shown]" marker; sources past the total budget are listed, not silently dropped.
RAG_CONTEXT_MAX_WORDS = _env_int("VANI_RAG_CONTEXT_MAX_WORDS", 7000)
RAG_CONTEXT_MAX_WORDS_PER_SOURCE = _env_int("VANI_RAG_CONTEXT_MAX_WORDS_PER_SOURCE", 1800)
GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
# "gemini-flash-latest" follows Google's current Flash model; a pinned name such
# as gemini-2.5-flash stops working when Google retires it.
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-flash-latest")
# Long, structured answers need room; thinking models also count reasoning tokens here.
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 8192)
# Tried in order when the primary model's quota is exhausted (429) or it is retired (404).
# Lite models have separate free-tier quotas, so they usually keep answering.
GEMINI_FALLBACK_MODELS = [m.strip() for m in _env(
    "GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest,gemini-3.5-flash-lite").split(",") if m.strip()]
GEMINI_COOLDOWN_SECONDS = _env_float("GEMINI_COOLDOWN_SECONDS", 300.0)
GEMINI_TIMEOUT_SECONDS = _env_float("GEMINI_TIMEOUT_SECONDS", 90.0)

# --- NewsOnAIR ingestion (newsonair_ingest.py) ------------------------------
NEWSONAIR_BASE = _env("NEWSONAIR_BASE", "https://newsonair.gov.in")
NEWSONAIR_USER_AGENT = _env("NEWSONAIR_USER_AGENT",
                            "VaniStream-academic-research/1.0 (independent student project; polite crawler)")
NEWSONAIR_DELAY_SECONDS = _env_float("NEWSONAIR_DELAY_SECONDS", 1.5)   # pause between requests
NEWSONAIR_TIMEOUT_SECONDS = _env_float("NEWSONAIR_TIMEOUT_SECONDS", 90.0)
NEWSONAIR_ARTICLE_LIMIT = _env_int("NEWSONAIR_ARTICLE_LIMIT", 80)
NEWSONAIR_BROADCAST_LIMIT = _env_int("NEWSONAIR_BROADCAST_LIMIT", 8)
NEWSONAIR_BULLETIN_PAGES = _env_int("NEWSONAIR_BULLETIN_PAGES", 6)       # archive pages scanned per run
NEWSONAIR_MAX_ATTEMPTS = _env_int("NEWSONAIR_MAX_ATTEMPTS", 3)           # retries for a failed document
NEWSONAIR_MIN_ARTICLE_WORDS = _env_int("NEWSONAIR_MIN_ARTICLE_WORDS", 60)  # below: flagged content_limited
# Dashboard "run ingestion" button: disabled unless a token is set, and only accepted from this machine.
INGEST_ADMIN_TOKEN = _env("INGEST_ADMIN_TOKEN", "").strip()
INGEST_MAX_ARTICLES = _env_int("INGEST_MAX_ARTICLES", 100)
INGEST_MAX_BROADCASTS = _env_int("INGEST_MAX_BROADCASTS", 10)
# Date-range jobs started from the dashboard: longest range, and most bulletins / articles one job may process.
INGEST_MAX_RANGE_DAYS = _env_int("INGEST_MAX_RANGE_DAYS", 31)
INGEST_MAX_JOB_BROADCASTS = _env_int("INGEST_MAX_JOB_BROADCASTS", 120)
INGEST_MAX_JOB_ARTICLES = _env_int("INGEST_MAX_JOB_ARTICLES", 2500)
# Earliest date a range job may start from (the NewsOnAIR archive is not checked before it).
INGEST_EARLIEST_DATE = _env("INGEST_EARLIEST_DATE", "2025-11-02")
# A running job that has not reported progress for this long is treated as dead (crashed process).
INGEST_STALE_SECONDS = _env_int("INGEST_STALE_SECONDS", 900)
# Bulletin listings discovered from the archive are reused for this long before the archive is scanned again.
CATALOG_MAX_AGE_HOURS = _env_float("CATALOG_MAX_AGE_HOURS", 6.0)

# --- Reporting window --------------------------------------------------------
# The period the project reports on. This is NOT the data coverage: coverage is always
# computed from the dates of successfully indexed records (article publication date,
# bulletin broadcast date, both in TIMEZONE).
REPORT_WINDOW_START = _env("VANI_REPORT_START", "2025-11-02")
REPORT_WINDOW_END = _env("VANI_REPORT_END", "2026-11-02")

# --- RAG query analytics -------------------------------------------------------
# One row per /chat request (timings, filters, retrieved doc ids; never the question text).
QUERY_LOG_ENABLED = _env("VANI_QUERY_LOG", "1") not in ("0", "false", "no")

# --- Logging ---------------------------------------------------------------
LOG_LEVEL = _env("VANI_LOG_LEVEL", "INFO")
LOG_FILE = _env("VANI_LOG_FILE", os.path.join("logs", "vanistream.log"))


def require_stream_url(url: str | None = None) -> str:
    """Return the stream URL to use, or raise a clear error if none is configured."""
    chosen = (url or LIVE_STREAM_URL or "").strip()
    if not chosen:
        raise ValueError(
            "No live stream URL configured. Set LIVE_STREAM_URL in your environment "
            "or .env file (it must be a direct audio endpoint such as an Icecast .mp3 "
            "or HLS .m3u8 URL, not a web page), or pass --url."
        )
    return chosen


def describe() -> dict:
    """Non-secret configuration summary for logs and /health."""
    return {
        "asr": f"{ASR_BACKEND}:{ASR_MODEL} ({ASR_DEVICE}/{ASR_COMPUTE_TYPE})",
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GEMINI_MODEL if GEMINI_API_KEY else "extractive (no GEMINI_API_KEY)",
        "timezone": TIMEZONE,
        "date_anchor": DATE_ANCHOR,
        "live_stream_url_configured": bool(LIVE_STREAM_URL),
        "mmr_threshold": MMR_THRESHOLD,
    }


# Configure logging as soon as config is imported by any entry point.
import tracing  # noqa: E402

tracing.setup_logging(LOG_LEVEL, LOG_FILE)
