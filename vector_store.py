"""
vector_store.py - FAISS + SQLite store for VaniStream.

  FAISS  (faiss_index.bin) : IndexFlatIP over L2-normalised embeddings -> cosine search
  SQLite (news_store.db)   : authoritative metadata / history, one row per segment
  SQLite FTS5 (segments_fts): BM25 keyword index over the same segments (hybrid retrieval)

Every segment row keeps enough metadata to trace an answer back to audio:
  faiss_id, text, genre, date, hour (HHMM int), source_file, source_name,
  source_url, ingest_mode (archive|live), language, audio_start/audio_end
  (seconds into the audio file), ts_start/ts_end (absolute wall-clock ISO
  time, live only), indexed_at.

Safety for long-running / concurrent use:
  - a process-wide lock serialises writes; the index is cached in memory and
    reloaded only when the file on disk changes (e.g. written by a separate
    live-ingest process)
  - the FAISS file is written to a temp file and atomically replaced
  - FAISS is saved before the SQLite commit, so a crash can only leave unused
    vectors, never rows that point at missing vectors
  - schema changes are additive (ALTER TABLE ADD COLUMN); old data is kept

CLI:
  python vector_store.py --stats
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Optional

import config

logging.getLogger("faiss.loader").setLevel(logging.WARNING)
import faiss  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("store")

FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
SQLITE_DB_PATH = config.SQLITE_DB_PATH
EMBEDDING_DIM = config.EMBEDDING_DIM

_write_lock = threading.RLock()
_index_cache: dict[str, tuple[float, faiss.Index]] = {}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    faiss_id INTEGER UNIQUE,
    text TEXT NOT NULL,
    genre TEXT,
    genre_confidence REAL,
    date TEXT,
    hour INTEGER,
    source_file TEXT,
    indexed_at TEXT,
    audio_start REAL,
    audio_end REAL
);
"""

# Columns added after the original schema (migrated in place, old rows keep NULL).
EXTRA_COLUMNS = {
    "source_name": "TEXT",
    "source_url": "TEXT",
    "ingest_mode": "TEXT",
    "language": "TEXT",
    "ts_start": "TEXT",
    "ts_end": "TEXT",
    "article_id": "TEXT",   # set for ingest_mode='article' chunks
    "section": "TEXT",      # article section the chunk came from (Report, Background, ...)
    # Source-document metadata (NewsOnAIR ingestion and later):
    "doc_id": "TEXT",           # stable document id, e.g. noa-post-1948469, noa-bulletin-1949031
    "chunk_index": "INTEGER",   # position of the chunk inside its document
    "content_type": "TEXT",     # article | audio_transcript
    "headline": "TEXT",         # article headline or bulletin title
    "source_category": "TEXT",  # publisher's own category (National, Business, Hourly News, ...)
    "published_at": "TEXT",     # original publication / broadcast time (ISO, IST)
    "active": "INTEGER DEFAULT 1",  # 0 = superseded by a newer version of the same document
    # Taxonomy metadata (taxonomy.py); recomputed in place by `python taxonomy.py --backfill`:
    "genre_id": "TEXT",         # primary genre id (national, sports, finance, ...)
    "genre_tags": "TEXT",       # secondary genre ids, stored as ',a,b,'
    "broadcast_type": "TEXT",   # bulletins only: midday | news-at-nine | hourly | morning | evening | other
}

# Columns added to the bookkeeping tables after they were first created.
TABLE_EXTRA_COLUMNS = {
    "ingestion_jobs": {"stage": "TEXT", "updated_at": "TEXT", "progress_json": "TEXT",
                       "cancel_requested": "INTEGER DEFAULT 0", "pid": "INTEGER", "kind": "TEXT",
                       "date_from": "TEXT", "date_to": "TEXT", "types": "TEXT"},
    "ingestion_state": {"meta_json": "TEXT"},   # what is needed to retry the document (e.g. bulletin audio URL)
    "articles": {"genre_id": "TEXT", "genre_tags": "TEXT", "source_category": "TEXT"},
}

# One row per indexed broadcast (bulletin or live window): the metadata the broadcast browser
# needs without touching the transcript chunks. Rebuilt from the chunks by sync_broadcasts().
CREATE_BROADCASTS_SQL = """
CREATE TABLE IF NOT EXISTS broadcasts (
    source_file TEXT PRIMARY KEY,
    doc_id TEXT,
    bulletin_id TEXT,
    title TEXT,
    broadcast_type TEXT,
    date TEXT,
    time TEXT,
    published_at TEXT,
    language TEXT,
    audio_url TEXT,
    page_url TEXT,
    duration_s REAL,
    segments INTEGER,
    timed_segments INTEGER,
    first_faiss_id INTEGER,
    genres TEXT,
    ingest_mode TEXT,
    status TEXT,
    indexed_at TEXT
);
"""
# Bulletin listings seen on the NewsOnAIR archive (reused instead of re-scanning the archive).
CREATE_CATALOG_SQL = """
CREATE TABLE IF NOT EXISTS bulletin_catalog (
    bulletin_id TEXT PRIMARY KEY,
    title TEXT,
    broadcast_type TEXT,
    language TEXT,
    broadcast_at TEXT,
    date TEXT,
    audio_url TEXT,
    page_url TEXT,
    category TEXT,
    seen_at TEXT
);
"""
# What the source had on a date, recorded only when that date was fully checked
# (an archive scan or a date-range article listing covering the whole day).
CREATE_AVAILABILITY_SQL = """
CREATE TABLE IF NOT EXISTS source_availability (
    date TEXT NOT NULL,
    kind TEXT NOT NULL,              -- article | broadcast
    available INTEGER NOT NULL,
    by_type_json TEXT,
    checked_at TEXT,
    PRIMARY KEY (date, kind)
);
"""
# RAG analytics: one row per answered question. The question text is not stored.
CREATE_QUERY_LOG_SQL = """
CREATE TABLE IF NOT EXISTS query_log (
    request_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    genre_scope TEXT,
    detected_genre TEXT,
    date_from TEXT,
    date_to TEXT,
    question_type TEXT,
    mode TEXT,
    model TEXT,
    retrieval_mode TEXT,
    retrieved_docs TEXT,
    n_chunks INTEGER,
    n_sources INTEGER,
    retrieval_ms REAL,
    generation_ms REAL,
    total_ms REAL,
    cited_count INTEGER,
    has_citations INTEGER,
    insufficient INTEGER,
    error TEXT,
    query_chars INTEGER
);
"""
INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_seg_date_hour ON segments(date, hour)",
    "CREATE INDEX IF NOT EXISTS idx_seg_genre ON segments(genre)",
    "CREATE INDEX IF NOT EXISTS idx_seg_source ON segments(source_file)",
    "CREATE INDEX IF NOT EXISTS idx_seg_article ON segments(article_id)",
    "CREATE INDEX IF NOT EXISTS idx_seg_doc ON segments(doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_seg_type_date ON segments(content_type, date)",
    "CREATE INDEX IF NOT EXISTS idx_seg_genre_id ON segments(genre_id, date)",
    "CREATE INDEX IF NOT EXISTS idx_seg_btype ON segments(broadcast_type, date)",
    "CREATE INDEX IF NOT EXISTS idx_art_date ON articles(date, hour)",
    "CREATE INDEX IF NOT EXISTS idx_art_genre ON articles(genre_id, date)",
    "CREATE INDEX IF NOT EXISTS idx_bc_date ON broadcasts(date, time)",
    "CREATE INDEX IF NOT EXISTS idx_bc_type ON broadcasts(broadcast_type, date)",
    "CREATE INDEX IF NOT EXISTS idx_bc_lang ON broadcasts(language)",
    "CREATE INDEX IF NOT EXISTS idx_state_status ON ingestion_state(status, kind)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_started ON ingestion_jobs(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_cat_date ON bulletin_catalog(date, broadcast_type)",
    "CREATE INDEX IF NOT EXISTS idx_qlog_ts ON query_log(ts)",
]

# Incremental ingestion bookkeeping: one row per source document and one per job.
CREATE_STATE_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_state (
    doc_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- article | broadcast
    source_url TEXT,
    title TEXT,
    source_modified TEXT,            -- publisher's modified time, when available
    content_hash TEXT,
    status TEXT NOT NULL,            -- indexed | failed | skipped
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    chunks INTEGER DEFAULT 0,
    first_seen TEXT,
    last_attempt TEXT,
    indexed_at TEXT,
    job_id TEXT
);
"""
CREATE_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,            -- running | completed | completed_with_errors | failed
    trigger TEXT,                    -- cli | dashboard
    params_json TEXT,
    counts_json TEXT,
    error TEXT
);
"""

# Full text + metadata of each ingested article, so answers can use the whole
# article as context rather than only the matched chunk.
CREATE_ARTICLES_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    headline TEXT NOT NULL,
    genre TEXT,
    date TEXT,
    hour INTEGER,
    source_name TEXT,
    word_count INTEGER,
    full_text TEXT NOT NULL,
    data_json TEXT NOT NULL
);
"""

STOPWORDS = set("""a an and are as at be been but by did do does for from had has have he her his how i in into
is it its me my no not of on or our she so than that the their them then there these they this to up
us was we were what when where which who why will with you your about tell give show news happened
latest recent today yesterday week month any some all please""".split())


def _db_path(p: Optional[str]) -> str:
    return p or SQLITE_DB_PATH


def _idx_path(p: Optional[str]) -> str:
    return p or FAISS_INDEX_PATH


# Databases whose schema was already checked by this process: (abs path, creation time).
# Checking on every connection cost a table scan of the keyword index per request.
_schema_ready: set[tuple[str, float]] = set()


def _schema_key(path: str) -> Optional[tuple[str, float]]:
    try:
        return (os.path.abspath(path), os.stat(path).st_ctime)
    except OSError:
        return None


def _get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open (or create) the SQLite database; the schema is checked once per database per process."""
    path = _db_path(db_path)
    key = _schema_key(path)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    if key is not None and key in _schema_ready:
        return conn
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)
    key = _schema_key(path)
    if key is not None:
        _schema_ready.add(key)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_ARTICLES_SQL)
    conn.execute(CREATE_STATE_SQL)
    conn.execute(CREATE_JOBS_SQL)
    for sql in (CREATE_BROADCASTS_SQL, CREATE_CATALOG_SQL, CREATE_AVAILABILITY_SQL, CREATE_QUERY_LOG_SQL):
        conn.execute(sql)
    for table, extra in TABLE_EXTRA_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, typ in extra.items():
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(segments)").fetchall()}
    for col in ("audio_start", "audio_end"):
        if col not in cols:
            conn.execute(f"ALTER TABLE segments ADD COLUMN {col} REAL")
    migrated = False
    for col, typ in EXTRA_COLUMNS.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE segments ADD COLUMN {col} {typ}")
            migrated = True
    if migrated:
        # Back-fill what can be derived honestly from existing rows.
        conn.execute("UPDATE segments SET ingest_mode = CASE WHEN source_file LIKE '%Live-%' "
                     "THEN 'live' ELSE 'archive' END WHERE ingest_mode IS NULL")
        # Older live rows stored hour as 0-23; archive rows use HHMM (e.g. 800).
        conn.execute("UPDATE segments SET hour = hour * 100 WHERE ingest_mode = 'live' AND hour < 24")
        if conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]:
            logger.info("Migrated SQLite schema in place (added %s); existing rows preserved",
                        ", ".join(c for c in EXTRA_COLUMNS if c not in cols))
    for sql in INDEX_SQL:
        conn.execute(sql)
    _ensure_fts(conn)
    conn.commit()


def _ensure_fts(conn: sqlite3.Connection) -> None:
    """Create the FTS5 keyword index and rebuild it if it is out of sync."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(text, tokenize='porter unicode61')")
    except sqlite3.OperationalError:
        return  # SQLite built without FTS5 -> keyword search disabled
    # Only active chunks belong in the keyword index (retired versions are removed from it).
    n_seg = conn.execute("SELECT COUNT(*) FROM segments WHERE IFNULL(active, 1) = 1").fetchone()[0]
    n_fts = conn.execute("SELECT COUNT(*) FROM segments_fts").fetchone()[0]
    if n_seg != n_fts:
        conn.execute("DELETE FROM segments_fts")
        conn.execute("INSERT INTO segments_fts(rowid, text) SELECT faiss_id, "
                     "CASE WHEN content_type = 'article' THEN IFNULL(headline, '') || char(10) || text ELSE text END "
                     "FROM segments WHERE IFNULL(active, 1) = 1")
        logger.info("Rebuilt keyword index for %d active segments", n_seg)


def has_fts(db_path: Optional[str] = None) -> bool:
    db = _get_db(db_path)
    try:
        return db.execute("SELECT 1 FROM sqlite_master WHERE name='segments_fts'").fetchone() is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# FAISS index handling
# ---------------------------------------------------------------------------

def _load_index(index_path: Optional[str] = None) -> faiss.Index:
    """Return the FAISS index, reusing the in-memory copy while the file is unchanged."""
    path = _idx_path(index_path)
    if not os.path.isfile(path):
        return faiss.IndexFlatIP(EMBEDDING_DIM)
    mtime = os.path.getmtime(path)
    cached = _index_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    for attempt in range(3):
        try:
            index = faiss.read_index(path)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(0.2)  # file being replaced by another process
    _index_cache[path] = (mtime, index)
    return index


# Kept for backward compatibility with older scripts.
_load_or_create_index = _load_index


def _save_index(index: faiss.Index, index_path: Optional[str] = None) -> None:
    path = _idx_path(index_path)
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:  # Windows: reader has the file open briefly
            if attempt == 4:
                raise
            time.sleep(0.2)
    _index_cache[path] = (os.path.getmtime(path), index)


def check_consistency(index_path: Optional[str] = None, db_path: Optional[str] = None) -> dict:
    """Detect rows pointing at vectors that do not exist (e.g. after a crash) and drop them."""
    with _write_lock:
        index = _load_index(index_path)
        db = _get_db(db_path)
        try:
            orphans = db.execute("SELECT COUNT(*) FROM segments WHERE faiss_id >= ?", (index.ntotal,)).fetchone()[0]
            if orphans:
                logger.warning("Removing %d SQLite rows with no FAISS vector (index has %d)", orphans, index.ntotal)
                db.execute("DELETE FROM segments_fts WHERE rowid >= ?", (index.ntotal,))
                db.execute("DELETE FROM segments WHERE faiss_id >= ?", (index.ntotal,))
                db.commit()
            rows = db.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        finally:
            db.close()
        return {"faiss_vectors": index.ntotal, "sqlite_rows": rows, "orphans_removed": orphans}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def add_segments(
    segments: list[dict],
    date: str,
    hour: int,
    source_file: str,
    index_path: Optional[str] = None,
    db_path: Optional[str] = None,
    source_name: Optional[str] = None,
    source_url: Optional[str] = None,
    ingest_mode: str = "archive",
    language: Optional[str] = None,
    doc: Optional[dict] = None,
) -> int:
    """Add novel segments (from nlp_pipeline) to FAISS and SQLite.

    Per-segment keys used: text, genre, genre_confidence, embedding and the
    optional audio_start, audio_end, ts_start, ts_end, hour, date, article_id,
    section and search_text (text for the keyword index when it should carry
    more than `text`, e.g. an article chunk plus its headline).
    Segment-level `date`/`hour` override the call-level values (live windows
    can straddle midnight or an hour boundary).

    `doc` carries document-level metadata stored on every chunk: doc_id,
    content_type, headline, source_category, published_at.

    Returns the list length added; the new faiss_ids are written back into
    each segment dict under "faiss_id".
    """
    doc = dict(doc or {})
    if not segments:
        return 0
    _fill_taxonomy(segments, doc, ingest_mode, db_path)
    with _write_lock:
        index = _load_index(index_path)
        embeddings = np.vstack([seg["embedding"] for seg in segments]).astype(np.float32)
        faiss.normalize_L2(embeddings)
        start_id = index.ntotal
        index.add(embeddings)
        try:
            _save_index(index, index_path)  # FAISS first: see module docstring
        except Exception:
            # Roll back the in-memory copy so ids stay consistent with disk.
            _index_cache.pop(_idx_path(index_path), None)
            raise

        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        db = _get_db(db_path)
        try:
            for i, seg in enumerate(segments):
                fid = start_id + i
                seg["faiss_id"] = fid
                db.execute(
                    "INSERT INTO segments (faiss_id, text, genre, genre_confidence, date, hour, source_file, "
                    "indexed_at, audio_start, audio_end, source_name, source_url, ingest_mode, language, ts_start, ts_end, "
                    "article_id, section, doc_id, chunk_index, content_type, headline, source_category, published_at, "
                    "genre_id, genre_tags, broadcast_type, active) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (fid, seg["text"], seg["genre"], seg.get("genre_confidence"),
                     seg.get("date") or date, int(seg.get("hour") if seg.get("hour") is not None else hour),
                     source_file, now, seg.get("audio_start"), seg.get("audio_end"),
                     source_name, source_url, ingest_mode, language, seg.get("ts_start"), seg.get("ts_end"),
                     seg.get("article_id"), seg.get("section"), seg.get("doc_id") or doc.get("doc_id"),
                     seg.get("chunk_index", i), seg.get("content_type") or doc.get("content_type"),
                     seg.get("headline") or doc.get("headline"), doc.get("source_category"),
                     doc.get("published_at"), seg.get("genre_id") or doc.get("genre_id"),
                     seg.get("genre_tags") or doc.get("genre_tags"), doc.get("broadcast_type")),
                )
                try:
                    db.execute("INSERT INTO segments_fts(rowid, text) VALUES (?, ?)",
                               (fid, seg.get("search_text") or seg["text"]))
                except sqlite3.OperationalError:
                    pass
            db.commit()
        finally:
            db.close()
    logger.info("Stored %d segments -> FAISS ids %d..%d (date=%s, source=%s)",
                len(segments), start_id, start_id + len(segments) - 1, date, os.path.basename(source_file))
    if ingest_mode != "article" and doc.get("content_type") != "article":
        try:
            sync_broadcasts(db_path=db_path, only_source=source_file)
        except Exception as exc:  # the browser row can be rebuilt later; never lose the chunks over it
            logger.warning("Broadcast row update failed for %s: %s", source_file, exc)
    return len(segments)


def _fill_taxonomy(segments: list[dict], doc: dict, ingest_mode: str, db_path: Optional[str]) -> None:
    """Give every chunk a taxonomy genre (and bulletins a broadcast type) when the caller did not.

    Article chunks share the article-level genre (already stored by upsert_article when available);
    transcript chunks are classified one by one, since a bulletin covers many stories.
    """
    import taxonomy
    is_article = ingest_mode == "article" or doc.get("content_type") == "article"
    if not is_article and not doc.get("broadcast_type"):
        doc["broadcast_type"] = taxonomy.broadcast_type_of(doc.get("source_category"))
    if all(seg.get("genre_id") for seg in segments):
        return
    if is_article:
        aid = segments[0].get("article_id") or doc.get("doc_id")
        gid = tags = None
        if aid and os.path.isfile(_db_path(db_path)):
            db = _get_db(db_path)
            try:
                row = db.execute("SELECT genre_id, genre_tags FROM articles WHERE article_id = ?", (aid,)).fetchone()
            finally:
                db.close()
            if row and row["genre_id"]:
                gid, tags = row["genre_id"], row["genre_tags"]
        if gid is None:
            g, t, _ = taxonomy.classify(doc.get("headline") or "", " ".join(s["text"] for s in segments),
                                        doc.get("source_category"), segments[0].get("genre"),
                                        segments[0].get("genre_confidence"), "article")
            gid, tags = g, taxonomy.encode_tags(t)
        for seg in segments:
            if not seg.get("genre_id"):
                seg["genre_id"], seg["genre_tags"] = gid, tags
        return
    for seg in segments:
        if not seg.get("genre_id"):
            g, t, _ = taxonomy.classify("", seg["text"], None, seg.get("genre"), seg.get("genre_confidence"),
                                        "audio_transcript")
            seg["genre_id"], seg["genre_tags"] = g, taxonomy.encode_tags(t)


def wipe(index_path: Optional[str] = None, db_path: Optional[str] = None) -> None:
    """Delete the FAISS index and SQLite DB for full re-indexing (explicit --reindex only)."""
    with _write_lock:
        for p in (_idx_path(index_path), _db_path(db_path)):
            for suffix in ("", "-wal", "-shm"):
                if os.path.isfile(p + suffix):
                    os.remove(p + suffix)
                    logger.info("Removed %s", p + suffix)
        _index_cache.clear()
        _schema_ready.clear()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

SEGMENT_COLUMNS = ("faiss_id, text, genre, genre_confidence, date, hour, source_file, source_name, source_url, "
                   "ingest_mode, language, audio_start, audio_end, ts_start, ts_end, indexed_at, article_id, section, "
                   "doc_id, chunk_index, content_type, headline, source_category, published_at, genre_id, genre_tags, "
                   "broadcast_type")
ACTIVE = "IFNULL(active, 1) = 1"


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    _, has_audio = resolve_audio_path(d.get("source_file") or "")
    d["has_audio"] = bool(has_audio)
    if not d.get("source_name"):
        d["source_name"] = _default_source_name(d.get("source_file") or "")
    return d


def _default_source_name(source_file: str) -> str:
    base = os.path.splitext(os.path.basename(source_file.replace("\\", "/")))[0]
    return f"AIR bulletin {base}" if base else "unknown source"


def metadata_prefilter(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    genre: Optional[str] = None,
    db_path: Optional[str] = None,
    hour_from: Optional[int] = None,
    hour_to: Optional[int] = None,
    source_file: Optional[str] = None,
    source_category: Optional[str] = None,
    content_type: Optional[str] = None,
    broadcast_types: Optional[list[str]] = None,
    language: Optional[str] = None,
) -> list[int]:
    """Return active faiss_ids matching optional date range, HHMM hour range, genre, source,
    publisher category, content type (article | audio_transcript), broadcast types and language.

    `genre` is a taxonomy id or display name (matches the primary genre or a secondary tag),
    or a legacy model label such as "Economy" (matches the `genre` column)."""
    conditions, params = [ACTIVE], []
    if date_from:
        conditions.append("date >= ?"); params.append(date_from)
    if date_to:
        conditions.append("date <= ?"); params.append(date_to)
    if hour_from is not None:
        conditions.append("hour >= ?"); params.append(int(hour_from))
    if hour_to is not None:
        conditions.append("hour <= ?"); params.append(int(hour_to))
    if genre:
        import taxonomy
        gid = taxonomy.genre_id_of(genre)
        if gid:
            conditions.append("(genre_id = ? OR genre_tags LIKE ?)"); params += [gid, f"%,{gid},%"]
        else:
            conditions.append("LOWER(TRIM(genre)) = LOWER(TRIM(?))"); params.append(genre)
    if broadcast_types:
        conditions.append(f"broadcast_type IN ({','.join('?' * len(broadcast_types))})"); params += list(broadcast_types)
    if language:
        conditions.append("LOWER(language) IN (LOWER(?), LOWER(?))"); params += [language, _lang_code(language)]
    if source_file:
        conditions.append("REPLACE(source_file, '\\', '/') = ?"); params.append(source_file.replace("\\", "/"))
    if source_category:
        conditions.append("LOWER(source_category) = LOWER(?)"); params.append(source_category)
    if content_type:
        conditions.append("content_type = ?"); params.append(content_type)
    where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    db = _get_db(db_path)
    try:
        rows = db.execute(f"SELECT faiss_id FROM segments{where_sql}", params).fetchall()
    finally:
        db.close()
    return [row["faiss_id"] for row in rows]


def get_segments(faiss_ids: list[int], db_path: Optional[str] = None) -> dict[int, dict]:
    """Fetch full metadata rows for a list of faiss_ids (chunked to respect SQLite limits)."""
    out: dict[int, dict] = {}
    if not faiss_ids:
        return out
    db = _get_db(db_path)
    try:
        ids = [int(i) for i in faiss_ids]
        for k in range(0, len(ids), 500):
            part = ids[k:k + 500]
            q = f"SELECT {SEGMENT_COLUMNS} FROM segments WHERE faiss_id IN ({','.join('?' * len(part))})"
            for row in db.execute(q, part).fetchall():
                out[row["faiss_id"]] = _row_to_dict(row)
    finally:
        db.close()
    return out


def semantic_search(
    query_embedding: np.ndarray,
    faiss_ids: Optional[list[int]] = None,
    top_k: int = 5,
    index_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Dense cosine search, optionally restricted to a subset of faiss_ids.

    Returns segment dicts (see SEGMENT_COLUMNS) plus `score` (cosine) and `has_audio`.
    """
    index = _load_index(index_path)
    if index.ntotal == 0:
        return []
    qvec = query_embedding.reshape(1, -1).astype(np.float32).copy()
    faiss.normalize_L2(qvec)

    if faiss_ids is not None:
        valid = [i for i in faiss_ids if 0 <= i < index.ntotal]
        if not valid:
            return []
        k = min(top_k, len(valid))
        id_array = np.array(valid, dtype=np.int64)  # must outlive the selector
        selector = faiss.IDSelectorArray(id_array)
        params = faiss.SearchParameters()
        params.sel = selector
        scores, ids = index.search(qvec, k, params=params)
    else:
        scores, ids = index.search(qvec, min(top_k, index.ntotal))

    hits = [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
    rows = get_segments([i for i, _ in hits], db_path)
    results = []
    for fid, score in hits:
        if fid in rows:
            r = dict(rows[fid])
            r["score"] = score
            results.append(r)
    return results


def _fts_query(text: str) -> str:
    tokens = [t for t in re.findall(r"[0-9a-zA-Zऀ-෿]+", text.lower())
              if t not in STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))


def keyword_search(
    query: str,
    faiss_ids: Optional[list[int]] = None,
    top_k: int = 10,
    db_path: Optional[str] = None,
) -> list[dict]:
    """BM25 keyword search via SQLite FTS5. Returns [{faiss_id, bm25}] (higher = better)."""
    match = _fts_query(query)
    if not match:
        return []
    db = _get_db(db_path)
    try:
        limit = top_k if faiss_ids is None else max(top_k * 50, 500)
        rows = db.execute(
            "SELECT rowid AS fid, bm25(segments_fts) AS score FROM segments_fts "
            "WHERE segments_fts MATCH ? ORDER BY score LIMIT ?", (match, limit)).fetchall()
    except sqlite3.OperationalError as exc:
        logger.debug("Keyword search unavailable: %s", exc)
        return []
    finally:
        db.close()
    allowed = set(faiss_ids) if faiss_ids is not None else None
    out = [{"faiss_id": r["fid"], "bm25": -float(r["score"])} for r in rows
           if allowed is None or r["fid"] in allowed]
    return out[:top_k]


def term_in_corpus(term: str, db_path: Optional[str] = None) -> bool:
    """True when the keyword index has at least one active passage containing `term` (stemmed)."""
    t = re.sub(r"[^0-9a-zA-Z\u0900-\u0DFF]+", "", term)
    if not t:
        return True
    db = _get_db(db_path)
    try:
        return db.execute("SELECT 1 FROM segments_fts WHERE segments_fts MATCH ? LIMIT 1", (f'"{t}"',)).fetchone() is not None
    except sqlite3.OperationalError:
        return True  # no keyword index -> cannot tell, do not block
    finally:
        db.close()


def get_neighbors(faiss_id: int, n: int = 1, db_path: Optional[str] = None) -> list[dict]:
    """Return the segment plus up to n neighbours on each side from the same source (reading order)."""
    db = _get_db(db_path)
    try:
        row = db.execute("SELECT id, source_file FROM segments WHERE faiss_id = ?", (faiss_id,)).fetchone()
        if not row:
            return []
        before = db.execute(f"SELECT {SEGMENT_COLUMNS} FROM segments WHERE source_file = ? AND id < ? AND {ACTIVE} "
                            "ORDER BY id DESC LIMIT ?", (row["source_file"], row["id"], n)).fetchall()
        after = db.execute(f"SELECT {SEGMENT_COLUMNS} FROM segments WHERE source_file = ? AND id >= ? AND {ACTIVE} "
                           "ORDER BY id ASC LIMIT ?", (row["source_file"], row["id"], n + 1)).fetchall()
    finally:
        db.close()
    return [_row_to_dict(r) for r in list(reversed(before)) + list(after)]


def get_transcript(source_file: str, db_path: Optional[str] = None) -> list[dict]:
    """All indexed segments for one source in reading order."""
    db = _get_db(db_path)
    try:
        rows = db.execute(f"SELECT {SEGMENT_COLUMNS} FROM segments WHERE REPLACE(source_file, '\\', '/') = ? "
                          f"AND {ACTIVE} ORDER BY id", (source_file.replace("\\", "/"),)).fetchall()
    finally:
        db.close()
    return [_row_to_dict(r) for r in rows]


def get_source_of(faiss_id: int, db_path: Optional[str] = None) -> Optional[str]:
    db = _get_db(db_path)
    try:
        row = db.execute("SELECT source_file FROM segments WHERE faiss_id = ?", (faiss_id,)).fetchone()
    finally:
        db.close()
    return row["source_file"] if row else None


def list_sources(limit: int = 200, db_path: Optional[str] = None) -> list[dict]:
    """One entry per broadcast transcript, newest first (articles are listed by list_articles)."""
    db = _get_db(db_path)
    try:
        rows = db.execute(
            "SELECT source_file, MIN(date) AS date, MIN(hour) AS hour, MAX(source_name) AS source_name, "
            "MAX(ingest_mode) AS ingest_mode, COUNT(*) AS segments, MIN(ts_start) AS ts_start, "
            "MIN(faiss_id) AS first_faiss_id, MAX(headline) AS headline, MAX(source_url) AS source_url, "
            "MAX(published_at) AS published_at FROM segments WHERE IFNULL(ingest_mode, '') != 'article' "
            f"AND {ACTIVE} GROUP BY source_file "
            "ORDER BY date DESC, hour DESC, MIN(id) DESC LIMIT ?", (limit,)).fetchall()
    finally:
        db.close()
    out = []
    for r in rows:
        d = dict(r)
        d["has_audio"] = resolve_audio_path(d["source_file"] or "")[1]
        d["source_name"] = d["source_name"] or _default_source_name(d["source_file"] or "")
        out.append(d)
    return out


def upsert_article(article: dict, full_text: str, db_path: Optional[str] = None) -> None:
    """Store the complete article (all sections + metadata) used for answer context."""
    if not article.get("genre_id"):
        import taxonomy
        gid, tags, _ = taxonomy.classify(article.get("headline") or "", full_text, article.get("source_category"),
                                         article.get("genre"), article.get("genre_confidence"), "article")
        article = dict(article, genre_id=gid, genre_tags=taxonomy.encode_tags(tags))
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.execute(
                "INSERT OR REPLACE INTO articles (article_id, headline, genre, date, hour, source_name, word_count, "
                "full_text, data_json, genre_id, genre_tags, source_category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (article["id"], article["headline"], article.get("genre"), article.get("published"),
                 article.get("hour"), article.get("source_name"), len(full_text.split()), full_text,
                 json.dumps(article, ensure_ascii=False), article.get("genre_id"), article.get("genre_tags"),
                 article.get("source_category")))
            db.commit()
        finally:
            db.close()


def list_articles(db_path: Optional[str] = None) -> list[dict]:
    """All stored articles (structured data), newest first."""
    if not os.path.isfile(_db_path(db_path)):
        return []
    db = _get_db(db_path)
    try:
        rows = db.execute("SELECT data_json FROM articles ORDER BY date DESC, hour DESC").fetchall()
    finally:
        db.close()
    return [json.loads(r["data_json"]) for r in rows]


LANG_NAMES = {"en": "English", "hi": "Hindi"}


def _lang_code(language: str) -> str:
    low = (language or "").lower()
    return next((c for c, n in LANG_NAMES.items() if n.lower() == low), low)


def _lang_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return LANG_NAMES.get(value.lower(), value)


def _article_where(genre: Optional[str] = None, category: Optional[str] = None, q: Optional[str] = None,
                   date_from: Optional[str] = None, date_to: Optional[str] = None) -> tuple[str, list]:
    import taxonomy
    cond, params = ["1 = 1"], []
    if genre:
        gid = taxonomy.genre_id_of(genre)
        if gid:
            cond.append("(genre_id = ? OR genre_tags LIKE ?)"); params += [gid, f"%,{gid},%"]
        else:
            cond.append("LOWER(genre) = LOWER(?)"); params.append(genre)
    if category:
        cond.append("LOWER(source_category) = LOWER(?)"); params.append(category)
    if date_from:
        cond.append("date >= ?"); params.append(date_from)
    if date_to:
        cond.append("date <= ?"); params.append(date_to)
    for term in (q or "").lower().split()[:8]:
        cond.append("(LOWER(headline) LIKE ? OR LOWER(full_text) LIKE ?)"); params += [f"%{term}%"] * 2
    return " AND ".join(cond), params


def list_articles_page(genre: Optional[str] = None, category: Optional[str] = None, q: Optional[str] = None,
                       date_from: Optional[str] = None, date_to: Optional[str] = None, limit: int = 24,
                       offset: int = 0, db_path: Optional[str] = None) -> tuple[list[dict], int]:
    """One page of stored articles (structured data), newest first, plus the total number of matches."""
    if not os.path.isfile(_db_path(db_path)):
        return [], 0
    where, params = _article_where(genre, category, q, date_from, date_to)
    db = _get_db(db_path)
    try:
        total = db.execute(f"SELECT COUNT(*) FROM articles WHERE {where}", params).fetchone()[0]
        rows = db.execute(f"SELECT data_json, genre_id, genre_tags FROM articles WHERE {where} "
                          "ORDER BY date DESC, hour DESC, article_id DESC LIMIT ? OFFSET ?",
                          params + [int(limit), int(offset)]).fetchall()
    finally:
        db.close()
    out = []
    for r in rows:
        a = json.loads(r["data_json"])
        a["genre_id"], a["genre_tags"] = r["genre_id"], r["genre_tags"]
        out.append(a)
    return out, total


def get_article_data(article_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """One stored article (structured data with taxonomy fields), or None."""
    row = get_article(article_id, db_path)
    if not row:
        return None
    a = row["data"]
    a["genre_id"], a["genre_tags"] = row.get("genre_id"), row.get("genre_tags")
    return a


def related_articles(article: dict, limit: int = 3, db_path: Optional[str] = None) -> list[dict]:
    """Newest other articles sharing the primary genre (taxonomy id, or the legacy label before backfill)."""
    db = _get_db(db_path)
    try:
        if article.get("genre_id"):
            rows = db.execute("SELECT data_json FROM articles WHERE genre_id = ? AND article_id != ? "
                              "ORDER BY date DESC, hour DESC LIMIT ?",
                              (article["genre_id"], article["id"], limit)).fetchall()
        else:
            rows = db.execute("SELECT data_json FROM articles WHERE genre = ? AND article_id != ? "
                              "ORDER BY date DESC, hour DESC LIMIT ?",
                              (article.get("genre"), article["id"], limit)).fetchall()
    finally:
        db.close()
    return [json.loads(r["data_json"]) for r in rows]


def article_facets(db_path: Optional[str] = None) -> dict:
    """Filter values that exist among stored articles: genre ids, publisher categories, date span."""
    import taxonomy
    db = _get_db(db_path)
    try:
        genres = {r[0]: r[1] for r in db.execute(
            "SELECT IFNULL(genre_id, 'other'), COUNT(*) FROM articles GROUP BY 1")}
        cats = {r[0]: r[1] for r in db.execute(
            "SELECT source_category, COUNT(*) FROM articles WHERE source_category IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")}
        span = db.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM articles").fetchone()
        legacy = sorted(r[0] for r in db.execute("SELECT DISTINCT genre FROM articles WHERE genre IS NOT NULL"))
    finally:
        db.close()
    return {"genres": [{"id": g, "name": taxonomy.genre_name(g), "count": genres[g]}
                       for g in taxonomy.GENRES if g in genres],
            "categories": cats, "date_range": {"from": span[0], "to": span[1]}, "total": span[2],
            "legacy_genres": legacy}


# ---------------------------------------------------------------------------
# Broadcasts (one row per indexed bulletin / live window)
# ---------------------------------------------------------------------------

def _audio_duration(path: str) -> Optional[float]:
    """Length of an audio file in seconds, read from its header (None if unreadable)."""
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                return round(c.duration / 1_000_000, 1)
            st = c.streams.audio[0]
            if st.duration and st.time_base:
                return round(float(st.duration * st.time_base), 1)
    except Exception:
        return None
    return None


def sync_broadcasts(db_path: Optional[str] = None, only_source: Optional[str] = None) -> int:
    """(Re)build `broadcasts` rows from the active transcript chunks.

    Keeps values that cannot be derived from chunks (audio URL, duration) from the existing row
    or from the ingestion state recorded when the bulletin was ingested.
    """
    import taxonomy
    with _write_lock:
        db = _get_db(db_path)
        try:
            cond, params = f"IFNULL(content_type, '') != 'article' AND IFNULL(ingest_mode, '') != 'article' AND {ACTIVE}", []
            if only_source:
                cond += " AND source_file = ?"; params.append(only_source)
            rows = db.execute(
                "SELECT source_file, MAX(doc_id) doc_id, MAX(headline) headline, MAX(source_category) cat, "
                "MAX(broadcast_type) btype, MIN(date) date, MIN(hour) hour, MAX(published_at) published_at, "
                "MAX(language) language, MAX(source_url) page_url, MAX(source_name) source_name, COUNT(*) n, "
                "SUM(audio_start IS NOT NULL) timed, MAX(audio_end) last_end, MIN(faiss_id) first_id, "
                f"MAX(ingest_mode) mode, MAX(indexed_at) indexed_at FROM segments WHERE {cond} GROUP BY source_file",
                params).fetchall()
            existing = {r["source_file"]: dict(r) for r in db.execute("SELECT * FROM broadcasts")}
            states = {r["doc_id"]: dict(r) for r in db.execute(
                "SELECT doc_id, status, meta_json FROM ingestion_state WHERE kind = 'broadcast'")}
            genre_rows = db.execute(
                f"SELECT source_file, genre_id, COUNT(*) n FROM segments WHERE {cond} AND genre_id IS NOT NULL "
                "AND genre_id != 'other' GROUP BY source_file, genre_id ORDER BY source_file, n DESC", params).fetchall()
            top_genres: dict[str, list[str]] = {}
            for g in genre_rows:
                top_genres.setdefault(g["source_file"], [])
                if len(top_genres[g["source_file"]]) < 3:
                    top_genres[g["source_file"]].append(g["genre_id"])
            out = []
            for r in rows:
                sf = r["source_file"]
                old = existing.get(sf, {})
                st = states.get(r["doc_id"] or "", {})
                meta = json.loads(st.get("meta_json") or "{}")
                title = meta.get("title") or r["cat"] or r["source_name"] or _default_source_name(sf)
                btype = r["btype"] or taxonomy.broadcast_type_of(meta.get("title") or r["cat"])
                hour = r["hour"] or 0
                audio_path, has_audio = resolve_audio_path(sf)
                duration = old.get("duration_s") or (_audio_duration(audio_path) if has_audio else None)
                out.append((sf, r["doc_id"], meta.get("bulletin_id") or (r["doc_id"] or "").replace("noa-bulletin-", "") or None,
                            title, btype, r["date"], f"{hour // 100:02d}:{hour % 100:02d}", r["published_at"],
                            _lang_name(meta.get("language") or r["language"]) or "English",
                            meta.get("audio_url") or old.get("audio_url"), r["page_url"] or meta.get("page_url"),
                            duration, r["n"], r["timed"], r["first_id"],
                            taxonomy.encode_tags(top_genres.get(sf, [])), r["mode"],
                            st.get("status") or "indexed", r["indexed_at"]))
            if not only_source:
                live = {r[0] for r in rows}
                gone = [sf for sf in existing if sf not in live]
                db.executemany("DELETE FROM broadcasts WHERE source_file = ?", [(g,) for g in gone])
            db.executemany(
                "INSERT OR REPLACE INTO broadcasts (source_file, doc_id, bulletin_id, title, broadcast_type, date, time, "
                "published_at, language, audio_url, page_url, duration_s, segments, timed_segments, first_faiss_id, "
                "genres, ingest_mode, status, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                out)
            db.commit()
        finally:
            db.close()
    return len(out)


def _broadcast_where(types: Optional[list[str]] = None, genre: Optional[str] = None, language: Optional[str] = None,
                     date_from: Optional[str] = None, date_to: Optional[str] = None,
                     q: Optional[str] = None, db: Optional[sqlite3.Connection] = None) -> tuple[str, list]:
    import taxonomy
    cond, params = ["1 = 1"], []
    if types:
        cond.append(f"broadcast_type IN ({','.join('?' * len(types))})"); params += list(types)
    if genre:
        gid = taxonomy.genre_id_of(genre) or genre
        # A bulletin covers many stories: it matches a genre when any of its chunks has that genre.
        cond.append(f"source_file IN (SELECT source_file FROM segments WHERE {ACTIVE} AND "
                    "IFNULL(content_type, '') != 'article' AND (genre_id = ? OR genre_tags LIKE ?))")
        params += [gid, f"%,{gid},%"]
    if language:
        cond.append("LOWER(language) = LOWER(?)"); params.append(_lang_name(language))
    if date_from:
        cond.append("date >= ?"); params.append(date_from)
    if date_to:
        cond.append("date <= ?"); params.append(date_to)
    if q and q.strip():
        terms = q.strip()
        like = f"%{terms.lower()}%"
        sub = ""
        fts = _fts_query(terms)
        if fts and db is not None and db.execute("SELECT 1 FROM sqlite_master WHERE name='segments_fts'").fetchone():
            sub = (" OR source_file IN (SELECT s.source_file FROM segments_fts f JOIN segments s ON s.faiss_id = f.rowid "
                   "WHERE segments_fts MATCH ? AND IFNULL(s.content_type, '') != 'article')")
        cond.append(f"(LOWER(title) LIKE ? OR date LIKE ? OR LOWER(IFNULL(broadcast_type, '')) LIKE ?{sub})")
        params += [like, like, like] + ([fts] if sub else [])
    return " AND ".join(cond), params


def list_broadcast_days(types: Optional[list[str]] = None, genre: Optional[str] = None,
                        language: Optional[str] = None, date_from: Optional[str] = None,
                        date_to: Optional[str] = None, q: Optional[str] = None, days: int = 7, offset: int = 0,
                        order: str = "desc", db_path: Optional[str] = None) -> dict:
    """Broadcasts grouped by broadcast date, paginated by date (no transcripts)."""
    import taxonomy
    empty = {"days": [], "total_days": 0, "total_broadcasts": 0, "next_offset": None}
    if not os.path.isfile(_db_path(db_path)):
        return empty
    order = "ASC" if order == "asc" else "DESC"
    db = _get_db(db_path)
    try:
        where, params = _broadcast_where(types, genre, language, date_from, date_to, q, db)
        totals = db.execute(f"SELECT COUNT(*), COUNT(DISTINCT date) FROM broadcasts WHERE {where}", params).fetchone()
        dates = [r[0] for r in db.execute(f"SELECT DISTINCT date FROM broadcasts WHERE {where} ORDER BY date {order} "
                                          "LIMIT ? OFFSET ?", params + [int(days), int(offset)])]
        rows = db.execute(f"SELECT * FROM broadcasts WHERE {where} AND date IN ({','.join('?' * len(dates))}) "
                          f"ORDER BY date {order}, time DESC", params + dates).fetchall() if dates else []
    finally:
        db.close()
    by_date: dict[str, list] = {d: [] for d in dates}
    for r in rows:
        b = dict(r)
        b["broadcast_type_name"] = taxonomy.broadcast_type_name(b["broadcast_type"])
        b["genres"] = [{"id": g, "name": taxonomy.genre_name(g)} for g in taxonomy.decode_tags(b["genres"])]
        b["has_audio"] = resolve_audio_path(b["source_file"])[1]
        by_date[b["date"]].append(b)
    nxt = offset + len(dates)
    return {"days": [{"date": d, "count": len(v), "broadcasts": v} for d, v in by_date.items()],
            "total_days": totals[1], "total_broadcasts": totals[0],
            "next_offset": nxt if nxt < totals[1] else None}


def list_broadcasts_flat(types: Optional[list[str]] = None, genre: Optional[str] = None,
                         language: Optional[str] = None, date_from: Optional[str] = None,
                         date_to: Optional[str] = None, q: Optional[str] = None, limit: int = 25, offset: int = 0,
                         order: str = "desc", db_path: Optional[str] = None) -> dict:
    """All broadcasts as one list (newest first), paginated by row (no transcripts)."""
    import taxonomy
    if not os.path.isfile(_db_path(db_path)):
        return {"broadcasts": [], "total": 0, "next_offset": None}
    order = "ASC" if order == "asc" else "DESC"
    db = _get_db(db_path)
    try:
        where, params = _broadcast_where(types, genre, language, date_from, date_to, q, db)
        total = db.execute(f"SELECT COUNT(*) FROM broadcasts WHERE {where}", params).fetchone()[0]
        rows = db.execute(f"SELECT * FROM broadcasts WHERE {where} ORDER BY date {order}, time {order} LIMIT ? OFFSET ?",
                          params + [int(limit), int(offset)]).fetchall()
    finally:
        db.close()
    out = []
    for r in rows:
        b = dict(r)
        b["broadcast_type_name"] = taxonomy.broadcast_type_name(b["broadcast_type"])
        b["genres"] = [{"id": g, "name": taxonomy.genre_name(g)} for g in taxonomy.decode_tags(b["genres"])]
        b["has_audio"] = resolve_audio_path(b["source_file"])[1]
        out.append(b)
    nxt = offset + len(out)
    return {"broadcasts": out, "total": total, "next_offset": nxt if nxt < total else None}


def broadcast_facets(db_path: Optional[str] = None) -> dict:
    """Broadcast types, languages and genres present among indexed broadcasts, with counts."""
    import taxonomy
    db = _get_db(db_path)
    try:
        types = {r[0] or "other": r[1] for r in db.execute(
            "SELECT broadcast_type, COUNT(*) FROM broadcasts GROUP BY 1")}
        langs = {r[0] or "unknown": r[1] for r in db.execute("SELECT language, COUNT(*) FROM broadcasts GROUP BY 1")}
        genres = {r[0]: r[1] for r in db.execute(
            f"SELECT genre_id, COUNT(DISTINCT source_file) FROM segments WHERE {ACTIVE} AND "
            "IFNULL(content_type, '') != 'article' AND genre_id IS NOT NULL GROUP BY 1")}
        span = db.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM broadcasts").fetchone()
    finally:
        db.close()
    return {"types": [{"id": t, "name": taxonomy.broadcast_type_name(t), "count": types[t]}
                      for t in taxonomy.BROADCAST_TYPES if t in types],
            "languages": langs,
            "genres": [{"id": g, "name": taxonomy.genre_name(g), "count": genres[g]}
                       for g in taxonomy.GENRES if g in genres],
            "date_range": {"from": span[0], "to": span[1]}, "total": span[2]}


def get_broadcast(source_file: str, db_path: Optional[str] = None) -> Optional[dict]:
    import taxonomy
    db = _get_db(db_path)
    try:
        row = db.execute("SELECT * FROM broadcasts WHERE REPLACE(source_file, '\\', '/') = ?",
                         (source_file.replace("\\", "/"),)).fetchone()
    finally:
        db.close()
    if not row:
        return None
    b = dict(row)
    b["broadcast_type_name"] = taxonomy.broadcast_type_name(b["broadcast_type"])
    b["genres"] = [{"id": g, "name": taxonomy.genre_name(g)} for g in taxonomy.decode_tags(b["genres"])]
    return b


# ---------------------------------------------------------------------------
# Bulletin catalog and source availability
# ---------------------------------------------------------------------------

def upsert_catalog(items: list[dict], db_path: Optional[str] = None) -> None:
    import taxonomy
    if not items:
        return
    now = _now_iso()
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.executemany(
                "INSERT OR REPLACE INTO bulletin_catalog (bulletin_id, title, broadcast_type, language, broadcast_at, "
                "date, audio_url, page_url, category, seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(str(b["bulletin_id"]), b["title"], taxonomy.broadcast_type_of(b["title"]), b["language"],
                  b["broadcast_at"], (b["broadcast_at"] or "")[:10], b["audio_url"], b["page_url"],
                  b.get("category"), now) for b in items])
            db.commit()
        finally:
            db.close()


def catalog_range(date_from: str, date_to: str, language: str = "English",
                  db_path: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
    """Catalogued bulletins in a range and the oldest time any date in the range was checked.

    The second value is None unless every date in the range has a recorded archive check.
    """
    db = _get_db(db_path)
    try:
        rows = db.execute("SELECT * FROM bulletin_catalog WHERE date BETWEEN ? AND ? AND LOWER(language) = LOWER(?) "
                          "ORDER BY broadcast_at DESC", (date_from, date_to, language)).fetchall()
        checked = db.execute("SELECT COUNT(*), MIN(checked_at) FROM source_availability WHERE kind = 'broadcast' "
                             "AND date BETWEEN ? AND ?", (date_from, date_to)).fetchone()
    finally:
        db.close()
    n_days = (dt.date.fromisoformat(date_to) - dt.date.fromisoformat(date_from)).days + 1
    items = [{"bulletin_id": r["bulletin_id"], "title": r["title"], "language": r["language"],
              "broadcast_at": r["broadcast_at"], "audio_url": r["audio_url"], "page_url": r["page_url"],
              "category": r["category"]} for r in rows]
    return items, (checked[1] if checked[0] == n_days else None)


def record_availability(kind: str, counts: dict[str, int], by_type: Optional[dict[str, dict]] = None,
                        db_path: Optional[str] = None) -> None:
    """Store what the source had on fully checked dates (counts may be 0: checked, nothing published)."""
    if not counts:
        return
    now = _now_iso()
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.executemany("INSERT OR REPLACE INTO source_availability (date, kind, available, by_type_json, checked_at) "
                           "VALUES (?, ?, ?, ?, ?)",
                           [(d, kind, n, json.dumps((by_type or {}).get(d)) if by_type else None, now)
                            for d, n in counts.items()])
            db.commit()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Query log (RAG analytics)
# ---------------------------------------------------------------------------

def log_query(row: dict, db_path: Optional[str] = None) -> None:
    cols = ["request_id", "ts", "genre_scope", "detected_genre", "date_from", "date_to", "question_type", "mode",
            "model", "retrieval_mode", "retrieved_docs", "n_chunks", "n_sources", "retrieval_ms", "generation_ms",
            "total_ms", "cited_count", "has_citations", "insufficient", "error", "query_chars"]
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.execute(f"INSERT OR REPLACE INTO query_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                       [row.get(c) for c in cols])
            db.commit()
        finally:
            db.close()


def deactivate_doc(doc_id: str, db_path: Optional[str] = None) -> int:
    """Retire every chunk of a document (used before re-indexing an updated version).

    FAISS ids are positional, so vectors are not deleted; the rows are marked
    inactive and removed from the keyword index, and every read path ignores them.
    """
    with _write_lock:
        db = _get_db(db_path)
        try:
            ids = [r[0] for r in db.execute(f"SELECT faiss_id FROM segments WHERE doc_id = ? AND {ACTIVE}", (doc_id,))]
            if ids:
                db.executemany("DELETE FROM segments_fts WHERE rowid = ?", [(i,) for i in ids])
                db.execute("UPDATE segments SET active = 0 WHERE doc_id = ?", (doc_id,))
                db.commit()
        finally:
            db.close()
    return len(ids)


# ---------------------------------------------------------------------------
# Ingestion state and jobs
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_state(doc_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    db = _get_db(db_path)
    try:
        row = db.execute("SELECT * FROM ingestion_state WHERE doc_id = ?", (doc_id,)).fetchone()
    finally:
        db.close()
    return dict(row) if row else None


def set_state(doc_id: str, db_path: Optional[str] = None, **fields) -> None:
    """Insert or update one document's ingestion state (attempts are incremented when given as +1)."""
    with _write_lock:
        db = _get_db(db_path)
        try:
            cur = db.execute("SELECT doc_id FROM ingestion_state WHERE doc_id = ?", (doc_id,)).fetchone()
            fields["last_attempt"] = fields.get("last_attempt") or _now_iso()
            if cur is None:
                fields.setdefault("first_seen", _now_iso())
                cols = ["doc_id"] + list(fields)
                db.execute(f"INSERT INTO ingestion_state ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                           [doc_id] + list(fields.values()))
            else:
                db.execute(f"UPDATE ingestion_state SET {', '.join(k + ' = ?' for k in fields)} WHERE doc_id = ?",
                           list(fields.values()) + [doc_id])
            db.commit()
        finally:
            db.close()


def start_job(job_id: str, trigger: str, params: dict, db_path: Optional[str] = None) -> None:
    now = _now_iso()
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.execute("INSERT INTO ingestion_jobs (job_id, started_at, status, trigger, params_json, stage, updated_at, "
                       "pid, kind, date_from, date_to, types) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (job_id, now, "running", trigger, json.dumps(params), "starting", now, os.getpid(),
                        params.get("kind"), params.get("date_from"), params.get("date_to"),
                        ",".join(params.get("types") or []) or None))
            db.commit()
        finally:
            db.close()


def update_job_progress(job_id: str, stage: str, counts: dict, db_path: Optional[str] = None) -> bool:
    """Record the current stage and counts; returns True when cancellation was requested."""
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.execute("UPDATE ingestion_jobs SET stage = ?, progress_json = ?, updated_at = ? WHERE job_id = ?",
                       (stage, json.dumps(counts), _now_iso(), job_id))
            db.commit()
            row = db.execute("SELECT cancel_requested FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            db.close()
    return bool(row and row[0])


def request_cancel(job_id: str, db_path: Optional[str] = None) -> bool:
    with _write_lock:
        db = _get_db(db_path)
        try:
            cur = db.execute("UPDATE ingestion_jobs SET cancel_requested = 1 WHERE job_id = ? AND status = 'running'",
                             (job_id,))
            db.commit()
        finally:
            db.close()
    return cur.rowcount > 0


def _process_alive(pid: Optional[int]) -> Optional[bool]:
    """True/False when the process can be checked, None when unknown.

    Never uses os.kill on Windows: there any signal other than CTRL events terminates the process.
    """
    if not pid:
        return None
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
            k32.CloseHandle(handle)
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        except Exception:
            return None
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return None


def running_jobs(db_path: Optional[str] = None) -> list[dict]:
    """Jobs marked running, in any process. A job whose process is gone is marked failed.

    Liveness comes from the recorded process id; without one (jobs started by older code),
    a job is only presumed dead after 24 hours without a recorded update.
    """
    if not os.path.isfile(_db_path(db_path)):
        return []
    db = _get_db(db_path)
    try:
        rows = [dict(r) for r in db.execute("SELECT * FROM ingestion_jobs WHERE status = 'running'")]
    finally:
        db.close()
    now = dt.datetime.now(dt.timezone.utc)
    alive = []
    for r in rows:
        seen = r.get("updated_at") or r["started_at"]
        age = (now - dt.datetime.strptime(seen, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)).total_seconds()
        state = _process_alive(r.get("pid"))
        dead = state is False or (state is None and age > max(config.INGEST_STALE_SECONDS, 86400))
        if dead:
            finish_job(r["job_id"], "failed", json.loads(r.get("progress_json") or "{}"),
                       "The ingestion process is no longer running (job marked failed); "
                       "documents indexed before it stopped are kept", db_path)
            continue
        r["params"] = json.loads(r.pop("params_json") or "{}")
        r["counts"] = json.loads(r.pop("progress_json") or "{}")
        alive.append(r)
    return alive


def finish_job(job_id: str, status: str, counts: dict, error: Optional[str] = None,
               db_path: Optional[str] = None) -> None:
    with _write_lock:
        db = _get_db(db_path)
        try:
            db.execute("UPDATE ingestion_jobs SET finished_at = ?, status = ?, counts_json = ?, error = ?, stage = ?, "
                       "updated_at = ? WHERE job_id = ?",
                       (_now_iso(), status, json.dumps(counts), error, status, _now_iso(), job_id))
            db.commit()
        finally:
            db.close()


def list_jobs(limit: int = 10, db_path: Optional[str] = None) -> list[dict]:
    if not os.path.isfile(_db_path(db_path)):
        return []
    db = _get_db(db_path)
    try:
        rows = db.execute("SELECT * FROM ingestion_jobs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    finally:
        db.close()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d.pop("params_json") or "{}")
        progress = json.loads(d.pop("progress_json", None) or "{}")
        d["counts"] = json.loads(d.pop("counts_json") or "null") or progress
        out.append(d)
    return out


def indexed_broadcasts_per_date(db_path: Optional[str] = None) -> dict[str, int]:
    """Number of indexed audio documents per broadcast date."""
    if not os.path.isfile(_db_path(db_path)):
        return {}
    db = _get_db(db_path)
    try:
        rows = db.execute(f"SELECT date, COUNT(DISTINCT doc_id) FROM segments WHERE content_type = 'audio_transcript' "
                          f"AND {ACTIVE} GROUP BY date").fetchall()
    finally:
        db.close()
    return {r[0]: r[1] for r in rows}


def get_dataset_stats(db_path: Optional[str] = None) -> dict:
    """Real counts for the dashboard and home page (active rows only)."""
    empty = {"articles": 0, "broadcasts": 0, "chunks": 0, "article_chunks": 0, "transcript_chunks": 0,
             "genres": {}, "source_categories": {}, "date_range": {"from": None, "to": None},
             "live_segments": 0, "documents": {}, "sources": {}, "last_indexed_at": None}
    if not os.path.isfile(_db_path(db_path)):
        return empty
    db = _get_db(db_path)
    try:
        q = lambda sql, *a: db.execute(sql, a).fetchone()[0]  # noqa: E731
        out = dict(empty)
        out["articles"] = q(f"SELECT COUNT(DISTINCT article_id) FROM segments WHERE ingest_mode = 'article' AND {ACTIVE}")
        out["broadcasts"] = q(f"SELECT COUNT(DISTINCT source_file) FROM segments WHERE IFNULL(ingest_mode, '') != 'article' "
                              f"AND {ACTIVE}")
        out["chunks"] = q(f"SELECT COUNT(*) FROM segments WHERE {ACTIVE}")
        out["article_chunks"] = q(f"SELECT COUNT(*) FROM segments WHERE ingest_mode = 'article' AND {ACTIVE}")
        out["transcript_chunks"] = out["chunks"] - out["article_chunks"]
        out["live_segments"] = q(f"SELECT COUNT(*) FROM segments WHERE ingest_mode = 'live' AND {ACTIVE}")
        out["genres"] = {r[0] or "Other": r[1] for r in db.execute(
            f"SELECT genre, COUNT(DISTINCT IFNULL(doc_id, source_file)) FROM segments WHERE {ACTIVE} "
            "GROUP BY genre ORDER BY 2 DESC")}
        out["source_categories"] = {r[0]: r[1] for r in db.execute(
            f"SELECT source_category, COUNT(DISTINCT doc_id) FROM segments WHERE {ACTIVE} AND source_category IS NOT NULL "
            "GROUP BY source_category ORDER BY 2 DESC")}
        out["sources"] = {r[0] or "unknown": r[1] for r in db.execute(
            f"SELECT source_name, COUNT(DISTINCT IFNULL(doc_id, source_file)) FROM segments WHERE {ACTIVE} "
            "GROUP BY source_name ORDER BY 2 DESC LIMIT 10")}
        dr = db.execute(f"SELECT MIN(date), MAX(date), MAX(indexed_at) FROM segments WHERE {ACTIVE}").fetchone()
        out["date_range"] = {"from": dr[0], "to": dr[1]}
        out["last_indexed_at"] = dr[2]
        out["documents"] = {r[0]: r[1] for r in db.execute(
            "SELECT status, COUNT(*) FROM ingestion_state GROUP BY status")}
    finally:
        db.close()
    return out


def get_article(article_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Full stored article: headline, metadata, full_text and the original structured data."""
    db = _get_db(db_path)
    try:
        row = db.execute("SELECT * FROM articles WHERE article_id = ?", (article_id,)).fetchone()
    finally:
        db.close()
    if not row:
        return None
    d = dict(row)
    d["data"] = json.loads(d.pop("data_json"))
    return d


def browse(faiss_ids: list[int], limit: int = 8, db_path: Optional[str] = None) -> list[dict]:
    """Chronological sample of segments within a filter (for 'what happened yesterday?').

    Picks evenly spaced segments across the time-ordered set so the answer
    covers the whole period rather than just its first minutes.
    """
    if not faiss_ids:
        return []
    rows = get_segments(faiss_ids, db_path)
    ordered = sorted(rows.values(), key=lambda r: (r["date"] or "", r["hour"] or 0, r["faiss_id"]))
    if len(ordered) <= limit:
        return ordered
    step = len(ordered) / limit
    return [ordered[int(i * step)] for i in range(limit)]


def latest_date(faiss_ids: Optional[list[int]] = None, db_path: Optional[str] = None) -> Optional[str]:
    rows = get_segments(faiss_ids, db_path) if faiss_ids is not None else None
    if rows is not None:
        dates = [r["date"] for r in rows.values() if r["date"]]
        return max(dates) if dates else None
    return get_date_range(db_path)[1]


def get_all_embeddings(index_path: Optional[str] = None) -> Optional[np.ndarray]:
    """All stored embeddings as an (N, dim) matrix, or None when empty."""
    index = _load_index(index_path)
    if index.ntotal == 0:
        return None
    return index.reconstruct_n(0, index.ntotal)


def get_pool(content_type: Optional[str] = None, index_path: Optional[str] = None,
             db_path: Optional[str] = None) -> tuple[Optional[np.ndarray], list[str]]:
    """Embeddings + texts of active segments (optionally one content type) for MMR dedup."""
    if not os.path.isfile(_db_path(db_path)):
        return None, []
    db = _get_db(db_path)
    try:
        q = f"SELECT faiss_id, text FROM segments WHERE {ACTIVE}"
        params: list = []
        if content_type:
            q += " AND content_type = ?"
            params.append(content_type)
        rows = db.execute(q + " ORDER BY faiss_id", params).fetchall()
    finally:
        db.close()
    index = _load_index(index_path)
    rows = [r for r in rows if r["faiss_id"] < index.ntotal]
    if not rows:
        return None, []
    return np.vstack([index.reconstruct(int(r["faiss_id"])) for r in rows]), [r["text"] for r in rows]


def get_all_texts(db_path: Optional[str] = None) -> list[str]:
    """All stored segment texts ordered by faiss_id."""
    if not os.path.isfile(_db_path(db_path)):
        return []
    db = _get_db(db_path)
    try:
        rows = db.execute("SELECT text FROM segments ORDER BY faiss_id ASC").fetchall()
    finally:
        db.close()
    return [r["text"] for r in rows]


def get_recent_pool(hours: int, index_path: Optional[str] = None,
                    db_path: Optional[str] = None) -> tuple[Optional[np.ndarray], list[str]]:
    """Embeddings + texts of segments indexed within the last `hours` (live dedup window)."""
    if not os.path.isfile(_db_path(db_path)):
        return None, []
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db = _get_db(db_path)
    try:
        rows = db.execute("SELECT faiss_id, text FROM segments WHERE indexed_at >= ? ORDER BY faiss_id",
                          (cutoff,)).fetchall()
    finally:
        db.close()
    index = _load_index(index_path)
    rows = [r for r in rows if r["faiss_id"] < index.ntotal]
    if not rows:
        return None, []
    embs = np.vstack([index.reconstruct(int(r["faiss_id"])) for r in rows])
    return embs, [r["text"] for r in rows]


def get_stats(index_path: Optional[str] = None, db_path: Optional[str] = None) -> dict:
    """Summary statistics: totals, genre breakdown, date range, live vs archive counts."""
    index = _load_index(index_path)
    db = _get_db(db_path)
    try:
        genres = {r["genre"]: r["cnt"] for r in
                  db.execute(f"SELECT genre, COUNT(*) AS cnt FROM segments WHERE {ACTIVE} GROUP BY genre ORDER BY cnt DESC")}
        modes = {r["m"] or "archive": r["cnt"] for r in
                 db.execute(f"SELECT ingest_mode AS m, COUNT(*) AS cnt FROM segments WHERE {ACTIVE} GROUP BY ingest_mode")}
        dr = db.execute("SELECT MIN(date) AS min_d, MAX(date) AS max_d, COUNT(DISTINCT CASE WHEN "
                        "IFNULL(ingest_mode, '') != 'article' THEN source_file END) AS n_src FROM segments "
                        f"WHERE {ACTIVE}").fetchone()
        n_articles = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        n_rows = db.execute(f"SELECT COUNT(*) FROM segments WHERE {ACTIVE}").fetchone()[0]
    finally:
        db.close()
    return {
        "total_segments": n_rows,
        "faiss_vectors": index.ntotal,
        "sources": dr["n_src"],
        "articles": n_articles,
        "genres": genres,
        "by_mode": modes,
        "date_range": {"from": dr["min_d"], "to": dr["max_d"]},
    }


def get_live_stats(today: str, db_path: Optional[str] = None) -> dict:
    """Counts used by /health to show whether live ingestion is producing data."""
    if not os.path.isfile(_db_path(db_path)):
        return {"segments_today": 0, "live_segments_today": 0, "last_segment_id": None, "last_indexed_at": None}
    db = _get_db(db_path)
    try:
        r = db.execute(
            "SELECT SUM(date = ?) AS today, SUM(date = ? AND ingest_mode = 'live') AS live_today, "
            "MAX(faiss_id) AS last_id, MAX(indexed_at) AS last_at FROM segments", (today, today)).fetchone()
    finally:
        db.close()
    return {"segments_today": r["today"] or 0, "live_segments_today": r["live_today"] or 0,
            "last_segment_id": r["last_id"], "last_indexed_at": r["last_at"]}


def get_date_range(db_path: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """(MIN(date), MAX(date)) of the corpus, or (None, None) when empty."""
    if not os.path.isfile(_db_path(db_path)):
        return (None, None)
    db = _get_db(db_path)
    try:
        row = db.execute(f"SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM segments WHERE {ACTIVE}").fetchone()
    finally:
        db.close()
    if row and row["min_d"] is not None:
        return (row["min_d"], row["max_d"])
    return (None, None)


# ---------------------------------------------------------------------------
# Audio file resolution
# ---------------------------------------------------------------------------

_metadata_audio_cache: Optional[dict[str, str]] = None
_metadata_cache_mtime: float = 0.0


def _load_metadata_audio_map(metadata_csv: str) -> dict[str, str]:
    """transcript_file -> audio_file mapping from metadata.csv (cached by mtime)."""
    global _metadata_audio_cache, _metadata_cache_mtime
    if not os.path.isfile(metadata_csv):
        return {}
    try:
        mtime = os.path.getmtime(metadata_csv)
        if _metadata_audio_cache is not None and mtime == _metadata_cache_mtime:
            return _metadata_audio_cache
        mapping: dict[str, str] = {}
        with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t_file = (row.get("transcript_file") or "").strip().replace("\\", "/")
                a_file = (row.get("audio_file") or "").strip().replace("\\", "/")
                if t_file and a_file:
                    mapping[t_file] = a_file
        _metadata_audio_cache, _metadata_cache_mtime = mapping, mtime
        return mapping
    except Exception:
        return {}


def resolve_audio_path(
    source_file: str,
    out_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
) -> tuple[str, bool]:
    """Resolve a transcript source_file to its audio file.

    Prefers the audio_file column in metadata.csv; otherwise swaps
    'text/'->'audio/' and '.txt'->'.mp3'. Returns (path, exists_on_disk).
    Never substitutes placeholder audio.
    """
    if not source_file:
        return ("", False)
    out_dir = out_dir or config.OUTPUT_DIR
    metadata_csv = metadata_csv or os.path.join(out_dir, "metadata.csv")
    norm_source = source_file.strip().replace("\\", "/")
    audio_rel = _load_metadata_audio_map(metadata_csv).get(norm_source, "")
    if not audio_rel:
        derived = norm_source
        if derived.startswith("text/"):
            derived = "audio/" + derived[5:]
        if derived.endswith(".txt"):
            derived = derived[:-4] + ".mp3"
        audio_rel = derived
    full_path = os.path.normpath(os.path.join(out_dir, audio_rel)).replace("\\", "/")
    return (full_path, os.path.isfile(full_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="FAISS + SQLite vector store management.")
    parser.add_argument("--stats", action="store_true", help="Print segment count, genres, date range")
    parser.add_argument("--check", action="store_true", help="Check FAISS/SQLite consistency")
    args = parser.parse_args()
    if args.stats:
        print(json.dumps(get_stats(), indent=2, default=str))
    elif args.check:
        print(json.dumps(check_consistency(), indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
