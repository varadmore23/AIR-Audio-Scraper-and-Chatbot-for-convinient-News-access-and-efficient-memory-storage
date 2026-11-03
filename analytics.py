"""
analytics.py - coverage, KPIs and chart data for the analytics dashboard, computed from the database.

Every number comes from SQLite (segments, articles, broadcasts, ingestion_state, ingestion_jobs,
source_availability, query_log). Nothing is estimated or hard-coded; a metric with no recorded
data is returned as None so the UI can say "Not yet measured".

Dates
  Content dates are the article publication date and the bulletin broadcast date as stored in
  `segments.date` (calendar date in config.TIMEZONE, Asia/Kolkata by default). Job and query
  timestamps are stored in UTC and converted to that zone before being bucketed by day.

Coverage
  Actual coverage = earliest and latest date with at least one *active indexed chunk* (a document
  that failed or was never indexed has no active chunk, a superseded version is inactive). The
  configured reporting window (config.REPORT_WINDOW_START / END) is reported separately.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from typing import Optional

import config
import taxonomy
import vector_store
from vector_store import ACTIVE

_DATE_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
JOB_STATUSES = ("completed", "completed_with_errors", "failed", "cancelled", "running")


class Filters:
    """Dashboard filters. Unset values mean 'all'."""

    def __init__(self, date_from: Optional[str] = None, date_to: Optional[str] = None, content: str = "all",
                 types: Optional[list[str]] = None, genre: Optional[str] = None, language: Optional[str] = None,
                 status: Optional[str] = None, interval: str = "day"):
        self.date_from = date_from or config.REPORT_WINDOW_START
        self.date_to = date_to or config.REPORT_WINDOW_END
        for d in (self.date_from, self.date_to):
            dt.date.fromisoformat(d)  # ValueError on a malformed date
        if self.date_from > self.date_to:
            raise ValueError("Start date is after end date")
        self.content = content if content in ("all", "article", "broadcast") else "all"
        self.types = taxonomy.broadcast_type_ids(types) if types else []
        self.genre = taxonomy.genre_id_of(genre) if genre else None
        if genre and not self.genre:
            raise ValueError(f"Unknown genre: {genre}")
        self.language = language or None
        self.status = status if status in JOB_STATUSES else None
        self.interval = interval if interval in ("day", "week", "month") else "day"

    def key(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _seg_where(f: Filters, dates: bool = True) -> tuple[str, list]:
    """WHERE clause over active, validly dated segments for the content filters."""
    cond, params = [ACTIVE, f"date GLOB '{_DATE_GLOB}'"], []
    if dates:
        cond.append("date BETWEEN ? AND ?"); params += [f.date_from, f.date_to]
    if f.content == "article" and not f.types:
        cond.append("content_type = 'article'")
    elif f.content == "broadcast" or f.types:
        cond.append("IFNULL(content_type, '') != 'article'")
        if f.content == "article":
            cond.append("0")  # article-only view with a broadcast-type filter: nothing matches
    if f.types:
        cond.append(f"broadcast_type IN ({','.join('?' * len(f.types))})"); params += f.types
    if f.genre:
        cond.append("(genre_id = ? OR genre_tags LIKE ?)"); params += [f.genre, f"%,{f.genre},%"]
    if f.language:
        cond.append("LOWER(language) IN (LOWER(?), LOWER(?))")
        params += [f.language, vector_store._lang_code(f.language)]
    return " AND ".join(cond), params


DOC = "IFNULL(doc_id, source_file)"
IS_ART = "content_type = 'article'"


def _days(a: str, b: str) -> list[str]:
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return [(d0 + dt.timedelta(n)).isoformat() for n in range((d1 - d0).days + 1)]


def _runs(dates: list[str]) -> list[list[str]]:
    """Consecutive dates collapsed to [start, end] pairs."""
    out: list[list[str]] = []
    for d in dates:
        if out and (dt.date.fromisoformat(d) - dt.date.fromisoformat(out[-1][1])).days == 1:
            out[-1][1] = d
        else:
            out.append([d, d])
    return out


def _bucket(date: str, interval: str) -> str:
    if interval == "month":
        return date[:7]
    if interval == "week":
        d = dt.date.fromisoformat(date)
        return (d - dt.timedelta(d.weekday())).isoformat()  # Monday of the week
    return date


def _buckets(a: str, b: str, interval: str) -> list[str]:
    return list(dict.fromkeys(_bucket(d, interval) for d in _days(a, b)))


def _local_date(utc_iso: Optional[str]) -> Optional[str]:
    if not utc_iso:
        return None
    t = dt.datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return t.astimezone(config.get_tz()).date().isoformat()


def _utc_bounds(f: Filters) -> tuple[str, str]:
    """UTC ISO bounds of the local-date range [date_from 00:00, date_to 24:00) in the configured zone."""
    tz = config.get_tz()
    lo = dt.datetime.combine(dt.date.fromisoformat(f.date_from), dt.time(), tz).astimezone(dt.timezone.utc)
    hi = dt.datetime.combine(dt.date.fromisoformat(f.date_to) + dt.timedelta(1), dt.time(), tz).astimezone(dt.timezone.utc)
    return lo.strftime("%Y-%m-%dT%H:%M:%SZ"), hi.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(f: Optional[Filters] = None, db_path: Optional[str] = None) -> dict:
    """Configured reporting window vs actual coverage of indexed records (with the content filters)."""
    f = f or Filters()
    win_start, win_end = config.REPORT_WINDOW_START, config.REPORT_WINDOW_END
    where_all, p_all = _seg_where(f, dates=False)
    db = vector_store._get_db(db_path)
    try:
        span = db.execute(f"SELECT MIN(date), MAX(date) FROM segments WHERE {where_all}", p_all).fetchone()
        by_kind = {r[0]: (r[1], r[2]) for r in db.execute(
            f"SELECT CASE WHEN {IS_ART} THEN 'article' ELSE 'broadcast' END k, MIN(date), MAX(date) "
            f"FROM segments WHERE {where_all} GROUP BY k", p_all)}
        rows = db.execute(f"SELECT date, SUM({IS_ART}) > 0, SUM(NOT ({IS_ART}) OR content_type IS NULL) > 0 "
                          f"FROM segments WHERE {where_all} AND date BETWEEN ? AND ? GROUP BY date",
                          p_all + [win_start, win_end]).fetchall()
        outside = db.execute(f"SELECT COUNT(DISTINCT {DOC}) FROM segments WHERE {where_all} AND "
                             "(date < ? OR date > ?)", p_all + [win_start, win_end]).fetchone()[0]
        avail = {}
        for r in db.execute("SELECT date, kind, available, checked_at FROM source_availability "
                            "WHERE date BETWEEN ? AND ?", (win_start, win_end)):
            avail.setdefault(r[0], {})[r[1]] = r[2]
    finally:
        db.close()
    window_days = _days(win_start, win_end)
    with_records = {r[0] for r in rows}
    missing = [d for d in window_days if d not in with_records]
    # Classify each missing date by what is known about the source on that date.
    kinds = ["article"] if f.content == "article" and not f.types else (
        ["broadcast"] if f.content == "broadcast" or f.types else ["article", "broadcast"])
    available_not_indexed, no_source, not_checked = [], [], []
    for d in missing:
        a = avail.get(d, {})
        if any(a.get(k, 0) > 0 for k in kinds):
            available_not_indexed.append(d)
        elif all(k in a for k in kinds):
            no_source.append(d)
        else:
            not_checked.append(d)
    return {
        "timezone": config.TIMEZONE,
        "today": config.now_local().date().isoformat(),
        "reporting_window": {"start": win_start, "end": win_end, "days": len(window_days)},
        "actual": {"earliest": span[0], "latest": span[1],
                   "articles": {"earliest": by_kind.get("article", (None, None))[0],
                                "latest": by_kind.get("article", (None, None))[1]},
                   "broadcasts": {"earliest": by_kind.get("broadcast", (None, None))[0],
                                  "latest": by_kind.get("broadcast", (None, None))[1]}},
        "dates_with_records": len(with_records),
        "dates_with_articles": sum(1 for r in rows if r[1]),
        "dates_with_broadcasts": sum(1 for r in rows if r[2]),
        "missing_dates": len(missing),
        "coverage_pct": round(100 * len(with_records) / len(window_days), 1) if window_days else None,
        "spans_full_window": bool(span[0] and span[0] <= win_start and span[1] >= win_end and not missing),
        "records_outside_window": outside,
        "missing": {
            "source_available_not_indexed": {"count": len(available_not_indexed), "ranges": _runs(available_not_indexed)},
            "no_source_content": {"count": len(no_source), "ranges": _runs(no_source)},
            "not_checked": {"count": len(not_checked), "ranges": _runs(not_checked)},
        },
        "definition": ("Actual coverage uses the publication date of indexed articles and the broadcast date of "
                       "indexed bulletins (dates in " + config.TIMEZONE + "); failed or unindexed documents are "
                       "excluded. Missing dates are dates in the reporting window without an indexed record; "
                       "'no source content' is shown only for dates whose source listing was checked."),
    }


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def dataset_kpis(f: Filters, db_path: Optional[str] = None) -> dict:
    where, p = _seg_where(f)
    db = vector_store._get_db(db_path)
    try:
        r = db.execute(
            f"SELECT COUNT(DISTINCT CASE WHEN {IS_ART} THEN {DOC} END), "
            f"COUNT(DISTINCT CASE WHEN NOT ({IS_ART}) OR content_type IS NULL THEN source_file END), COUNT(*), "
            f"COUNT(DISTINCT {DOC}), COUNT(DISTINCT IFNULL(genre_id, 'other')), COUNT(DISTINCT LOWER(language)), "
            f"SUM({IS_ART}) FROM segments WHERE {where}", p).fetchone()
        audio = db.execute(
            f"SELECT COUNT(*), SUM(duration_s), SUM(timed_segments > 0), SUM(duration_s IS NOT NULL) FROM broadcasts "
            f"WHERE source_file IN "
            f"(SELECT DISTINCT source_file FROM segments WHERE {where} AND (NOT ({IS_ART}) OR content_type IS NULL))",
            p).fetchone()
        langs = {vector_store._lang_name(x[0]) or "unknown": x[1] for x in db.execute(
            f"SELECT LOWER(language), COUNT(DISTINCT {DOC}) FROM segments WHERE {where} GROUP BY 1", p)}
    finally:
        db.close()
    return {"articles": r[0], "broadcasts": r[1], "chunks": r[2], "documents": r[3], "genres": r[4],
            "languages": len(langs), "language_names": sorted(langs), "article_chunks": r[6] or 0,
            "transcript_chunks": (r[2] or 0) - (r[6] or 0),
            # audio files present on disk and readable / broadcasts whose transcript is indexed
            "audio_files_processed": audio[3] or 0,
            "broadcasts_transcribed": audio[0] or 0,
            "broadcasts_with_timestamps": audio[2] or 0,
            "audio_hours": round((audio[1] or 0) / 3600, 1) if audio[1] else None}


def _jobs(f: Filters, db_path: Optional[str] = None) -> list[dict]:
    lo, hi = _utc_bounds(f)
    cond, params = ["started_at >= ? AND started_at < ?"], [lo, hi]
    if f.status:
        cond.append("status = ?"); params.append(f.status)
    db = vector_store._get_db(db_path)
    try:
        rows = [dict(r) for r in db.execute(f"SELECT * FROM ingestion_jobs WHERE {' AND '.join(cond)} "
                                            "ORDER BY started_at", params)]
    finally:
        db.close()
    for r in rows:
        r["params"] = json.loads(r.pop("params_json") or "{}")
        progress = json.loads(r.pop("progress_json", None) or "{}")
        r["counts"] = json.loads(r.pop("counts_json") or "null") or progress
    return rows


def ingestion_kpis(f: Filters, db_path: Optional[str] = None) -> dict:
    vector_store.running_jobs(db_path)  # marks dead "running" jobs as failed first
    jobs = _jobs(f, db_path)
    c = lambda k: sum((j["counts"] or {}).get(k, 0) or 0 for j in jobs)  # noqa: E731
    done = [j for j in jobs if j["status"] in ("completed", "completed_with_errors") and j["counts"].get("duration_s")]
    last_ok = max((j["finished_at"] for j in jobs if j["status"] in ("completed", "completed_with_errors")
                   and j["finished_at"]), default=None)
    db = vector_store._get_db(db_path)
    try:
        docs = {f"{r[0]}:{r[1]}": r[2] for r in db.execute(
            "SELECT kind, status, COUNT(*) FROM ingestion_state GROUP BY kind, status")}
    finally:
        db.close()
    return {
        "jobs_total": len(jobs),
        "jobs_successful": sum(j["status"] == "completed" for j in jobs),
        "jobs_with_errors": sum(j["status"] == "completed_with_errors" for j in jobs),
        "jobs_failed": sum(j["status"] == "failed" for j in jobs),
        "jobs_cancelled": sum(j["status"] == "cancelled" for j in jobs),
        "jobs_running": sum(j["status"] == "running" for j in jobs),
        "discovered": c("articles_discovered") + c("broadcasts_discovered"),
        "duplicates_skipped": c("duplicates_skipped"),
        "processed": c("indexed_new") + c("updated"),
        "failed": c("failed"),
        "avg_duration_s": round(sum(j["counts"]["duration_s"] for j in done) / len(done), 1) if len(done) >= 2 else None,
        "last_successful_at": last_ok,
        "documents_by_status": docs,
        "documents_failed_now": sum(v for k, v in docs.items() if k.endswith(":failed")),
    }


def rag_kpis(f: Filters, db_path: Optional[str] = None) -> dict:
    lo, hi = _utc_bounds(f)
    cond, params = "ts >= ? AND ts < ?", [lo, hi]
    if f.genre:
        cond += " AND (genre_scope = ? OR detected_genre = ?)"; params += [f.genre, f.genre]
    db = vector_store._get_db(db_path)
    try:
        r = db.execute(f"SELECT COUNT(*), AVG(retrieval_ms), AVG(generation_ms), AVG(total_ms), AVG(n_chunks), "
                       f"SUM(error IS NOT NULL), SUM(insufficient), SUM(has_citations), "
                       f"SUM(insufficient = 0 AND error IS NULL), SUM(mode = 'generative'), SUM(mode = 'extractive') "
                       f"FROM query_log WHERE {cond}", params).fetchone()
        by_genre = {x[0]: x[1] for x in db.execute(
            f"SELECT IFNULL(IFNULL(genre_scope, detected_genre), 'all'), COUNT(*) FROM query_log WHERE {cond} "
            "GROUP BY 1 ORDER BY 2 DESC", params)}
        gen_rows = db.execute(f"SELECT COUNT(*) FROM query_log WHERE {cond} AND generation_ms IS NOT NULL",
                              params).fetchone()[0]
    finally:
        db.close()
    n = r[0] or 0
    rnd = lambda v: round(v, 1) if v is not None else None  # noqa: E731
    answered = r[8] or 0
    return {"queries": n,
            "avg_retrieval_ms": rnd(r[1]) if n else None,
            "avg_generation_ms": rnd(r[2]) if gen_rows else None,
            "avg_total_ms": rnd(r[3]) if n else None,
            "avg_chunks": rnd(r[4]) if n else None,
            "retrieval_failures": (r[5] or 0) if n else None,
            "no_relevant_context": (r[6] or 0) if n else None,
            "citation_coverage_pct": round(100 * (r[7] or 0) / answered, 1) if answered else None,
            "generative": r[9] or 0, "extractive": r[10] or 0,
            "by_genre": {("all" if g == "all" else taxonomy.genre_name(g) if taxonomy.genre_id_of(g) else g): v
                         for g, v in by_genre.items()}}


# ---------------------------------------------------------------------------
# Chart series
# ---------------------------------------------------------------------------

def indexed_over_time(f: Filters, db_path: Optional[str] = None) -> dict:
    where, p = _seg_where(f)
    db = vector_store._get_db(db_path)
    try:
        rows = db.execute(f"SELECT date, COUNT(DISTINCT CASE WHEN {IS_ART} THEN {DOC} END), "
                          f"COUNT(DISTINCT CASE WHEN NOT ({IS_ART}) OR content_type IS NULL THEN source_file END) "
                          f"FROM segments WHERE {where} GROUP BY date", p).fetchall()
    finally:
        db.close()
    labels = _buckets(f.date_from, f.date_to, f.interval)
    art, bc = dict.fromkeys(labels, 0), dict.fromkeys(labels, 0)
    for d, a, b in rows:
        k = _bucket(d, f.interval)
        art[k] += a
        bc[k] += b
    return {"interval": f.interval, "labels": labels,
            "series": [{"name": "Articles", "values": list(art.values())},
                       {"name": "Broadcasts", "values": list(bc.values())}]}


def distributions(f: Filters, metric: str = "documents", db_path: Optional[str] = None) -> dict:
    where, p = _seg_where(f)
    metric = metric if metric in ("documents", "articles", "broadcasts", "chunks") else "documents"
    measure = {"documents": f"COUNT(DISTINCT {DOC})", "chunks": "COUNT(*)",
               "articles": f"COUNT(DISTINCT CASE WHEN {IS_ART} THEN {DOC} END)",
               "broadcasts": f"COUNT(DISTINCT CASE WHEN NOT ({IS_ART}) OR content_type IS NULL THEN source_file END)"}[metric]
    db = vector_store._get_db(db_path)
    try:
        genres = {r[0]: r[1] for r in db.execute(
            f"SELECT IFNULL(genre_id, 'other'), {measure} FROM segments WHERE {where} GROUP BY 1", p)}
        types = {r[0] or "other": r[1] for r in db.execute(
            f"SELECT broadcast_type, COUNT(DISTINCT source_file) FROM segments WHERE {where} AND "
            f"(NOT ({IS_ART}) OR content_type IS NULL) GROUP BY 1", p)}
        langs = {r[0]: r[1] for r in db.execute(
            f"SELECT LOWER(language), COUNT(DISTINCT {DOC}) FROM segments WHERE {where} GROUP BY 1", p)}
        cats = {r[0]: r[1] for r in db.execute(
            f"SELECT source_category, COUNT(DISTINCT {DOC}) FROM segments WHERE {where} AND {IS_ART} "
            "AND source_category IS NOT NULL GROUP BY 1 ORDER BY 2 DESC", p)}
    finally:
        db.close()
    return {
        "metric": metric,
        "genres": [{"id": g, "name": taxonomy.genre_name(g), "value": genres[g]}
                   for g in sorted(genres, key=lambda g: -genres[g]) if genres[g]],
        "broadcast_types": [{"id": t, "name": taxonomy.broadcast_type_name(t), "value": types[t]}
                            for t in taxonomy.BROADCAST_TYPES if types.get(t)],
        "languages": [{"id": k, "name": vector_store._lang_name(k) or "unknown", "value": v}
                      for k, v in sorted(langs.items(), key=lambda x: -x[1])],
        "source_categories": [{"id": k, "name": k, "value": v} for k, v in cats.items()],
    }


def job_history(f: Filters, db_path: Optional[str] = None) -> dict:
    jobs = _jobs(f, db_path)
    labels = _buckets(f.date_from, f.date_to, f.interval)
    series = {s: dict.fromkeys(labels, 0) for s in JOB_STATUSES}
    activity = {k: dict.fromkeys(labels, 0) for k in ("processed", "duplicates", "failed")}
    for j in jobs:
        k = _bucket(_local_date(j["started_at"]), f.interval)
        if k not in series[j["status"] if j["status"] in series else "failed"]:
            continue
        series[j["status"] if j["status"] in series else "failed"][k] += 1
        c = j["counts"] or {}
        activity["processed"][k] += (c.get("indexed_new", 0) or 0) + (c.get("updated", 0) or 0)
        activity["duplicates"][k] += c.get("duplicates_skipped", 0) or 0
        activity["failed"][k] += c.get("failed", 0) or 0
    names = {"completed": "Completed", "completed_with_errors": "Completed with errors", "failed": "Failed",
             "cancelled": "Cancelled", "running": "Running"}
    return {"interval": f.interval, "labels": labels,
            "jobs": [{"name": names[s], "id": s, "values": list(v.values())} for s, v in series.items()
                     if any(v.values())],
            "activity": [{"name": n.capitalize() if n != "duplicates" else "Duplicates skipped", "id": n,
                          "values": list(v.values())} for n, v in activity.items()],
            "recent": [{k: j.get(k) for k in ("job_id", "started_at", "finished_at", "status", "trigger", "kind",
                                              "date_from", "date_to", "types", "error")} | {"counts": j["counts"]}
                       for j in reversed(jobs[-25:])]}


def funnel(f: Filters, db_path: Optional[str] = None) -> dict:
    """Processing funnel summed over the jobs in range, kept separate for articles and broadcasts.

    Chunk/embedding counters are split by kind from per-kind counters; older jobs recorded a single
    total, which is attributed to a kind only when that job processed one kind.
    """
    jobs = _jobs(f, db_path)
    art = dict.fromkeys(("discovered", "downloaded", "chunked", "embedded", "indexed"), 0)
    bc = dict.fromkeys(("discovered", "downloaded", "transcribed", "chunked", "embedded", "indexed"), 0)
    unattributed = {"chunked": 0, "embedded": 0}
    for j in jobs:
        c = j["counts"] or {}
        g = lambda k: c.get(k, 0) or 0  # noqa: E731
        art["discovered"] += g("articles_discovered")
        art["downloaded"] += g("articles_downloaded")
        art["indexed"] += g("articles_indexed")
        bc["discovered"] += g("broadcasts_discovered")
        bc["downloaded"] += g("broadcasts_downloaded") + g("audio_cached")
        bc["transcribed"] += g("transcriptions_ok")
        bc["indexed"] += g("broadcasts_indexed")
        if "article_chunks" in c or "transcript_chunks" in c:
            art["chunked"] += g("article_chunks"); art["embedded"] += g("article_embeddings")
            bc["chunked"] += g("transcript_chunks"); bc["embedded"] += g("transcript_embeddings")
        elif g("broadcasts_discovered") == 0:
            art["chunked"] += g("chunks"); art["embedded"] += g("embeddings")
        elif g("articles_discovered") == 0:
            bc["chunked"] += g("chunks"); bc["embedded"] += g("embeddings")
        else:
            unattributed["chunked"] += g("chunks"); unattributed["embedded"] += g("embeddings")
    return {"articles": [{"stage": k.capitalize(), "value": v} for k, v in art.items()],
            "broadcasts": [{"stage": k.capitalize(), "value": v} for k, v in bc.items()],
            "not_applicable": {"articles": ["Transcribed"]},
            "unattributed": unattributed,
            "jobs": len(jobs)}


def rag_series(f: Filters, db_path: Optional[str] = None) -> dict:
    lo, hi = _utc_bounds(f)
    db = vector_store._get_db(db_path)
    try:
        rows = db.execute("SELECT ts, retrieval_ms, generation_ms, total_ms FROM query_log WHERE ts >= ? AND ts < ?",
                          (lo, hi)).fetchall()
    finally:
        db.close()
    labels = _buckets(f.date_from, f.date_to, f.interval)
    count = dict.fromkeys(labels, 0)
    sums = {k: {b: [] for b in labels} for k in ("retrieval_ms", "generation_ms", "total_ms")}
    for r in rows:
        k = _bucket(_local_date(r["ts"]), f.interval)
        if k not in count:
            continue
        count[k] += 1
        for m in sums:
            if r[m] is not None:
                sums[m][k].append(r[m])
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    return {"interval": f.interval, "labels": labels, "queries": list(count.values()),
            "latency": [{"name": n, "id": m, "values": [avg(v) for v in sums[m].values()]}
                        for m, n in (("retrieval_ms", "Retrieval"), ("generation_ms", "Generation"),
                                     ("total_ms", "End-to-end"))],
            "measured": len(rows)}


# ---------------------------------------------------------------------------
# Cache (dashboard requests repeat the same filters; data changes only on ingestion/queries)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, str, dict]] = {}
_cache_lock = threading.Lock()


def _data_version(db_path: Optional[str] = None) -> str:
    db = vector_store._get_db(db_path)
    try:
        r = db.execute("SELECT (SELECT MAX(faiss_id) FROM segments), (SELECT MAX(indexed_at) FROM segments), "
                       "(SELECT COUNT(*) FROM segments WHERE IFNULL(active, 1) = 0), "
                       "(SELECT MAX(updated_at) FROM ingestion_jobs), (SELECT COUNT(*) FROM query_log), "
                       "(SELECT MAX(checked_at) FROM source_availability), "
                       "(SELECT COUNT(*) || '/' || IFNULL(MAX(indexed_at), '') FROM broadcasts)").fetchone()
    finally:
        db.close()
    return json.dumps(list(r))


def invalidate() -> None:
    """Drop cached results (after metadata changes that do not alter the data version, e.g. a re-labelling)."""
    with _cache_lock:
        _cache.clear()


def cached(name: str, f: Filters, fn, *args, ttl: float = 300.0, db_path: Optional[str] = None) -> dict:
    """Result of fn(f, *args) reused while the data version is unchanged (and for at most ttl seconds)."""
    key = f"{name}|{f.key()}|{args}|{db_path}"
    ver = _data_version(db_path)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[1] == ver and time.time() - hit[0] < ttl:
            return hit[2]
    val = fn(f, *args, db_path=db_path) if args else fn(f, db_path=db_path)
    with _cache_lock:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = (time.time(), ver, val)
    return val


def summary(f: Filters, db_path: Optional[str] = None) -> dict:
    return {"filters": f.as_dict(),
            "generated_at": config.now_local().isoformat(timespec="seconds"),
            "coverage": cached("coverage", f, coverage, db_path=db_path),
            "dataset": cached("dataset", f, dataset_kpis, db_path=db_path),
            "ingestion": cached("ingestion", f, ingestion_kpis, ttl=10, db_path=db_path),
            "rag": cached("rag", f, rag_kpis, db_path=db_path)}
