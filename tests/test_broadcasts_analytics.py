"""Broadcast browsing, date-range ingestion, taxonomy, coverage and analytics (offline, deterministic).

Bulletins are stored through the real vector_store path (stand-in embeddings from conftest);
the NewsOnAIR archive is a fake client serving pages in the real share-block format.
"""

import datetime as dt
import json
import os

import pytest
from fastapi.testclient import TestClient

import analytics
import config
import newsonair_ingest as noa
import rag_engine
import taxonomy
import vector_store
from conftest import fake_embed
from test_newsonair import BUSINESS, NATIONAL, _counts, _share


def today():
    return config.now_local().date()


def ago(n):
    return (today() - dt.timedelta(days=n)).isoformat()


def bulletin(bid, title, date, hhmm, sentences, audio=True, lang="en"):
    """Index one bulletin the way index_bulletin stores it (state + chunks + broadcast row)."""
    name = f"NOA-{title.replace(' ', '-')}-{hhmm:04d}-{bid}"
    src = f"text/{date}/{name}.txt"
    if audio:
        p = os.path.join(config.OUTPUT_DIR, "audio", date)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, name + ".mp3"), "wb") as fh:
            fh.write(b"ID3fake")
    embs = fake_embed(sentences)
    segs = [{"text": s, "genre": "Other", "genre_confidence": 0.2, "embedding": embs[i], "audio_start": float(i * 5),
             "audio_end": float(i * 5 + 4), "chunk_index": i} for i, s in enumerate(sentences)]
    doc_id = f"noa-bulletin-{bid}"
    meta = {"bulletin_id": str(bid), "title": title, "category": title, "language": "English",
            "broadcast_at": f"{date}T{hhmm // 100:02d}:{hhmm % 100:02d}:00",
            "audio_url": f"https://newsonair.gov.in/wp-content/uploads/{bid}.mp3",
            "page_url": f"https://newsonair.gov.in/listen-news/{bid}/"}
    vector_store.set_state(doc_id, kind="broadcast", source_url=meta["page_url"], title=title, status="indexed",
                           content_hash="h", meta_json=json.dumps(meta))
    vector_store.add_segments(segs, date, hhmm, src, source_name=f"NewsOnAIR {title}", source_url=meta["page_url"],
                              ingest_mode="archive", language=lang,
                              doc={"doc_id": doc_id, "content_type": "audio_transcript",
                                   "headline": f"{title} (English)", "source_category": title,
                                   "published_at": meta["broadcast_at"]})
    return src


@pytest.fixture
def bc(env):
    """Six bulletins over three days, several programmes and topics."""
    d0, d1, d2 = ago(3), ago(4), ago(6)
    bulletin(101, "Midday News", d0, 1400, ["India beat Australia in the cricket match at Ahmedabad.",
                                            "The Reserve Bank kept the repo rate unchanged."])
    bulletin(102, "News At Nine", d0, 2100, ["Heavy rainfall and a cyclone warning for coastal Odisha.",
                                             "Farmers were advised to protect the paddy crop."])
    bulletin(103, "Hourly News", d0, 1805, ["The hockey team won the tournament final."])
    bulletin(104, "Midday News", d1, 1400, ["The Army conducted a joint exercise near the border."])
    bulletin(105, "Evening News", d1, 2100, ["Parliament passed the data protection bill after debate."])
    bulletin(106, "Some New Programme", d2, 1000, ["A special bulletin was broadcast."])
    return {"d0": d0, "d1": d1, "d2": d2}


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_taxonomy_ids_names_and_rules():
    assert len(set(taxonomy.GENRES.values())) == len(taxonomy.GENRES)  # no duplicate display names
    assert taxonomy.genre_id_of("Business & Economy") == "business" and taxonomy.genre_id_of("sports") == "sports"
    assert taxonomy.genre_id_of("Economy") is None  # legacy model label, not a taxonomy name
    # publisher category wins, refined by keywords for a broad category
    assert taxonomy.classify("Sensex and Nifty rise as RBI holds repo rate", "", "Business")[0] == "finance"
    assert taxonomy.classify("Team selected", "", "Sports")[0] == "sports"
    # keywords for a broad desk; the desk is kept as a secondary tag
    g, tags, rule = taxonomy.classify("Farmers get MSP for paddy procurement", "Kharif crop farmers", "National")
    assert (g, rule) == ("agriculture", "keywords") and "national" in tags
    # nothing specific -> the publisher desk, and nothing at all -> Other / Unclassified
    assert taxonomy.classify("Minister addresses gathering", "", "Regional")[:1] == ("regional",)
    assert taxonomy.classify("A statement was issued", "", None)[0] == "other"
    # low-confidence model labels are ignored, confident ones used
    assert taxonomy.classify("A statement", "", None, "Politics", 0.2)[0] == "other"
    assert taxonomy.classify("A statement", "", None, "Politics", 0.8)[0] == "politics"
    # transcript sentences: one unambiguous keyword decides; a tie stays unclassified
    assert taxonomy.classify("", "Heavy rainfall expected.", content_type="audio_transcript")[0] == "environment"
    assert taxonomy.classify("", "Cricket and rainfall.", content_type="audio_transcript")[0] == "other"
    assert taxonomy.encode_tags(["a", "b"]) == ",a,b," and taxonomy.decode_tags(",a,b,") == ["a", "b"]


def test_broadcast_types_from_titles():
    assert taxonomy.broadcast_type_of("Midday News") == "midday"
    assert taxonomy.broadcast_type_of("News At Nine") == "news-at-nine"
    assert taxonomy.broadcast_type_of("Hourly News") == "hourly"
    assert taxonomy.broadcast_type_of("Some New Programme") == "other"
    assert taxonomy.broadcast_type_ids(["Midday News", "news-at-nine"]) == ["midday", "news-at-nine"]
    with pytest.raises(ValueError):
        taxonomy.broadcast_type_ids(["bogus"])


def test_backfill_relabels_without_reembedding(bc):
    db = vector_store._get_db()
    db.execute("UPDATE segments SET genre_id = NULL, genre_tags = NULL, broadcast_type = NULL")
    db.commit()
    db.close()
    n_vectors = vector_store._load_index().ntotal
    stats = taxonomy.backfill()
    assert stats["transcript_chunks"] == 8 and stats["broadcasts"] == 6
    assert vector_store._load_index().ntotal == n_vectors  # metadata only
    db = vector_store._get_db()
    types = dict(db.execute("SELECT source_category, MAX(broadcast_type) FROM segments GROUP BY 1").fetchall())
    db.close()
    assert types["Midday News"] == "midday" and types["Some New Programme"] == "other"


# ---------------------------------------------------------------------------
# Broadcast browsing
# ---------------------------------------------------------------------------

def test_day_wise_grouping_order_and_metadata(bc):
    r = vector_store.list_broadcast_days(days=7)
    assert [d["date"] for d in r["days"]] == [bc["d0"], bc["d1"], bc["d2"]]  # newest first
    assert r["total_broadcasts"] == 6 and r["total_days"] == 3 and r["next_offset"] is None
    first = r["days"][0]["broadcasts"]
    assert [b["time"] for b in first] == ["21:00", "18:05", "14:00"]  # one date only, latest first
    b = first[0]
    assert b["title"] == "News At Nine" and b["broadcast_type"] == "news-at-nine" and b["language"] == "English"
    assert b["audio_url"].endswith("102.mp3") and b["page_url"].endswith("/102/") and b["has_audio"]
    assert b["bulletin_id"] == "102" and b["status"] == "indexed" and b["segments"] == 2 and b["timed_segments"] == 2
    asc = vector_store.list_broadcast_days(order="asc")
    assert asc["days"][0]["date"] == bc["d2"]


@pytest.mark.parametrize("type_id,titles", [("midday", {"Midday News"}), ("news-at-nine", {"News At Nine"}),
                                            ("hourly", {"Hourly News"}), ("other", {"Some New Programme"})])
def test_broadcast_type_filters(bc, type_id, titles):
    r = vector_store.list_broadcast_days(types=[type_id])
    assert {b["title"] for d in r["days"] for b in d["broadcasts"]} == titles


def test_multi_type_and_combined_filters(bc):
    r = vector_store.list_broadcast_days(types=["midday", "news-at-nine"])
    assert r["total_broadcasts"] == 3
    # Midday + a genre: only the Midday bulletin that covers sports
    r = vector_store.list_broadcast_days(types=["midday"], genre="sports")
    assert [b["bulletin_id"] for d in r["days"] for b in d["broadcasts"]] == ["101"]
    # day-wise + genre + date range
    r = vector_store.list_broadcast_days(genre="defence", date_from=bc["d1"], date_to=bc["d1"])
    assert [b["bulletin_id"] for d in r["days"] for b in d["broadcasts"]] == ["104"]
    # language
    assert vector_store.list_broadcast_days(language="English")["total_broadcasts"] == 6
    assert vector_store.list_broadcast_days(language="Hindi")["total_broadcasts"] == 0


def test_broadcast_pagination_and_search(bc):
    p1 = vector_store.list_broadcast_days(days=2)
    assert len(p1["days"]) == 2 and p1["next_offset"] == 2
    p2 = vector_store.list_broadcast_days(days=2, offset=2)
    assert [d["date"] for d in p2["days"]] == [bc["d2"]] and p2["next_offset"] is None
    # transcript content, title and date are searchable
    assert [b["bulletin_id"] for d in vector_store.list_broadcast_days(q="cyclone")["days"] for b in d["broadcasts"]] == ["102"]
    assert vector_store.list_broadcast_days(q="hourly")["total_broadcasts"] == 1
    assert vector_store.list_broadcast_days(q=bc["d1"])["total_broadcasts"] == 2


def test_facets_list_only_present_values(bc):
    f = vector_store.broadcast_facets()
    assert {t["id"]: t["count"] for t in f["types"]} == {"midday": 2, "news-at-nine": 1, "hourly": 1, "evening": 1,
                                                         "other": 1}
    assert f["languages"] == {"English": 6}
    assert "sports" in {g["id"] for g in f["genres"]}


# ---------------------------------------------------------------------------
# Articles: pagination, search, combined filters
# ---------------------------------------------------------------------------

def test_article_pagination_filters_and_related(env):
    for i in range(30):
        rec = noa.parse_article(dict(BUSINESS, id=5000 + i, date=f"{ago(i % 10)}T10:{i:02d}:00",
                                     title={"rendered": f"Sensex rises on RBI repo rate pause {i}"}), "x")
        rec["genre"] = "Economy"
        noa.index_article(rec, _counts(), "J")
    page, total = vector_store.list_articles_page(limit=12)
    assert total == 30 and len(page) == 12
    page2, _ = vector_store.list_articles_page(limit=12, offset=12)
    assert not {a["id"] for a in page} & {a["id"] for a in page2}
    dates = [a["published"] for a in page]
    assert dates == sorted(dates, reverse=True)
    fin, n = vector_store.list_articles_page(genre="finance", date_from=ago(2), date_to=ago(0), q="repo")
    assert n == 9 and all(a["genre_id"] == "finance" and ago(2) <= a["published"] <= ago(0) for a in fin)
    rel = vector_store.related_articles(page[0])
    assert rel and all(r["id"] != page[0]["id"] for r in rel)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_uses_indexed_records_not_the_window(bc, monkeypatch):
    monkeypatch.setattr(config, "REPORT_WINDOW_START", ago(10))
    monkeypatch.setattr(config, "REPORT_WINDOW_END", ago(0))
    c = analytics.coverage()
    assert c["reporting_window"] == {"start": ago(10), "end": ago(0), "days": 11}
    assert c["actual"]["earliest"] == bc["d2"] and c["actual"]["latest"] == bc["d0"]  # not the window bounds
    assert c["dates_with_records"] == 3 and c["missing_dates"] == 8 and c["spans_full_window"] is False
    assert c["coverage_pct"] == round(100 * 3 / 11, 1)
    # missing dates: checked with nothing / checked with content not indexed / not checked
    vector_store.record_availability("article", {ago(5): 0, ago(7): 4})
    vector_store.record_availability("broadcast", {ago(5): 0, ago(7): 0})
    m = analytics.coverage()["missing"]
    assert m["no_source_content"]["ranges"] == [[ago(5), ago(5)]]
    assert m["source_available_not_indexed"]["ranges"] == [[ago(7), ago(7)]]
    assert m["not_checked"]["count"] == 6


def test_coverage_excludes_retired_and_invalid_records(bc, monkeypatch):
    monkeypatch.setattr(config, "REPORT_WINDOW_START", ago(10))
    monkeypatch.setattr(config, "REPORT_WINDOW_END", ago(0))
    vector_store.deactivate_doc("noa-bulletin-106")  # superseded / withdrawn -> not coverage
    db = vector_store._get_db()
    db.execute("UPDATE segments SET date = 'unknown' WHERE doc_id = 'noa-bulletin-105'")
    db.commit()
    db.close()
    c = analytics.coverage()
    assert c["actual"]["earliest"] == bc["d1"] and c["dates_with_records"] == 2
    only_articles = analytics.coverage(analytics.Filters(content="article"))
    assert only_articles["actual"]["earliest"] is None and only_articles["dates_with_records"] == 0


# ---------------------------------------------------------------------------
# Date-range discovery, validation, selection
# ---------------------------------------------------------------------------

class FakeArchive:
    """Archive pages, newest first: page n holds bulletins of day n // 2 (4 per page)."""

    def __init__(self, n_pages=60):
        self.requests = []
        self.pages = {}
        for p in range(1, n_pages + 1):
            day = dt.date.fromisoformat(ago((p - 1) // 2))
            blocks = []
            for k in range(4):
                hh = 21 - ((p - 1) % 2) * 8 - k * 2
                title = {0: "Midday News", 1: "Hourly News", 2: "News At Nine", 3: "Hourly News"}[k]
                bid = 900000 - p * 10 - k
                blocks.append(_share(bid, title, title, "English", day.strftime("%d %B %Y") + f" {hh:02d}:00",
                                     f"https://x/{bid}.mp3", f"https://x/listen/{bid}/"))
            self.pages[p] = "<html>" + "".join(blocks) + "</html>"

    def get(self, url, params=None, stream=False):
        self.requests.append(url)
        page = int(url.rstrip("/").split("/")[-1]) if "/page/" in url else 1

        class R:
            text = self.pages.get(page, "<html></html>")
            status_code = 200
        return R()


def test_validate_range_messages_and_no_silent_changes():
    errs, _ = noa.validate_range(ago(1), ago(5), ["midday"])
    assert any("after end date" in e for e in errs)
    future = (today() + dt.timedelta(days=2)).isoformat()
    assert any("future" in e for e in noa.validate_range(ago(1), future, ["midday"])[0])
    assert any("at most" in e for e in noa.validate_range(ago(60), ago(1), ["midday"])[0])
    assert any("valid date" in e for e in noa.validate_range("2026-13-01", ago(1), ["midday"])[0])
    assert any("at least one" in e for e in noa.validate_range(ago(3), ago(1), [])[0])
    assert any("Unknown broadcast type" in e for e in noa.validate_range(ago(3), ago(1), ["nope"])[0])
    errs, ids = noa.validate_range(ago(3), ago(1), ["Midday News", "news-at-nine"])
    assert errs == [] and ids == ["midday", "news-at-nine"]


def test_scan_archive_finds_range_with_few_requests_and_uses_catalog(env):
    fake = FakeArchive()
    items = noa.scan_archive(fake, ago(12), ago(10))
    dates = {b["broadcast_at"][:10] for b in items}
    assert dates == {ago(10), ago(11), ago(12)} and len(items) == 24   # 4 per page, 2 pages per day
    assert len(fake.requests) < 20  # pages 21-26 found by doubling/bisecting, not by reading pages 1-20
    # what the source had on those (completed) dates is recorded, and the listing is catalogued
    db = vector_store._get_db()
    avail = dict(db.execute("SELECT date, available FROM source_availability WHERE kind='broadcast'").fetchall())
    db.close()
    assert avail == {ago(10): 8, ago(11): 8, ago(12): 8}
    # a second scan of the same range is served from the catalog: no requests
    fake2 = FakeArchive()
    again = noa.scan_archive(fake2, ago(12), ago(10))
    assert fake2.requests == [] and {b["bulletin_id"] for b in again} == {b["bulletin_id"] for b in items}


def test_select_by_types_skips_indexed_and_counts_failed(env):
    items = noa.scan_archive(FakeArchive(), ago(12), ago(10))
    sel = noa.select_bulletins(items, types=["midday", "news-at-nine"])
    assert len(sel["to_process"]) == 12 and all(taxonomy.broadcast_type_of(b["title"]) in ("midday", "news-at-nine")
                                                for b in sel["to_process"])
    done, failed = sel["to_process"][0], sel["to_process"][1]
    vector_store.set_state(f"noa-bulletin-{done['bulletin_id']}", kind="broadcast", status="indexed")
    vector_store.set_state(f"noa-bulletin-{failed['bulletin_id']}", kind="broadcast", status="failed", attempts=3)
    sel2 = noa.select_bulletins(items, types=["midday", "news-at-nine"])
    assert len(sel2["to_process"]) == 11 and len(sel2["already_indexed"]) == 1 and sel2["previously_failed"] == 1
    assert len(noa.select_bulletins(items, types=["midday"], include_indexed=True)["to_process"]) == 6


# ---------------------------------------------------------------------------
# Jobs: confirmed items, incremental, cancel, retry, overlap, liveness
# ---------------------------------------------------------------------------

@pytest.fixture
def offline(env, monkeypatch):
    """run() without network: robots allowed, bulletin indexing recorded instead of executed."""
    monkeypatch.setattr(noa.PoliteClient, "robots_allows", lambda self, path: True)
    done = []

    def fake_index(b, client, counts, job_id, transcribe=True, force=False, stage=None):
        st = vector_store.get_state(f"noa-bulletin-{b['bulletin_id']}")
        if st and st["status"] == "failed" and not b.get("ok"):
            counts["failed"] += 1
            vector_store.set_state(f"noa-bulletin-{b['bulletin_id']}", kind="broadcast", status="failed",
                                   attempts=(st["attempts"] or 0) + 1)
            return
        stage("transcribing")
        done.append(b["bulletin_id"])
        counts["broadcasts_indexed"] += 1
        counts["indexed_new"] += 1
        vector_store.set_state(f"noa-bulletin-{b['bulletin_id']}", kind="broadcast", status="indexed",
                               meta_json=json.dumps(b))
    monkeypatch.setattr(noa, "index_bulletin", fake_index)
    return done


def _items(n):
    return [{"bulletin_id": str(700 + i), "title": "Midday News", "category": "Midday News", "language": "English",
             "broadcast_at": f"{ago(2)}T14:00:00", "audio_url": f"https://x/{700 + i}.mp3",
             "page_url": f"https://x/{700 + i}/"} for i in range(n)]


def test_confirmed_job_processes_selected_items_and_records_progress(offline):
    res = noa.run(date_from=ago(2), date_to=ago(2), types=["midday"], items=_items(3), articles=0,
                  include_articles=False)
    assert res["status"] == "completed" and res["counts"]["broadcasts_indexed"] == 3 and offline == ["700", "701", "702"]
    job = vector_store.list_jobs(1)[0]
    assert job["status"] == "completed" and job["kind"] == "range" and job["types"] == "midday"
    assert job["date_from"] == ago(2) and job["counts"]["broadcasts_indexed"] == 3 and job["pid"] == os.getpid()


def test_cancel_stops_between_documents_and_keeps_work(offline, monkeypatch):
    real = noa.index_bulletin

    def cancelling(b, client, counts, job_id, **kw):
        real(b, client, counts, job_id, **kw)
        vector_store.request_cancel(job_id)
    monkeypatch.setattr(noa, "index_bulletin", cancelling)
    res = noa.run(date_from=ago(2), date_to=ago(2), types=["midday"], items=_items(4), articles=0,
                  include_articles=False)
    assert res["status"] == "cancelled" and offline == ["700"]
    assert vector_store.get_state("noa-bulletin-700")["status"] == "indexed"
    assert vector_store.list_jobs(1)[0]["status"] == "cancelled"


def test_retry_failed_reprocesses_from_stored_listing(offline):
    b = dict(_items(1)[0], ok=True)
    vector_store.set_state("noa-bulletin-700", kind="broadcast", status="failed", attempts=3,
                           meta_json=json.dumps(b), last_error="IncompleteRead")
    vector_store.set_state("noa-bulletin-999", kind="broadcast", status="failed", attempts=1)  # no listing stored
    res = noa.run(retry_failed=True)
    assert offline == ["700"] and vector_store.get_state("noa-bulletin-700")["status"] == "indexed"
    assert res["counts"]["broadcasts_discovered"] == 2 and res["counts"]["failed"] == 1  # 999 cannot be retried
    assert res["status"] == "completed_with_errors"


def test_overlapping_job_is_refused_and_dead_jobs_are_released(env, monkeypatch):
    vector_store.start_job("NOA-A", "cli", {"kind": "range", "date_from": ago(10), "date_to": ago(5),
                                            "types": ["midday"], "articles": 0, "include_articles": False})
    assert noa._overlapping_job(ago(6), ago(4), {"midday"}, False)["job_id"] == "NOA-A"
    assert noa._overlapping_job(ago(6), ago(4), {"hourly"}, False) is None        # other type
    assert noa._overlapping_job(ago(3), ago(1), {"midday"}, False) is None        # other dates
    assert noa._overlapping_job(ago(6), ago(4), None, True) is None               # articles only
    with pytest.raises(RuntimeError, match="overlapping"):
        noa.run(date_from=ago(7), date_to=ago(6), types=["midday"], items=[], articles=0, include_articles=False)
    # The job's process is gone -> it is marked failed instead of blocking forever.
    db = vector_store._get_db()
    db.execute("UPDATE ingestion_jobs SET pid = 999999 WHERE job_id = 'NOA-A'")
    db.commit()
    db.close()
    monkeypatch.setattr(vector_store, "_process_alive", lambda pid: False)
    assert vector_store.running_jobs() == []
    assert vector_store.list_jobs(1)[0]["status"] == "failed"


def test_incremental_article_run_skips_unchanged(env, monkeypatch):
    monkeypatch.setattr(noa.PoliteClient, "robots_allows", lambda self, path: True)
    monkeypatch.setattr(noa, "discover_articles_range", lambda c, a, b, limit=None: [dict(BUSINESS), dict(NATIONAL)])
    r1 = noa.run(date_from=ago(1), date_to=ago(1), articles=None, include_articles=True, broadcasts=0)
    r2 = noa.run(date_from=ago(1), date_to=ago(1), articles=None, include_articles=True, broadcasts=0)
    assert r1["counts"]["indexed_new"] == 2 and r2["counts"]["indexed_new"] == 0
    assert r2["counts"]["duplicates_skipped"] == 2


# ---------------------------------------------------------------------------
# RAG: genre-aware retrieval, query log
# ---------------------------------------------------------------------------

def test_rag_retrieval_by_genre_and_across_genres(bc):
    res = rag_engine.answer_query("What happened in the match?", genre="sports")
    assert res["sources"] and all(s["genre_id"] == "sports" or s["genre_id"] is None for s in res["sources"])
    cross = rag_engine.answer_query("cricket match and repo rate and cyclone warning")
    assert len({s["doc_id"] for s in cross["sources"]}) >= 2
    assert {s["broadcast_type"] for s in cross["sources"]} - {None}
    assert cross["cited"]  # citations survive


def test_near_duplicate_chunks_across_documents_are_dropped():
    hits = [{"text": "The Reserve Bank kept the repo rate unchanged today.", "doc_id": "a", "faiss_id": 1},
            {"text": "The Reserve Bank kept the repo rate unchanged today", "doc_id": "b", "faiss_id": 2},
            {"text": "The Reserve Bank kept the repo rate unchanged today.", "doc_id": "a", "faiss_id": 3},
            {"text": "Cricket result.", "doc_id": "c", "faiss_id": 4}]
    assert [h["faiss_id"] for h in rag_engine._drop_near_duplicates(hits)] == [1, 3, 4]


def test_query_log_records_timings_not_question_text(bc, monkeypatch):
    monkeypatch.setattr(config, "QUERY_LOG_ENABLED", True)
    rag_engine.answer_query("Tell me about the cricket match", genre="sports")
    rag_engine.answer_query("What is the price of bitcoin?")
    db = vector_store._get_db()
    rows = [dict(r) for r in db.execute("SELECT * FROM query_log ORDER BY ts")]
    db.close()
    assert len(rows) == 2 and rows[0]["genre_scope"] == "sports" and rows[0]["retrieval_ms"] is not None
    assert rows[0]["n_sources"] >= 1 and json.loads(rows[0]["retrieved_docs"])
    assert not {"question", "query", "message", "text"} & set(rows[0]) and "cricket" not in json.dumps(rows)
    assert len({r["request_id"] for r in rows}) == 2  # unique ids, no overwrite
    f = analytics.Filters(date_from=ago(1), date_to=today().isoformat())
    k = analytics.rag_kpis(f)
    assert k["queries"] == 2 and k["avg_total_ms"] is not None and k["by_genre"]["Sports"] == 1


# ---------------------------------------------------------------------------
# Dashboard data
# ---------------------------------------------------------------------------

def test_dashboard_kpis_series_and_filters(bc):
    noa.index_article(noa.parse_article(dict(BUSINESS, date=f"{bc['d0']}T10:00:00"), "x"), _counts(), "J")
    f = analytics.Filters(date_from=ago(8), date_to=ago(0), interval="day")
    k = analytics.dataset_kpis(f)
    assert (k["articles"], k["broadcasts"], k["documents"]) == (1, 6, 7)
    assert k["languages"] == 1 and k["audio_files_processed"] == 0  # fake mp3s have no readable duration
    ts = analytics.indexed_over_time(f)
    assert len(ts["labels"]) == 9 and sum(ts["series"][1]["values"]) == 6 and sum(ts["series"][0]["values"]) == 1
    assert ts["series"][1]["values"][ts["labels"].index(bc["d0"])] == 3
    # filters narrow every query
    only_b = analytics.Filters(date_from=ago(8), date_to=ago(0), content="broadcast", types=["midday"])
    assert analytics.dataset_kpis(only_b)["broadcasts"] == 2 and analytics.dataset_kpis(only_b)["articles"] == 0
    d = analytics.distributions(only_b, "documents")
    assert [t["id"] for t in d["broadcast_types"]] == ["midday"]
    monthly = analytics.indexed_over_time(analytics.Filters(date_from=ago(40), date_to=ago(0), interval="month"))
    assert all(len(lbl) == 7 for lbl in monthly["labels"])
    weekly = analytics.indexed_over_time(analytics.Filters(date_from=ago(20), date_to=ago(0), interval="week"))
    assert all(dt.date.fromisoformat(lbl).weekday() == 0 for lbl in weekly["labels"])
    with pytest.raises(ValueError):
        analytics.Filters(date_from=ago(1), date_to=ago(3))


def test_ingestion_kpis_history_and_funnel(env):
    vector_store.start_job("J1", "cli", {"kind": "range", "date_from": ago(3), "date_to": ago(2)})
    vector_store.finish_job("J1", "completed", {"articles_discovered": 10, "articles_downloaded": 10, "articles_indexed": 8,
                                                "indexed_new": 8, "duplicates_skipped": 2, "chunks": 20,
                                                "embeddings": 20, "duration_s": 30.0})
    vector_store.start_job("J2", "dashboard", {"kind": "range"})
    vector_store.finish_job("J2", "completed_with_errors", {"broadcasts_discovered": 3, "broadcasts_downloaded": 2,
                                                            "audio_cached": 1, "transcriptions_ok": 2,
                                                            "broadcasts_indexed": 2, "indexed_new": 2, "failed": 1,
                                                            "transcript_chunks": 40, "transcript_embeddings": 55,
                                                            "chunks": 40, "embeddings": 55, "duration_s": 90.0})
    f = analytics.Filters(date_from=ago(1), date_to=today().isoformat())
    k = analytics.ingestion_kpis(f)
    assert (k["jobs_total"], k["jobs_successful"], k["jobs_with_errors"], k["failed"]) == (2, 1, 1, 1)
    assert k["discovered"] == 13 and k["processed"] == 10 and k["avg_duration_s"] == 60.0
    fn = analytics.funnel(f)
    art = {s["stage"]: s["value"] for s in fn["articles"]}
    bcs = {s["stage"]: s["value"] for s in fn["broadcasts"]}
    assert art == {"Discovered": 10, "Downloaded": 10, "Chunked": 20, "Embedded": 20, "Indexed": 8}
    assert bcs == {"Discovered": 3, "Downloaded": 3, "Transcribed": 2, "Chunked": 40, "Embedded": 55, "Indexed": 2}
    h = analytics.job_history(analytics.Filters(date_from=ago(1), date_to=today().isoformat(), status="completed"))
    assert sum(sum(s["values"]) for s in h["jobs"]) == 1


def test_empty_dashboard_reports_not_measured(env):
    f = analytics.Filters(date_from=ago(5), date_to=ago(0))
    r = analytics.rag_kpis(f)
    assert r["queries"] == 0 and r["avg_retrieval_ms"] is None and r["citation_coverage_pct"] is None
    assert analytics.ingestion_kpis(f)["avg_duration_s"] is None
    assert analytics.coverage()["actual"]["earliest"] is None
    assert analytics.distributions(f)["genres"] == []


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture
def client(bc, monkeypatch):
    import chatbot_api
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    with TestClient(chatbot_api.app) as c:
        yield c


def test_api_broadcasts_transcript_and_audio(client, bc):
    d = client.get("/broadcasts", params={"mode": "type", "type": "news-at-nine"}).json()
    b = d["days"][0]["broadcasts"][0]
    assert b["title"] == "News At Nine" and "text" not in b  # list carries no transcript
    assert client.get("/broadcasts", params={"type": "bogus"}).status_code == 400
    assert client.get("/broadcasts", params={"mode": "type"}).status_code == 400
    assert client.get("/broadcasts", params={"date_from": ago(1), "date_to": ago(5)}).status_code == 400
    t = client.get("/transcript", params={"source_file": b["source_file"]}).json()
    assert len(t["segments"]) == 2 and t["broadcast"]["broadcast_type"] == "news-at-nine"
    assert t["segments"][1]["audio_start"] == 5.0  # timestamps preserved
    audio = client.get(f"/audio/{b['first_faiss_id']}")
    assert audio.status_code == 200 and audio.content == b"ID3fake"


def test_api_coverage_analytics_and_pages(client, bc):
    c = client.get("/coverage").json()
    assert c["actual"]["latest"] == bc["d0"] and c["reporting_window"]["start"] == config.REPORT_WINDOW_START
    s = client.get("/analytics/summary", params={"date_from": ago(8), "date_to": ago(0)}).json()
    assert s["dataset"]["broadcasts"] == 6 and s["rag"]["queries"] == 0
    assert client.get("/analytics/summary", params={"date_from": ago(1), "date_to": ago(3)}).status_code == 400
    assert client.get("/analytics/timeseries", params={"interval": "week"}).json()["interval"] == "week"
    assert client.get("/analytics/distributions", params={"metric": "chunks"}).json()["metric"] == "chunks"
    assert "funnel" in client.get("/analytics/jobs").json()
    assert "latency" in client.get("/analytics/rag").json()
    page = client.get("/analytics").text
    assert "Analytics" in page and "/analytics/summary" in page and "charts.js" in page
    nr = client.get("/newsroom").text
    assert 'id="bcView"' in nr and "/broadcasts?" in nr and 'id="rangeBox"' in nr


def test_api_ingest_preview_is_validated_and_admin_only(client, monkeypatch):
    monkeypatch.setattr(config, "INGEST_ADMIN_TOKEN", "s3cret")
    bad = client.post("/ingest/preview", json={"date_from": ago(1), "date_to": ago(5), "types": ["midday"]},
                      headers={"X-Admin-Token": "s3cret"})
    assert bad.status_code == 400 and bad.json()["errors"]
    assert client.post("/ingest/preview", json={"date_from": ago(5), "date_to": ago(1), "types": ["midday"]}
                       ).status_code == 401
    assert client.post("/ingest/confirm", json={"preview_id": "nope", "confirm": True},
                       headers={"X-Admin-Token": "s3cret"}).status_code == 404
    noa._previews["PRV-x"] = {"status": "ready", "blocking": [], "nothing_to_do": False, "created": 0,
                              "date_from": ago(2), "date_to": ago(2), "types": ["midday"], "_items": [],
                              "include_articles": False, "include_indexed": False}
    assert client.post("/ingest/confirm", json={"preview_id": "PRV-x", "confirm": False},
                       headers={"X-Admin-Token": "s3cret"}).status_code == 400
    assert client.post("/ingest/cancel/NOA-none", headers={"X-Admin-Token": "s3cret"}).status_code == 404
    noa._previews.pop("PRV-x", None)
