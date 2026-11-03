"""NewsOnAIR collector + incremental indexing (offline: network, ASR and models are stand-ins).

Fixtures mirror the real page/API shapes observed on newsonair.gov.in (Nov 2026).
"""

import json
import os
import urllib.parse

import pytest
from fastapi.testclient import TestClient

import config
import newsonair_ingest as noa
import rag_engine
import vector_store


def _share(bid, title, cat, lang, date, audio, page):
    text = f"{title} | Category: {cat} | Language: {lang} | Date: {date} | Audio: {audio} {page}"
    fb = urllib.parse.quote(f"https://newsonair.gov.in/share-bulletin/?id={bid}", safe="")
    return (f'<div class="bulletin-share"><a href="https://www.facebook.com/sharer/sharer.php?u={fb}&quote=x">fb</a>'
            f'<a href="javascript:void(0);" onclick="navigator.clipboard.writeText(\'{text}\'); alert(1);">ig</a></div>')


BULLETIN_PAGE = "<html><body>" + "".join([
    _share(1949031, "समाचार संध्या", "समाचार संध्या", "Hindi", "02 November 2026 20:45",
           "https://newsonair.gov.in/wp-content/uploads/2026/11/Samachar-Sandhya.mp3",
           "https://newsonair.gov.in/listen-news/x-954/"),
    _share(1949020, "Hourly News", "Hourly News", "English", "02 November 2026 20:05",
           "https://newsonair.gov.in/wp-content/uploads/2026/11/English-2005.mp3",
           "https://newsonair.gov.in/listen-news/hourly-news-16462/"),
    _share(1949020, "Hourly News", "Hourly News", "English", "02 November 2026 20:05",  # repeated block
           "https://newsonair.gov.in/wp-content/uploads/2026/11/English-2005.mp3",
           "https://newsonair.gov.in/listen-news/hourly-news-16462/"),
    _share(1948990, "Midday News", "Midday News", "English", "02 November 2026 14:00",
           "https://newsonair.gov.in/wp-content/uploads/2026/11/middday.mp3",
           "https://newsonair.gov.in/listen-news/midday-news-995/"),
]) + "</body></html>"


def _post(pid, slug, title, paras, date="2026-11-02T20:30:21", modified="2026-11-02T15:16:00"):
    return {"id": pid, "date": date, "modified_gmt": modified, "link": f"https://newsonair.gov.in/{pid}/",
            "title": {"rendered": title}, "lang": "en", "author": 1, "categories": [1],
            "excerpt": {"rendered": f"<p>{paras[0][:60]}</p>"},
            "content": {"rendered": "<div> </div>" + "".join(f"<div><span>{p}</span></div>" for p in paras)},
            "_slug": slug, "_category_slugs": [slug]}


BUSINESS = _post(1948469, "business", "Capital expenditure remains important part of govt&#8217;s growth strategy",
                 ["Union Finance Minister Nirmala Sitharaman said the government will continue to prioritise "
                  "capital expenditure to sustain growth while ensuring fiscal discipline.",
                  "She identified credit access, insurance and pensions as priorities for financial inclusion."])
NATIONAL = _post(1948355, "national", "HM directs FRROs in all states by 2027",
                 ["The Home Minister directed that Foreigners Regional Registration Offices be set up in all states "
                  "by 2027 to promote India as an attractive destination."])


def test_bulletin_page_parsing_keeps_ids_metadata_and_dedups():
    items = noa.parse_bulletin_page(BULLETIN_PAGE)
    assert [i["bulletin_id"] for i in items] == ["1949031", "1949020", "1948990"]
    hourly = items[1]
    assert hourly["language"] == "English" and hourly["title"] == "Hourly News"
    assert hourly["broadcast_at"] == "2026-11-02T20:05:00"
    assert hourly["audio_url"].endswith("English-2005.mp3")
    assert hourly["page_url"] == "https://newsonair.gov.in/listen-news/hourly-news-16462/"


def test_html_to_paragraphs_and_article_mapping():
    rec = noa.parse_article(BUSINESS, "2026-11-02T15:30:00Z")
    assert rec["id"] == "noa-post-1948469" and rec["headline"].startswith("Capital expenditure remains important")
    assert "’" in rec["headline"] and "&#" not in rec["headline"]
    assert rec["genre"] == "Economy" and rec["source_category"] == "Business"
    assert rec["published"] == "2026-11-02" and rec["time"] == "20:30" and rec["hour"] == 2030
    assert len(rec["body"]) == 2 and " " not in " ".join(rec["body"])
    assert rec["source_url"] == "https://newsonair.gov.in/1948469/" and rec["content_limited"] is True
    nat = noa.parse_article(NATIONAL, "x")
    assert nat["genre"] is None and nat["source_category"] == "National"  # genre comes from the classifier


def test_scale_plan_respects_total():
    plan = noa.scale_plan(10)
    assert sum(plan.values()) <= 10 and "national" in plan


def _counts():
    return {k: 0 for k in noa.COUNT_KEYS}


def test_incremental_article_indexing_dedup_update_and_failure(env):
    recs = [noa.parse_article(p, "2026-11-02T15:30:00Z") for p in (BUSINESS, NATIONAL)]
    noa._classify_missing_genres(recs)  # stand-in classifier
    c = _counts()
    for r in recs:
        noa.index_article(r, c, "JOB-1")
    assert c["indexed_new"] == 2 and c["failed"] == 0
    chunks_after_first = vector_store.get_dataset_stats()["chunks"]

    # Same content again: nothing re-embedded.
    c2 = _counts()
    for r in recs:
        noa.index_article(r, c2, "JOB-2")
    assert c2["duplicates_skipped"] == 2 and c2["embeddings"] == 0
    assert vector_store.get_dataset_stats()["chunks"] == chunks_after_first

    # Publisher edits one article: old chunks are retired, the new version is indexed.
    changed = dict(recs[0], body=recs[0]["body"] + ["The minister also announced an AI skills centre in every district."])
    c3 = _counts()
    noa.index_article(changed, c3, "JOB-3")
    assert c3["updated"] == 1
    db = vector_store._get_db()
    rows = db.execute("SELECT active, COUNT(*) FROM segments WHERE doc_id = ? GROUP BY active",
                      ("noa-post-1948469",)).fetchall()
    db.close()
    assert dict((r[0], r[1]) for r in rows)[0] >= 1  # retired chunks kept for FAISS id stability
    res = rag_engine.answer_query("AI skills centre in every district")
    assert res["sources"] and res["sources"][0]["doc_id"] == "noa-post-1948469"
    assert "AI skills centre" in res["sources"][0]["context"]

    # A record with no text fails, is tracked, and is retried up to the limit.
    empty = dict(recs[1], id="noa-post-1", body=[], headline="Empty")
    for _ in range(config.NEWSONAIR_MAX_ATTEMPTS):
        noa.index_article(empty, _counts(), "JOB-4")
    st = vector_store.get_state("noa-post-1")
    assert st["status"] == "failed" and st["attempts"] == config.NEWSONAIR_MAX_ATTEMPTS
    c5 = _counts()
    noa.index_article(empty, c5, "JOB-5")
    assert c5["gave_up"] == 1


def test_article_chunks_preserve_metadata(env):
    rec = noa.parse_article(BUSINESS, "2026-11-02T15:30:00Z")
    noa.index_article(rec, _counts(), "JOB-1")
    db = vector_store._get_db()
    row = db.execute("SELECT * FROM segments WHERE doc_id = 'noa-post-1948469' ORDER BY chunk_index").fetchone()
    db.close()
    assert row["content_type"] == "article" and row["source_url"] == "https://newsonair.gov.in/1948469/"
    assert row["source_name"] == "NewsOnAIR" and row["headline"].startswith("Capital expenditure")
    assert row["genre"] == "Economy" and row["source_category"] == "Business" and row["language"] == "en"
    assert row["published_at"] == "2026-11-02T20:30:21" and row["indexed_at"] and row["chunk_index"] == 0


class FakeResult:
    def __init__(self, segments):
        self._segs = segments

    @property
    def text(self):
        return " ".join(s["text"] for s in self._segs)

    def to_json(self):
        return {"segments": self._segs}


def test_bulletin_transcribed_indexed_with_timestamps(env, monkeypatch):
    import transcriber
    words = []
    t = 0.0
    sentences = ["Cyclone Murasu is likely to make landfall on the Andhra Pradesh coast tonight.",
                 "The state government has evacuated people from low lying areas."]
    segs = []
    for sent in sentences:
        ws = []
        for w in sent.split():
            ws.append([t, t + 0.4, " " + w])
            t += 0.5
        segs.append({"start": ws[0][0], "end": ws[-1][1], "text": sent, "words": ws})
        words += ws

    class FakeASR:
        def transcribe(self, path):
            return FakeResult(segs)

    calls = []
    monkeypatch.setattr(transcriber, "get_transcriber", lambda *a, **k: (calls.append(1), FakeASR())[1])
    b = noa.parse_bulletin_page(BULLETIN_PAGE)[1]
    audio = env / "output" / "audio" / "2026-11-02"
    audio.mkdir(parents=True)
    (audio / f"NOA-Hourly-News-2005-{b['bulletin_id']}.mp3").write_bytes(b"ID3fake")  # cached: no download

    class NoNet:
        def get(self, *a, **k):
            raise AssertionError("network must not be used for a cached file")

    c = _counts()
    noa.index_bulletin(b, NoNet(), c, "JOB-1")
    assert c["broadcasts_indexed"] == 1 and c["transcriptions_ok"] == 1 and c["failed"] == 0
    res = rag_engine.answer_query("Where will cyclone Murasu make landfall?")
    s1 = res["sources"][0]
    assert s1["kind"] == "broadcast" and s1["doc_id"] == f"noa-bulletin-{b['bulletin_id']}"
    assert s1["audio_start"] == 0.0 and s1["audio_end"] is not None and s1["has_audio"]
    assert s1["source_url"].endswith("/listen-news/hourly-news-16462/") and s1["time"] == "20:05"
    assert s1["source_category"] == "Hourly News"
    # Second run: transcript already on disk and state indexed -> skipped, ASR not called again.
    c2 = _counts()
    noa.index_bulletin(b, NoNet(), c2, "JOB-2")
    assert c2["duplicates_skipped"] == 1 and len(calls) == 1


def test_national_news_uses_publisher_category(env):
    recs = [noa.parse_article(p, "x") for p in (BUSINESS, NATIONAL)]
    noa._classify_missing_genres(recs)
    for r in recs:
        noa.index_article(r, _counts(), "J")
    f = rag_engine.parse_query("Summarize the latest national news")
    assert f.source_category == "National"
    res = rag_engine.answer_query("Summarize the latest national news")
    assert {s["source_category"] for s in res["sources"]} == {"National"}
    assert rag_engine.parse_query("national highway project").source_category is None


@pytest.fixture
def client(env, monkeypatch):
    import chatbot_api
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    with TestClient(chatbot_api.app) as c:
        yield c


def test_ingest_status_reports_database_values(client):
    recs = [noa.parse_article(p, "x") for p in (BUSINESS, NATIONAL)]
    noa._classify_missing_genres(recs)
    for r in recs:
        noa.index_article(r, _counts(), "J")
    d = client.get("/ingest/status").json()
    assert d["dataset"]["articles"] == 2 and d["dataset"]["documents"]["indexed"] == 2
    assert d["dataset"]["source_categories"] == {"Business": 1, "National": 1}
    assert d["source"]["url"].startswith("https://newsonair.gov.in")
    arts = client.get("/articles").json()
    assert {a["id"] for a in arts["articles"]} == {"noa-post-1948469", "noa-post-1948355"}
    assert "NewsOnAIR" in arts["disclaimer"]
    full = client.get("/articles/noa-post-1948469").json()["article"]
    assert full["source_url"] and full["body"] and full["demo"] is False


def test_ingest_trigger_is_locked_down(client, monkeypatch):
    monkeypatch.setattr(config, "INGEST_ADMIN_TOKEN", "")
    assert client.post("/ingest/run", json={}).status_code == 403
    monkeypatch.setattr(config, "INGEST_ADMIN_TOKEN", "s3cret")
    assert client.post("/ingest/run", json={}, headers={"X-Admin-Token": "nope"}).status_code == 401
    started = {}
    monkeypatch.setattr(noa, "run_in_background", lambda **kw: started.update(kw) or "NOA-test")
    r = client.post("/ingest/run", json={"articles": 5000, "broadcasts": 99}, headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200 and started["articles"] == config.INGEST_MAX_ARTICLES
    assert started["broadcasts"] == config.INGEST_MAX_BROADCASTS and started["trigger"] == "dashboard"


def test_question_about_absent_topic_is_not_covered(env, monkeypatch):
    for p in (BUSINESS, NATIONAL):
        r = noa.parse_article(p, "x")
        noa._classify_missing_genres([r])
        noa.index_article(r, _counts(), "J")
    monkeypatch.setattr(config, "RAG_STRONG_MATCH", 1.01)  # treat every match as moderate
    res = rag_engine.answer_query("What is the price of bitcoin?")
    assert res["insufficient_context"] and res["sources"] == []
    # A question whose topic words are in the corpus still gets answered.
    res = rag_engine.answer_query("capital expenditure growth strategy")
    assert not res["insufficient_context"] and res["sources"][0]["doc_id"] == "noa-post-1948469"
