"""Detailed articles: complete ingestion, full-article context, answer depth by question type, API."""

import re

import pytest
from fastapi.testclient import TestClient

import config
import news_articles
import rag_engine
import vector_store

BATTERY = "solapur-battery-storage-pilot"
GRID = "western-grid-evening-peak"


@pytest.fixture
def articles(env):
    summary = news_articles.ingest()
    return summary


def _words(answer: str) -> int:
    return len(re.sub(r"\[S\d+\]", "", answer).split())


def _article_ids(res):
    return [s.get("article_id") for s in res["sources"]]


def test_dataset_shape_and_length():
    arts = news_articles.load_articles()
    assert 8 <= len(arts) <= 12
    for a in arts:
        for key in ("headline", "genre", "published", "time", "summary", "body", "background",
                    "key_developments", "entities", "implications", "takeaways"):
            assert a.get(key), (a["id"], key)
        assert a["demo"] is True and "AIR" not in a["source_name"]
        assert 350 <= a["word_count"] <= 650, (a["id"], a["word_count"])


def test_chunks_cover_every_word_and_respect_sections():
    for a in news_articles.load_articles():
        chunks = news_articles.chunk_article(a)
        news_articles.verify_coverage(a, chunks)  # raises if anything was dropped
        assert [c["section"] for c in chunks][0] == "Summary"
        assert {"Report", "Background", "Implications", "Key takeaways"} <= {c["section"] for c in chunks}


def test_verify_coverage_detects_truncation():
    a = news_articles.load_articles()[0]
    chunks = news_articles.chunk_article(a)
    chunks[1] = dict(chunks[1], text=chunks[1]["text"][:40])
    with pytest.raises(ValueError):
        news_articles.verify_coverage(a, chunks)


def test_ingest_preserves_metadata_and_full_text(articles):
    arts = news_articles.load_articles()
    assert articles["articles"] == len(arts)
    db = vector_store._get_db()
    rows = db.execute("SELECT * FROM segments WHERE article_id = ? ORDER BY id", (BATTERY,)).fetchall()
    db.close()
    a = news_articles.get_article(BATTERY)
    assert rows and all(r["ingest_mode"] == "article" and r["genre"] == a["genre"] for r in rows)
    assert rows[0]["date"] == a["published"] and rows[0]["hour"] == a["hour"]
    assert [r["section"] for r in rows] == [c["section"] for c in news_articles.chunk_article(a)]
    stored = vector_store.get_article(BATTERY)
    assert stored["word_count"] == len(news_articles.full_text(a).split())
    assert a["implications"][-1] in stored["full_text"]
    # Articles are listed in the newsroom, not among broadcasts.
    assert not any(s["source_file"].startswith("articles/") for s in vector_store.list_sources())
    assert vector_store.get_stats()["articles"] == len(arts)


def test_article_source_carries_whole_article(articles):
    res = rag_engine.answer_query("How does the Solapur battery storage pilot work?")
    s1 = res["sources"][0]
    assert s1["kind"] == "article" and s1["article_id"] == BATTERY
    full = vector_store.get_article(BATTERY)["full_text"]
    assert s1["context"] == full and "Background:" in full and "Implications:" in full
    assert "article" not in s1  # structured copy is not sent to clients


def test_explanatory_answer_is_detailed_structured_and_cited(articles):
    res = rag_engine.answer_query(
        "Explain the background, key developments and implications of the Deccan reservoir shortage")
    assert res["question_type"] == "explain" and res["mode"] == "extractive"
    assert _article_ids(res)[0] == "deccan-reservoirs-late-monsoon"
    ans = res["answer"]
    assert 300 <= _words(ans) <= 900
    for heading in ("### Direct answer", "### Background and context", "### Key developments", "### Implications",
                    "### Sources"):
        assert heading in ans
    body = [ln for ln in ans.splitlines() if ln.strip() and not ln.startswith("###") and not ln.startswith("From ")]
    assert all(re.search(r"\[S\d+\]", ln) for ln in body)
    # every cited passage exists verbatim in its source article
    by_id = {s["id"]: s for s in res["sources"]}
    for ln in body:
        sid = re.findall(r"\[(S\d+)\]", ln)[0]
        text = re.sub(r"\s*\[S\d+\]", "", ln).lstrip("- ").strip()
        if ln.startswith("- [S"):
            continue  # Sources list line
        text = re.sub(r'^"[^"]+" \(\d{4}-\d{2}-\d{2}\): ', "", text)
        assert text in by_id[sid]["context"], text


def test_simple_question_stays_short(articles):
    res = rag_engine.answer_query("How many fever-ward beds were added for dengue patients?")
    assert res["question_type"] == "simple"
    assert _words(res["answer"]) <= 100 and "###" not in res["answer"]
    assert "320" in res["answer"]


def test_comparison_uses_both_articles_and_attributes_claims(articles):
    res = rag_engine.answer_query("Compare the Solapur battery storage pilot with the record evening power demand")
    ids = _article_ids(res)
    assert BATTERY in ids and GRID in ids
    ans = res["answer"]
    cited = set(re.findall(r"\[(S\d+)\]", ans))
    sid = {s["article_id"]: s["id"] for s in res["sources"] if s.get("article_id")}
    assert {sid[BATTERY], sid[GRID]} <= cited
    assert ans.count("### Key developments") == 1 and _words(ans) >= 300


def test_gap_question_lists_what_sources_do_not_say(articles):
    res = rag_engine.answer_query("What information is missing about the battery pilot's costs?")
    assert res["question_type"] == "gap"
    gap = res["answer"].split("### What the sources do not say", 1)[1]
    assert "disclosed the project cost" in gap


def test_genre_overview_covers_each_story(articles):
    res = rag_engine.answer_query("Summarise the major health developments")
    ids = set(_article_ids(res))
    assert {"dengue-fever-wards-extended-rains", "mobile-screening-vans-rural"} <= ids


def test_question_classification():
    c = rag_engine.classify_question
    assert c("Why did dengue cases rise?") == "explain"
    assert c("Compare the two energy stories") == "compare"
    assert c("What is not disclosed about the port pilot?") == "gap"
    assert c("Summarise the sports news") == "overview"
    assert c("Who is the chief engineer of the water board?") == "simple"


def test_partial_answer_noting_a_gap_is_not_a_refusal(articles, monkeypatch):
    long = ("### Direct answer\nThe battery shifts solar output to the evening [S1]. "
            "The sources do not contain enough information on its cost [S1].\n" + "More detail [S1]. " * 30)
    monkeypatch.setattr(rag_engine, "_generate_with_gemini", lambda c, q, h=None, guidance="": long)
    res = rag_engine.answer_query("Explain the Solapur battery pilot")
    assert not res["insufficient_context"] and res["mode"] == "generative"


def test_generator_receives_length_guidance(articles, monkeypatch):
    seen = {}

    def fake(context, q, h=None, guidance=""):
        seen["guidance"], seen["context"] = guidance, context
        return "Answer [S1]."
    monkeypatch.setattr(rag_engine, "_generate_with_gemini", fake)
    rag_engine.answer_query("Why did evening power demand hit a record?")
    assert "300-600 words" in seen["guidance"] and "### Implications" in seen["guidance"]
    assert "Implications:" in seen["context"] and "illustrative demo content" in seen["context"]


@pytest.fixture
def client(articles, monkeypatch):
    import chatbot_api
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    with TestClient(chatbot_api.app) as c:
        yield c


def test_articles_api_previews_and_detail(client):
    d = client.get("/articles").json()
    assert len(d["articles"]) == len(news_articles.load_articles()) and d["disclaimer"]
    assert all("body" not in a and a["summary"] for a in d["articles"])
    # Genre filters use the taxonomy: an article matches on its primary genre or a secondary tag.
    health = client.get("/articles", params={"genre": "health"}).json()
    assert health["total"] == len(health["articles"]) >= 2
    assert all(a["genre_id"] == "health" or "health" in [t["id"] for t in a["genre_tags"]] for a in health["articles"])
    legacy_health = {a["id"] for a in news_articles.load_articles() if a["genre"] == "Health"}
    assert legacy_health <= {a["id"] for a in health["articles"]}
    assert client.get("/articles", params={"q": "lithium iron phosphate"}).json()["articles"][0]["id"] == BATTERY
    full = client.get(f"/articles/{BATTERY}").json()
    assert full["article"]["body"] and full["related"] and all(r["genre_id"] == full["article"]["genre_id"]
                                                               for r in full["related"])
    assert client.get("/articles/nope").status_code == 404


def test_chat_reports_question_type(client):
    d = client.post("/chat", json={"message": "Why did dengue cases rise this year?"}).json()
    assert d["question_type"] == "explain" and d["sources"][0]["kind"] == "article"


def test_weak_best_match_is_reported_as_not_covered(articles, monkeypatch):
    q = "How does the Solapur battery storage pilot work?"
    top = rag_engine.retrieve(q)["hits"][0]["score"]
    monkeypatch.setattr(config, "RAG_MIN_TOP_SCORE", top + 0.01)
    res = rag_engine.answer_query(q)
    assert res["insufficient_context"] and res["sources"] == []


def test_genre_scope_is_a_strict_filter(articles):
    res = rag_engine.answer_query("What happened with the grid and electricity demand?", genre="Health")
    assert all(s["genre"] == "Health" for s in res["sources"])
    res = rag_engine.answer_query("Why did dengue cases rise?", genre="Health")
    assert res["sources"] and {s["genre"] for s in res["sources"]} == {"Health"}
    assert res["filters_applied"]["genre"] == "Health"


def test_chat_accepts_genre_scope(client):
    d = client.post("/chat", json={"message": "Summarise the developments", "genre": "Sports"}).json()
    assert d["filters"]["genre"] == "Sports" and {s["genre"] for s in d["sources"]} == {"Sports"}
