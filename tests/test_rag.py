"""Retrieval, query understanding, grounding and citation tests (realtime prompt tests 9 & 14)."""

import datetime as dt

import rag_engine
from conftest import add


def _files(res):
    return {s["source_file"] for s in res["sources"]}


def test_parse_relative_dates_use_real_today():
    t = dt.date(2026, 11, 2)
    f = rag_engine.parse_query("What happened yesterday?", t)
    assert (f.date_from, f.date_to) == ("2026-11-01", "2026-11-01") and f.topic == ""
    f = rag_engine.parse_query("what happened on September 20?", t)
    assert f.date_from == "2026-09-20"
    f = rag_engine.parse_query("news from 20 December", dt.date(2027, 1, 5))
    assert f.date_from == "2026-12-20"  # future date in same year -> previous year
    f = rag_engine.parse_query("what happened last week", t)
    assert (f.date_from, f.date_to) == ("2026-10-26", "2026-11-02")


def test_parse_time_of_day_and_genre():
    t = dt.date(2026, 11, 2)
    f = rag_engine.parse_query("What happened around 7 PM yesterday?", t)
    assert (f.hour_from, f.hour_to) == (1830, 1959)
    f = rag_engine.parse_query("Show sports stories between 6 PM and 8 PM", t)
    assert (f.hour_from, f.hour_to, f.genre) == (1800, 2059, "Sports")
    f = rag_engine.parse_query("What happened at 10:30 AM?", t)
    assert (f.hour_from, f.hour_to) == (1030, 1129)
    # topic words are a soft hint, not a hard filter
    f = rag_engine.parse_query("Prime Minister semiconductor plant", t)
    assert f.genre is None and f.genre_hint == "Politics"


def test_yesterday_browse_only_returns_yesterday(corpus):
    res = rag_engine.answer_query("What happened yesterday?")
    assert res["sources"] and not res["insufficient_context"]
    assert {s["date"] for s in res["sources"]} == {corpus["yesterday"]}


def test_today_date_plus_genre_and_specific_date(corpus):
    res = rag_engine.answer_query("Give me yesterday's sports news")
    assert {s["genre"] for s in res["sources"]} == {"Sports"}
    assert {s["date"] for s in res["sources"]} == {corpus["yesterday"]}
    res = rag_engine.answer_query("What happened today?")
    assert {s["date"] for s in res["sources"]} == {corpus["today"]}
    res = rag_engine.answer_query("what happened on 20 September 2026?")
    assert {s["date"] for s in res["sources"]} == {"2026-09-20"}


def test_time_of_day_filter(corpus):
    res = rag_engine.answer_query("What happened around 7 PM yesterday?")
    assert res["sources"]
    assert all(1830 <= s["hour"] <= 1959 for s in res["sources"])


def test_semantic_topic_finds_right_source(corpus):
    res = rag_engine.answer_query("Tell me about the semiconductor story")
    assert res["sources"][0]["source_file"] == "text/t/Live.txt"
    assert "semiconductor" in res["sources"][0]["context"].lower()


def test_unrelated_question_is_insufficient(corpus):
    res = rag_engine.answer_query("What did the volcano in Iceland do to European flights?")
    assert res["insufficient_context"] and res["sources"] == []
    assert "don't have enough information" in res["answer"]


def test_empty_period_reports_coverage(corpus):
    res = rag_engine.answer_query("What happened on 1 January 2020?")
    assert res["insufficient_context"]
    assert "covers" in res["answer"]


def test_newly_added_segment_is_immediately_retrievable(corpus):
    """Realtime test 9: a segment stored a moment ago is searchable at once."""
    add("A new bullet train corridor between Mumbai and Ahmedabad opened for trials.", corpus["today"], 1101,
        "text/t/Live2.txt", mode="live")
    res = rag_engine.answer_query("bullet train corridor")
    assert res["sources"][0]["source_file"] == "text/t/Live2.txt"
    assert res["sources"][0]["ingest_mode"] == "live"


def test_extractive_answer_cites_every_line(corpus):
    res = rag_engine.answer_query("cricket match result")
    assert res["mode"] == "extractive"
    lines = [ln for ln in res["answer"].splitlines() if ln.startswith("- ")]
    assert lines and all("[S" in ln for ln in lines)
    assert set(res["cited"]) <= {s["id"] for s in res["sources"]}


def test_citation_validation_removes_fabricated_ids():
    sources = [{"id": "S1"}, {"id": "S2"}]
    text, cited, removed = rag_engine.validate_citations("A [S1]. B [S7]. C [s2].", sources)
    assert cited == ["S1", "S2"] and removed == ["S7"] and "[S7]" not in text


def test_generative_path_validates_citations(corpus, monkeypatch):
    monkeypatch.setattr(rag_engine, "_generate_with_gemini", lambda c, q, h=None, guidance="": "Rates unchanged [S1][S9].")
    res = rag_engine.answer_query("repo rate inflation")
    assert res["mode"] == "generative" and "[S9]" not in res["answer"] and res["cited"] == ["S1"]


def test_model_insufficient_reply_is_reported(corpus, monkeypatch):
    monkeypatch.setattr(rag_engine, "_generate_with_gemini",
                        lambda c, q, h=None, guidance="": "The broadcasts I have do not contain enough information to answer this.")
    res = rag_engine.answer_query("cricket match")
    assert res["insufficient_context"]


def test_sources_carry_traceability_metadata(corpus):
    s = rag_engine.answer_query("ISRO satellite launch")["sources"][0]
    for key in ("id", "faiss_id", "source_name", "source_file", "date", "time", "audio_start", "audio_end", "context"):
        assert key in s
    assert s["date"] == "2026-09-20" and s["time"] == "08:00"


def test_timeline_groups_by_date(corpus):
    tl = rag_engine.timeline_query("sports")
    assert tl["groups"] and all(g["segments"] for g in tl["groups"])


def test_part_of_day_word_is_a_filter_only_when_used_as_time():
    t = dt.date(2026, 11, 2)
    assert rag_engine.parse_query("What was said this evening?", t).hour_from == 1700
    assert rag_engine.parse_query("evening news from yesterday", t).hour_from == 1700
    f = rag_engine.parse_query("Why did evening power demand hit a record?", t)
    assert f.hour_from is None and "power" in f.topic


def test_gemini_falls_back_to_next_model_on_quota_error(corpus, monkeypatch):
    import sys
    import types

    import config
    calls = []

    class Resp:
        text = "Rates unchanged [S1]."

    class Model:
        def __init__(self, name, system_instruction=None):
            self.name = name

        def generate_content(self, contents, generation_config=None, request_options=None):
            calls.append(self.name)
            if self.name == "primary":
                raise RuntimeError("429 You exceeded your current quota")
            return Resp()

    fake = types.SimpleNamespace(configure=lambda api_key: None, GenerativeModel=Model)
    google = types.ModuleType("google")
    google.generativeai = fake
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)
    monkeypatch.setattr(config, "GEMINI_MODEL", "primary")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", ["lite"])
    monkeypatch.setattr(rag_engine, "_generation_mode", "LIVE")
    monkeypatch.setattr(rag_engine, "_model_cooldown", {})
    res = rag_engine.answer_query("repo rate inflation")
    assert calls == ["primary", "lite"] and res["mode"] == "generative" and res["model"] == "lite"
    rag_engine.answer_query("repo rate inflation")
    assert calls[-1] == "lite" and calls.count("primary") == 1  # exhausted model is skipped while cooling down


def test_grouped_citations_are_split_and_validated():
    sources = [{"id": "S1"}, {"id": "S2"}]
    text, cited, removed = rag_engine.validate_citations("Both rose [S1, S2]. Also [S2; S9].", sources)
    assert text == "Both rose [S1][S2]. Also [S2]." and cited == ["S1", "S2"] and removed == ["S9"]
