"""API tests: chat response shape, health, transcript, audio (no placeholder audio), stream control."""

import pytest
from fastapi.testclient import TestClient

import config


@pytest.fixture
def client(corpus, monkeypatch):
    import chatbot_api
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    with TestClient(chatbot_api.app) as c:
        yield c


def test_chat_returns_grounded_answer(client):
    d = client.post("/chat", json={"message": "Tell me about the semiconductor story"}).json()
    assert d["sources"] and d["sources"][0]["id"] == "S1"
    assert d["trace_id"].startswith("REQ-") and d["mode"] == "extractive" and "[S1]" in d["reply"]


def test_chat_insufficient(client):
    d = client.post("/chat", json={"message": "volcano eruption in Iceland"}).json()
    assert d["insufficient_context"] and d["sources"] == []


def test_health_reports_stream_and_corpus(client):
    h = client.get("/health").json()
    for key in ("status", "stream_connected", "last_audio_received", "last_transcription",
                "segments_today", "last_segment_id", "models", "corpus_date_range"):
        assert key in h
    assert h["segments_today"] == 2 and h["stream_running"] is False


def test_sources_and_transcript(client):
    srcs = client.get("/sources").json()["sources"]
    assert any(s["source_file"] == "text/t/Live.txt" for s in srcs)
    t = client.get("/transcript", params={"source_file": "text/t/Live.txt"}).json()
    assert len(t["segments"]) == 2 and t["segments"][0]["audio_start"] is not None
    assert client.get("/transcript", params={"source_file": "nope.txt"}).status_code == 404


def test_audio_404_when_file_missing(client, corpus):
    r = client.get(f"/audio/{corpus['ids'][0]}")
    assert r.status_code == 404  # no audio stored -> no placeholder


def test_audio_serves_real_file(client, corpus, env):
    p = env / "output" / "audio" / "t"
    p.mkdir(parents=True)
    (p / "Live.mp3").write_bytes(b"ID3real")
    r = client.get(f"/audio/{corpus['ids'][5]}")
    assert r.status_code == 200 and r.content == b"ID3real"


def test_stream_start_without_url_is_clear_error(client):
    r = client.post("/stream/start", json={})
    assert r.status_code == 400 and "LIVE_STREAM_URL" in r.json()["error"]
    assert client.post("/stream/stop").json()["status"] == "not_running"


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VaniStream" in r.text


def test_home_and_newsroom_are_separate_pages(client):
    home = client.get("/").text
    newsroom = client.get("/newsroom")
    assert newsroom.status_code == 200 and 'id="newsView"' in newsroom.text and 'id="askView"' in newsroom.text
    assert 'id="newsView"' not in home
    # Both pages: brand links home, theme toggle, shared theme assets.
    for page in (home, newsroom.text):
        assert 'class="brand-link" href="/"' in page and "theme-toggle" in page
        assert "/static/theme.css" in page and "/static/theme.js" in page
    assert client.get("/static/theme.css").status_code == 200
    assert "vaniToggleTheme" in client.get("/static/theme.js").text


def test_home_page_content_is_accurate(client):
    home = client.get("/").text
    # Counts come from the database at load time, never hardcoded.
    for el in ('id="stArticles"', 'id="stBroadcasts"', 'id="stChunks"', 'id="stLive"', 'id="stCoverage"'):
        assert el in home
    assert "fetch('/ingest/status')" in home and 'data-count="95"' not in home
    for label in ("Articles", "Broadcasts", "Indexed chunks", "Coverage", "From live audio"):
        assert label in home
    assert 'href="/newsroom"' in home and "Enter Newsroom" in home and "Explore Stories" in home
    assert "Powered by India's Public Broadcasting Network" in home
    assert 'href="https://newsonair.gov.in/"' in home
    assert "not affiliated with" in home.lower()
    assert ("News content sourced from NewsOnAIR. AIR is an independent academic project built to explore "
            "audio-based news retrieval and contextual knowledge discovery.") in home
    assert "TODO" not in home
