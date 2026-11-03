"""Simplified broadcast UI, live-ingestion controls, stream validation and per-type coverage (offline)."""

import pytest
from fastapi.testclient import TestClient

import analytics
import config
import live_stream_ingest as live
import vector_store
from test_broadcasts_analytics import ago, bc  # noqa: F401  (fixture)


@pytest.fixture
def client(bc, monkeypatch):  # noqa: F811
    import chatbot_api
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    monkeypatch.setattr(config, "INGEST_ADMIN_TOKEN", "")
    with TestClient(chatbot_api.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Layout: no broadcast list in a left sidebar, broadcasts still reachable
# ---------------------------------------------------------------------------

def test_broadcasts_removed_from_sidebar_but_reachable(client):
    html = client.get("/newsroom").text
    assert 'class="rail"' not in html and 'id="sources"' not in html and "srcFilter" not in html
    assert 'id="tabBc"' in html and 'id="bcView"' in html and "h === 'broadcasts'" in html  # tab + #broadcasts link
    assert 'id="onair"' in html and "Start live ingestion" in html                   # live control kept
    for mode in ("All broadcasts", "Day-wise", "Midday", "News at Nine", "Hourly News", "Other / Unclassified"):
        assert mode in html
    assert client.get("/sources").status_code == 200 and client.get("/broadcasts/facets").status_code == 200
    assert '/newsroom#broadcasts' in client.get("/analytics").text


def test_all_mode_is_a_paginated_flat_list(client, bc):  # noqa: F811
    p1 = client.get("/broadcasts", params={"mode": "all", "limit": 4}).json()
    assert p1["total"] == 6 and len(p1["broadcasts"]) == 4 and p1["next_offset"] == 4
    keys = [(b["date"], b["time"]) for b in p1["broadcasts"]]
    assert keys == sorted(keys, reverse=True) and "segments" in p1["broadcasts"][0]
    p2 = client.get("/broadcasts", params={"mode": "all", "limit": 4, "offset": 4}).json()
    assert len(p2["broadcasts"]) == 2 and p2["next_offset"] is None
    # combined filters still apply in the flat list
    mid = client.get("/broadcasts", params={"mode": "all", "type": "midday", "genre": "defence"}).json()
    assert [b["bulletin_id"] for b in mid["broadcasts"]] == ["104"]
    assert client.get("/broadcasts", params={"mode": "all", "q": "cyclone"}).json()["total"] == 1


def test_transcript_only_loaded_on_open(client, bc):  # noqa: F811
    lst = client.get("/broadcasts", params={"mode": "all"}).json()["broadcasts"]
    assert all("segments" in b and "text" not in b for b in lst)  # counts only, no transcript text
    t = client.get("/transcript", params={"source_file": lst[0]["source_file"]}).json()
    assert t["segments"] and t["segments"][0]["text"]


# ---------------------------------------------------------------------------
# Coverage per content type
# ---------------------------------------------------------------------------

def test_broadcast_coverage_is_separate_from_articles_and_window(bc, monkeypatch):  # noqa: F811
    import newsonair_ingest as noa
    from test_newsonair import BUSINESS, _counts
    monkeypatch.setattr(config, "REPORT_WINDOW_START", ago(40))
    monkeypatch.setattr(config, "REPORT_WINDOW_END", ago(0))
    noa.index_article(noa.parse_article(dict(BUSINESS, date=f"{ago(30)}T10:00:00"), "x"), _counts(), "J")
    c = analytics.coverage()
    assert c["actual"]["broadcasts"] == {"earliest": bc["d2"], "latest": bc["d0"]}
    assert c["actual"]["articles"] == {"earliest": ago(30), "latest": ago(30)}
    assert c["reporting_window"]["start"] == ago(40)  # the window is reported, never as coverage
    only_b = analytics.coverage(analytics.Filters(content="broadcast"))
    assert only_b["actual"]["earliest"] == bc["d2"] and only_b["dates_with_records"] == 3


# ---------------------------------------------------------------------------
# Live ingestion controls
# ---------------------------------------------------------------------------

class FakeManager(live.LiveStreamManager):
    """The real manager (validation, start/stop, status) with the ingestion thread replaced by a wait:
    the status shown is exactly what the manager reports."""

    def _run(self, **opts):
        self.opts = opts
        self._stop_event.wait(10)


@pytest.fixture
def fake_live(client, monkeypatch):
    import chatbot_api
    m = FakeManager()
    monkeypatch.setattr(chatbot_api, "live_manager", m)
    return m


def test_start_status_stop_reflect_backend(client, fake_live, monkeypatch, tmp_path):
    audio = tmp_path / "bulletin.mp3"
    audio.write_bytes(b"ID3x")
    r = client.post("/stream/start", json={"url": str(audio), "programme": "Midday News"})
    assert r.status_code == 200 and r.json()["stream_id"] == fake_live.stream_id
    assert fake_live.opts["programme"] == "Midday News" and fake_live.url_source == "manual"
    st = client.get("/stream/status").json()
    assert st["running"] and st["state"] == "connecting" and not st["stream_connected"]  # not "on" yet
    fake_live.stream_connected = True
    assert client.get("/stream/status").json()["state"] == "on"
    assert client.post("/stream/start", json={"url": str(audio)}).status_code == 400  # already running
    assert client.post("/stream/stop", json={}).json()["status"] == "stopped"
    assert client.get("/stream/status").json()["state"] == "off"


def test_invalid_and_unreachable_streams_are_refused_with_reasons(client, fake_live, monkeypatch):
    r = client.post("/stream/start", json={"url": "ftp://example.com/x"})
    assert r.status_code == 400 and "http" in r.json()["error"]
    r = client.post("/stream/start", json={"url": "https://example.com/a.mp3", "programme": "Cricket Hour"})
    assert r.status_code == 400 and "Unknown broadcast type" in r.json()["error"]
    import requests

    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", refuse)
    r = client.post("/stream/start", json={"url": "https://radio.example.com/live.mp3"})
    assert r.status_code == 400 and "Could not connect to radio.example.com" in r.json()["error"]
    assert not fake_live.running
    chk = client.post("/stream/check", json={}).json()
    assert chk["ok"] is False and "not configured" in chk["detail"]


def test_stream_control_is_local_and_token_protected(client, fake_live, monkeypatch):
    monkeypatch.setattr(config, "INGEST_ADMIN_TOKEN", "s3cret")
    assert client.post("/stream/start", json={}).status_code == 401
    assert client.post("/stream/stop", json={}).status_code == 401
    assert client.post("/stream/check", json={}, headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert client.get("/health").json()["stream_control"]["requires_token"] is True


class _Resp:
    def __init__(self, status, text="", ctype="text/plain", headers=None):
        self.status_code, self.text = status, text
        self.headers = {"content-type": ctype, **(headers or {})}

    def iter_content(self, n):
        yield self.text.encode()

    def close(self):
        pass


def test_probe_detects_offline_hls_and_web_pages(monkeypatch):
    import requests
    master = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=32000\nlow.m3u8\n"
    media = "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:9.7,\nseg_1.aac\n#EXTINF:9.7,\nseg_2.aac\n"
    pages = {
        "https://cdn.example.com/live/master.m3u8": _Resp(200, master, "application/vnd.apple.mpegurl"),
        "https://cdn.example.com/live/low.m3u8": _Resp(200, media, "application/vnd.apple.mpegurl",
                                                       {"last-modified": "Fri, 09 Oct 2026 06:05:20 GMT"}),
        "https://cdn.example.com/live/seg_2.aac": _Resp(404),
        "https://example.com/": _Resp(200, "<!DOCTYPE html><html>", "text/html"),
        "https://example.com/ok.mp3": _Resp(200, "ID3....", "audio/mpeg"),
    }
    monkeypatch.setattr(requests, "get", lambda url, **k: pages[url])
    off = live.probe_stream("https://cdn.example.com/live/master.m3u8")
    assert not off["ok"] and off["kind"] == "hls-offline" and "2026-10-09" in off["detail"] and "404" in off["detail"]
    assert live.probe_stream("https://example.com/")["kind"] == "html"
    assert live.probe_stream("https://example.com/ok.mp3")["ok"]
    assert live.probe_stream("not a url")["kind"] == "invalid"


def test_stream_urls_are_never_exposed_with_credentials():
    url = "https://user:secret@cdn.example.com/a/b/live.m3u8?token=abc"
    shown = live.public_url(url)
    assert shown == "https://cdn.example.com/.../live.m3u8" and "secret" not in shown and "token" not in shown
    m = live.LiveStreamManager()
    m.running, m.url, m.started_at = True, url, config.now_local().isoformat(timespec="seconds")
    st = m.get_status()
    assert "secret" not in str(st) and "token=" not in str(st) and st["state"] == "connecting"
    m.running = False


def test_status_states_follow_the_ingestion_thread():
    m = live.LiveStreamManager()
    assert m.get_status()["state"] == "off"
    m.running, m.started_at = True, config.now_local().isoformat(timespec="seconds")
    assert m.get_status()["state"] == "connecting"
    m.stream_connected = True
    assert m.get_status()["state"] == "on" and m.get_status()["uptime_seconds"] is not None
    m.stream_connected, m.reconnects = False, 1
    assert m.get_status()["state"] == "reconnecting"
    m.running, m.last_error = False, "stream ended"
    assert m.get_status()["state"] == "error"
