"""Live ingestion tests (realtime prompt tests 1-8, 10-13).

A real ffmpeg process reads a generated WAV "stream". Each second of that
audio carries a distinct constant sample value, which lets the scripted fake
ASR know exactly which part of the broadcast each window covers, and return
the words a real model would hear there (with timestamps). Everything after
ASR - windowing, overlap stitching, NLP, MMR, SQLite, FAISS, audio files - is
the production code path.
"""

import os
import shutil
import time
import wave

import numpy as np
import pytest

import config
import live_stream_ingest as live
import rag_engine
import transcriber
import vector_store
from transcriber import Segment, TranscriptionResult, Word

pytestmark = pytest.mark.skipif(live.audio_backend() is None, reason="an audio decoder (ffmpeg or PyAV) is required")

SCRIPT = [
    "Good evening, this is All India Radio with the news.",
    "The Prime Minister inaugurated a new semiconductor facility in Gujarat today.",
    "The plant is expected to create twenty thousand jobs in the state.",
    "In cricket, India defeated England by five wickets in Chennai.",
    "The Prime Minister inaugurated a new semiconductor facility in Gujarat today.",  # repeated headline
    "The weather office has forecast heavy rain along the Konkan coast.",
    "That is the end of this bulletin.",
]


def _timed_words(script, word_s=0.5, gap=1.0, t0=0.5):
    words, t = [], t0
    for sent in script:
        for w in sent.split():
            words.append((t, t + word_s * 0.9, " " + w))
            t += word_s
        t += gap
    return words, t


WORDS, DURATION = _timed_words(SCRIPT)


def _make_wav(path, seconds):
    n = int(seconds) + 1
    samples = np.concatenate([np.full(16000, (i + 1) * 100, dtype=np.int16) for i in range(n)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())


class ScriptedASR:
    label = "scripted:test"

    def __init__(self, words=None):
        self.calls = 0
        self.words = words or WORDS

    def _load(self):
        return self

    def transcribe(self, audio, offset=0.0, word_timestamps=True):
        self.calls += 1
        ws = int(round(audio[0] * 32768 / 100)) - 1  # second index encoded in the samples
        length = audio.size / 16000
        heard = [Word(a - ws, b - ws, t) for a, b, t in self.words if ws <= a and b <= ws + length]
        segs = [Segment(heard[0].start, heard[-1].end, "".join(w.text for w in heard), heard)] if heard else []
        return TranscriptionResult(segs, "en", length, self.label)


@pytest.fixture(params=[0.5, 3.2, 8.7], ids=["boundary-between-sentences", "split-mid-sentence", "split-late"])
def live_env(env, monkeypatch, request):
    words, duration = _timed_words(SCRIPT, t0=request.param)
    wav = env / "stream.wav"
    _make_wav(wav, duration)
    asr = ScriptedASR(words)
    monkeypatch.setattr(transcriber, "get_transcriber", lambda **kw: asr)
    monkeypatch.setattr(config, "LIVE_RETRY_BASE_SECONDS", 0.05)
    monkeypatch.setattr(config, "LIVE_DEDUP_WINDOW_HOURS", 24)
    return {"url": str(wav), "asr": asr}


def test_stream_url_required(env, monkeypatch):
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "")
    with pytest.raises(ValueError, match="LIVE_STREAM_URL"):
        live.LiveStreamManager().start()
    monkeypatch.setattr(config, "LIVE_STREAM_URL", "http://example.invalid/live.mp3")
    assert config.require_stream_url() == "http://example.invalid/live.mp3"


def test_select_window_words_defers_overlap_and_drops_committed():
    words = [Word(1, 2, " a"), Word(24, 24.5, " b"), Word(26, 27, " c")]
    commit, defer = live.select_window_words(words, window_start=25, window_len=30, overlap=5,
                                             committed_until=26.5, final=False)
    # a: abs 26 < 26.5 -> already committed; b: before cutoff 25 -> commit; c: in overlap tail -> defer
    assert [w.text for w in commit] == [" b"] and [w.text for w in defer] == [" c"]


def test_live_pipeline_end_to_end(live_env):
    mgr = live.LiveStreamManager()
    mgr.start(url=live_env["url"], chunk_seconds=30, overlap_seconds=5, max_retries=1)
    mgr.wait()
    st = mgr.get_status()

    # Audio was received, windows were transcribed, and the stream reconnected after EOF (test 13)
    assert live_env["asr"].calls >= 3 and st["chunks_processed"] >= 3 and st["chunks_failed"] == 0
    assert st["reconnects"] >= 1 and st["last_audio_received"] and not st["running"]

    db = vector_store._get_db()
    rows = [dict(r) for r in db.execute("SELECT * FROM segments ORDER BY faiss_id")]
    db.close()
    texts = [r["text"] for r in rows]
    unique_script = list(dict.fromkeys(SCRIPT))
    # Every sentence stored exactly once, in order, nothing lost at window boundaries,
    # the repeated headline and the replay after reconnect rejected as duplicates (test 10).
    assert texts == unique_script
    assert st["duplicates_rejected"] >= 1 + len(unique_script)

    for r in rows:  # tests 5, 7, 11
        assert r["ingest_mode"] == "live" and r["ts_start"] and r["ts_end"]
        assert r["audio_start"] is not None and 0 <= r["audio_start"] <= r["audio_end"] <= 30.5
        assert 0 <= r["hour"] <= 2359
        path, exists = vector_store.resolve_audio_path(r["source_file"])
        assert exists and path.endswith(".mp3") and "/Live-" in path

    # Test 8 + 9: searchable in FAISS and through RAG immediately
    res = rag_engine.answer_query("semiconductor facility Gujarat")
    assert res["sources"][0]["ingest_mode"] == "live" and "semiconductor" in res["sources"][0]["text"]
    res = rag_engine.answer_query("What happened today?")
    assert res["sources"] and all(s["ingest_mode"] == "live" for s in res["sources"])


def test_bad_window_does_not_kill_stream(live_env, monkeypatch):
    asr = live_env["asr"]
    orig = asr.transcribe

    def flaky(audio, **kw):
        if asr.calls == 0:
            asr.calls += 1
            raise RuntimeError("simulated ASR crash")
        return orig(audio, **kw)

    monkeypatch.setattr(asr, "transcribe", flaky)
    mgr = live.LiveStreamManager()
    mgr.start(url=live_env["url"], chunk_seconds=30, overlap_seconds=5, max_retries=1)
    mgr.wait()
    st = mgr.get_status()
    assert st["chunks_failed"] == 1 and st["segments_ingested"] == 0  # status resets once stopped
    assert vector_store.get_stats()["total_segments"] >= 4  # later windows still stored


def test_restart_resumes_without_duplicates(live_env):
    for _ in range(2):  # two separate runs over the same audio = crash + restart
        mgr = live.LiveStreamManager()
        mgr.start(url=live_env["url"], chunk_seconds=30, overlap_seconds=5, max_retries=1)
        mgr.wait()
        vector_store._index_cache.clear()
    assert vector_store.get_stats()["total_segments"] == len(set(SCRIPT))
