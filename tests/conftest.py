"""
Shared fixtures. Tests run offline and fast: the embedding model and genre
classifier are replaced by deterministic stand-ins (hashed bag-of-words
vectors, keyword genres) so retrieval behaviour is reproducible without
downloading models. ASR is replaced by a scripted fake in the live tests;
ffmpeg is used for real.
"""

import datetime as dt
import hashlib
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["VANI_LOG_FILE"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["VANI_DATE_ANCHOR"] = "today"
# The stand-in embeddings have a different score scale from MiniLM; the top-score floor
# is tested explicitly in test_articles.py.
os.environ["VANI_RAG_MIN_TOP_SCORE"] = "0"
os.environ["VANI_RAG_STRONG_MATCH"] = "0"
# Test questions must never reach the real query log (analytics); the query-log test enables it.
os.environ["VANI_QUERY_LOG"] = "0"
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import config  # noqa: E402
import nlp_pipeline  # noqa: E402
import rag_engine  # noqa: E402
import vector_store  # noqa: E402

_STOP = {"the", "a", "an", "of", "in", "on", "and", "to", "for", "is", "was", "at", "by", "with", "has", "have"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def fake_embed(sentences):
    out = np.zeros((len(sentences), config.EMBEDDING_DIM), dtype=np.float32)
    for i, s in enumerate(sentences):
        for w in re.findall(r"[a-z0-9]+", s.lower()):
            if w in _STOP:
                continue
            h = int(hashlib.md5(_stem(w).encode()).hexdigest(), 16)
            out[i, h % config.EMBEDDING_DIM] += 1.0 if (h >> 20) & 1 else -1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


def fake_classify(sentences, fast=False):
    res = []
    for s in sentences:
        g = rag_engine._extract_genre(s) or "Other"
        res.append((g, 0.9))
    return res


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated working dir + empty store + offline models."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vector_store, "FAISS_INDEX_PATH", str(tmp_path / "faiss_index.bin"))
    monkeypatch.setattr(vector_store, "SQLITE_DB_PATH", str(tmp_path / "news_store.db"))
    vector_store._index_cache.clear()
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(nlp_pipeline, "embed_sentences", fake_embed)
    monkeypatch.setattr(nlp_pipeline, "classify_genres_batch", fake_classify)
    monkeypatch.setattr(rag_engine, "_generation_mode", "FALLBACK (tests)")
    (tmp_path / "output").mkdir()
    yield tmp_path
    vector_store._index_cache.clear()


def today():
    return config.now_local().date()


def add(text, date, hour, source, genre=None, a=None, b=None, mode="archive", name=None):
    """Store one segment directly (bypasses MMR) for retrieval fixtures."""
    emb = fake_embed([text])[0]
    seg = {"text": text, "genre": genre or (rag_engine._extract_genre(text) or "Other"),
           "genre_confidence": 0.9, "embedding": emb, "audio_start": a, "audio_end": b}
    vector_store.add_segments([seg], date, hour, source, source_name=name or source, ingest_mode=mode)
    return seg["faiss_id"]


@pytest.fixture
def corpus(env):
    """A small multi-day corpus: today, yesterday and a fixed historical date."""
    t = today()
    y = (t - dt.timedelta(days=1)).isoformat()
    t = t.isoformat()
    rows = [
        ("India beat Australia by six wickets in the cricket match at Ahmedabad.", y, 1900, "text/y/Eve.txt", "Sports"),
        ("The hockey team won the tournament final in Bhubaneswar.", y, 1905, "text/y/Eve.txt", "Sports"),
        ("The Reserve Bank kept the repo rate unchanged to control inflation.", y, 900, "text/y/Morn.txt", "Economy"),
        ("GDP growth for the quarter was estimated at seven percent.", y, 905, "text/y/Morn.txt", "Economy"),
        ("Parliament passed the new data protection bill after a long debate.", y, 1300, "text/y/Noon.txt", "Politics"),
        ("The Prime Minister inaugurated a semiconductor fabrication plant in Gujarat.", t, 1030, "text/t/Live.txt", "Science & Technology"),
        ("The plant will manufacture chips for automobiles and phones.", t, 1030, "text/t/Live.txt", "Science & Technology"),
        ("ISRO launched a communication satellite from Sriharikota.", "2026-09-20", 800, "text/2026-09-20/A.txt", "Science & Technology"),
        ("Heavy rain is expected in coastal Karnataka over the weekend.", "2026-09-20", 810, "text/2026-09-20/A.txt", "Environment"),
    ]
    ids = {}
    for i, (txt, d, h, src, g) in enumerate(rows):
        ids[i] = add(txt, d, h, src, g, a=float(i), b=float(i) + 4.0, name=os.path.basename(src))
    return {"today": t, "yesterday": y, "ids": ids}
