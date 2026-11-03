"""Archive ingestion, persistence, schema migration, restart and audio mapping (tests 6, 7, 8, 10, 11, 12)."""

import csv
import json
import os
import sqlite3

import numpy as np

import ingest_pipeline
import vector_store
from conftest import fake_embed
from transcriber import Word, align_sentences


def _write_bulletin(root, date, name, text, words=None):
    tdir = root / "output" / "text" / date
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{name}.txt").write_text(text, encoding="utf-8")
    if words:
        segs = [{"start": words[0][0], "end": words[-1][1], "text": text, "words": [list(w) for w in words]}]
        (tdir / f"{name}.segments.json").write_text(json.dumps({"segments": segs}), encoding="utf-8")
    adir = root / "output" / "audio" / date
    adir.mkdir(parents=True, exist_ok=True)
    (adir / f"{name}.mp3").write_bytes(b"ID3fake-mp3")
    return f"text/{date}/{name}.txt", f"audio/{date}/{name}.mp3"


def _csv(root, rows):
    path = root / "output" / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "date", "language", "category", "audio_url", "audio_file",
                                          "transcript_file", "sha256", "downloaded_at"])
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_align_sentences_with_words():
    words = [Word(0.0, 0.4, " The"), Word(0.4, 0.9, " budget"), Word(0.9, 1.5, " passed."),
             Word(2.0, 2.4, " Rain"), Word(2.4, 3.0, " expected.")]
    assert align_sentences(["The budget passed.", "Rain expected."], words) == [(0.0, 1.5), (2.0, 3.0)]


def test_archive_ingest_end_to_end(env):
    text = ("The Union Cabinet approved a new railway line in Assam. "
            "The Union Cabinet approved a new railway line in Assam. "
            "Heavy rain lashed Mumbai, disrupting local trains.")
    words, t = [], 0.0
    for w in text.split():
        words.append((t, t + 0.4, " " + w))
        t += 0.5
    tf, af = _write_bulletin(env, "2026-10-30", "English-0800-0810-1", text, words)
    csv_path = _csv(env, [{"title": "Morning News 30 Oct", "date": "2026-10-30", "language": "english",
                           "category": "hourly-news", "audio_url": "https://example.org/a.mp3", "audio_file": af,
                           "transcript_file": tf, "sha256": "x", "downloaded_at": "2026-10-30T08:00:00"}])
    summary = ingest_pipeline.run(metadata_csv=csv_path, fast=True, articles=False)
    assert summary == {"files_processed": 1, "segments_added": 2, "duplicates_filtered": 1,
                       "articles_added": 0, "article_chunks": 0}

    db = vector_store._get_db()
    rows = db.execute("SELECT * FROM segments ORDER BY faiss_id").fetchall()
    db.close()
    assert len(rows) == 2
    assert rows[0]["source_name"] == "Morning News 30 Oct" and rows[0]["hour"] == 800
    assert rows[0]["audio_start"] == 0.0 and rows[1]["audio_start"] > rows[0]["audio_end"]
    # SQLite row <-> FAISS vector consistency, and the vector is searchable
    hit = vector_store.semantic_search(fake_embed([rows[1]["text"]])[0], None, 1)[0]
    assert hit["faiss_id"] == rows[1]["faiss_id"]
    # Audio mapping: segment -> metadata audio_file -> existing file
    path, exists = vector_store.resolve_audio_path(rows[0]["source_file"])
    assert exists and path.endswith("English-0800-0810-1.mp3")
    # Re-running does not re-ingest
    assert ingest_pipeline.run(metadata_csv=csv_path, fast=True, articles=False)["files_processed"] == 0


def test_restart_keeps_records_and_continues_ids(env):
    seg = lambda t: {"text": t, "genre": "Other", "genre_confidence": 1.0, "embedding": fake_embed([t])[0]}  # noqa: E731
    vector_store.add_segments([seg("first story"), seg("second story")], "2026-11-01", 900, "text/a.txt")
    vector_store._index_cache.clear()  # simulate a fresh process
    assert vector_store.get_stats()["total_segments"] == 2
    vector_store.add_segments([seg("third story")], "2026-11-02", 900, "text/b.txt")
    stats = vector_store.get_stats()
    assert stats["total_segments"] == 3 and stats["faiss_vectors"] == 3
    assert vector_store.check_consistency()["orphans_removed"] == 0


def test_migration_preserves_old_rows(env):
    db_path = vector_store.SQLITE_DB_PATH
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE segments (id INTEGER PRIMARY KEY AUTOINCREMENT, faiss_id INTEGER UNIQUE, text TEXT NOT NULL,"
                " genre TEXT, genre_confidence REAL, date TEXT, hour INTEGER, source_file TEXT, indexed_at TEXT)")
    con.execute("INSERT INTO segments (faiss_id, text, genre, date, hour, source_file) VALUES "
                "(0, 'old archive line', 'Other', '2024-05-01', 800, 'text/2024-05-01/English-0800.txt'),"
                "(1, 'old live line', 'Other', '2026-09-22', 15, 'text\\2026-09-22\\Live-Integration-1526-0001.txt')")
    con.commit()
    con.close()
    rows = {r["faiss_id"]: dict(r) for r in vector_store._get_db().execute("SELECT * FROM segments")}
    assert rows[0]["ingest_mode"] == "archive" and rows[0]["hour"] == 800
    assert rows[1]["ingest_mode"] == "live" and rows[1]["hour"] == 1500
    assert vector_store.keyword_search("archive line")[0]["faiss_id"] == 0


def test_orphan_rows_removed_after_crash(env):
    seg = {"text": "only one", "genre": "Other", "genre_confidence": 1.0, "embedding": fake_embed(["only one"])[0]}
    vector_store.add_segments([seg], "2026-11-01", 900, "text/a.txt")
    db = vector_store._get_db()
    db.execute("INSERT INTO segments (faiss_id, text, date, hour, source_file) VALUES (5, 'ghost', '2026-11-01', 900, 'x')")
    db.commit()
    db.close()
    assert vector_store.check_consistency()["orphans_removed"] == 1


def test_mmr_rejects_repeat_but_keeps_new_development():
    import nlp_pipeline
    base = "Prime Minister inaugurated the semiconductor plant in Gujarat."
    sents = [base, base, "Prime Minister announced a second semiconductor plant in Assam with new funding."]
    embs = np.vstack([fake_embed([s])[0] for s in sents])
    novel, rej = nlp_pipeline.mmr_novelty_filter_detailed(sents, embs, threshold=0.85)
    assert novel == [0, 2] and len(rej) == 1 and rej[0]["cosine_score"] > 0.99
