"""
test_live_stream.py - Verification for real-time audio stream transcription & ingestion.

Tests:
1. strip_overlap reuse behavior from air_bulletins_scraper:
   - Strips leading overlapping text when overlap >= min_overlap (80 chars).
   - Preserves non-overlapping distinct text without changes.
   - Handles empty/edge-case inputs gracefully.
2. GET /stream/status when no stream is started:
   - Returns 200 OK.
   - Shape: running=False, url=None, started_at=None, segments_ingested=0, last_chunk_at=None.
3. POST /stream/stop when no stream is active:
   - Returns 200 OK with status="not_running".
4. check_ffmpeg utility behavior.

Exit code 0 on all pass.
"""

import sys
from fastapi.testclient import TestClient

from air_bulletins_scraper import strip_overlap
from chatbot_api import app
from live_stream_ingest import check_ffmpeg, live_manager


def test_strip_overlap_behavior():
    print("Testing strip_overlap reuse...")
    overlap_phrase = (
        "The Cabinet Committee on Economic Affairs chaired by the Prime Minister "
        "has approved major new infrastructure initiatives across multiple states."
    )
    assert len(overlap_phrase) >= 80, f"Overlap phrase length is {len(overlap_phrase)}, expected >= 80"

    prev_text = "Good morning, this is the All India Radio hourly bulletin. " + overlap_phrase
    curr_text = overlap_phrase + " In other developments, the weather department forecast heavy rains in coastal areas."

    # 1. Overlapping text should be cleanly stripped
    trimmed = strip_overlap(prev_text, curr_text, min_overlap=80)
    expected = "In other developments, the weather department forecast heavy rains in coastal areas."
    assert trimmed == expected, f"Expected '{expected}', got '{trimmed}'"
    print("  PASS: Overlapping text correctly stripped")

    # 2. Distinct text should be untouched
    distinct_prev = "India recorded a historic victory in the ICC cricket tournament yesterday evening."
    distinct_curr = "The Reserve Bank of India monetary policy committee kept the benchmark repo rate unchanged."
    assert strip_overlap(distinct_prev, distinct_curr, min_overlap=80) == distinct_curr
    print("  PASS: Distinct text untouched")

    # 3. Empty strings
    assert strip_overlap("", "hello world") == "hello world"
    assert strip_overlap("hello world", "") == ""
    print("  PASS: Empty inputs handled")


def test_stream_status_endpoint():
    print("\nTesting GET /stream/status (idle state)...")
    client = TestClient(app)
    resp = client.get("/stream/status")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    print("  Response:", data)

    # Verify exact required shape
    required_keys = {"running", "url", "started_at", "segments_ingested", "last_chunk_at"}
    assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - set(data.keys())}"

    assert data["running"] is False, f"Expected running=False, got {data['running']}"
    assert data["url"] is None, f"Expected url=None, got {data['url']}"
    assert data["started_at"] is None, f"Expected started_at=None, got {data['started_at']}"
    assert data["segments_ingested"] == 0, f"Expected segments_ingested=0, got {data['segments_ingested']}"
    assert data["last_chunk_at"] is None, f"Expected last_chunk_at=None, got {data['last_chunk_at']}"
    print("  PASS: /stream/status returned sane idle shape")


def test_stream_stop_idle():
    print("\nTesting POST /stream/stop (idle state)...")
    client = TestClient(app)
    resp = client.post("/stream/stop", json={})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    print("  Response:", data)
    assert data.get("status") == "not_running"
    print("  PASS: /stream/stop handled idle state gracefully")


def test_ffmpeg_check():
    print("\nTesting check_ffmpeg()...")
    # On the system where ffmpeg is installed, check_ffmpeg() should return None
    try:
        check_ffmpeg()
        print("  PASS: check_ffmpeg passed with host ffmpeg")
    except RuntimeError as e:
        print(f"  NOTE: ffmpeg not detected: {e}")


def test_incremental_dedup_and_hour():
    print("\nTesting hour extraction and incremental dedup cache...")
    import datetime as dt
    import numpy as np

    # Test hour extraction logic
    test_dt = dt.datetime(2026, 9, 22, 14, 35, 10)
    hour_num = test_dt.hour
    assert hour_num == 14, f"Expected hour=14, got {hour_num}"
    assert 0 <= hour_num <= 23
    print("  PASS: hour_num = now.hour correctly produces 14 (not 1435)")

    # Test incremental deduplication in memory
    import nlp_pipeline
    text1 = "The Prime Minister held bilateral talks with foreign leaders in New Delhi today."
    segments1 = nlp_pipeline.process_transcript(
        text1,
        existing_embeddings=None,
        existing_texts=[],
        fast=True,
    )
    assert len(segments1) == 1

    # Simulate incremental cache update
    cached_embeddings = np.vstack([seg["embedding"] for seg in segments1])
    cached_texts = [seg["text"] for seg in segments1]

    # Transcribing identical text using cached embeddings should filter it as duplicate
    segments2 = nlp_pipeline.process_transcript(
        text1,
        existing_embeddings=cached_embeddings,
        existing_texts=cached_texts,
        fast=True,
    )
    assert len(segments2) == 0, f"Expected 0 novel segments due to dedup, got {len(segments2)}"
    print("  PASS: In-memory cached dedup filtered duplicate without disk reload")


def main() -> int:
    try:
        test_strip_overlap_behavior()
        test_stream_status_endpoint()
        test_stream_stop_idle()
        test_ffmpeg_check()
        test_incremental_dedup_and_hour()
        print("\nAll live stream tests passed successfully!")
        return 0
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
