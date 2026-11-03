"""
test_audio_endpoint.py - Verification for GET /audio/{faiss_id} endpoint.

Tests:
1. Missing faiss_id (99999) -> 404 JSON.
2. Existing faiss_id with missing audio on disk -> 404 JSON.
3. Existing faiss_id with audio on disk -> 200 audio/mpeg.
4. HTTP Range request on valid audio -> 206 Partial Content with Accept-Ranges.
5. Path traversal boundary check.

Exit code 0 on all pass.
"""

import os
import sqlite3
from fastapi.testclient import TestClient
from chatbot_api import app

client = TestClient(app)


def main() -> int:
    # 1. Nonexistent faiss_id
    r_missing = client.get("/audio/99999")
    print(f"Nonexistent ID 99999: status={r_missing.status_code}, body={r_missing.json()}")
    assert r_missing.status_code == 404
    assert "error" in r_missing.json()
    assert r_missing.json()["error"] == "Segment 99999 not found"

    # 2. Existing faiss_id (0) with audio missing on disk
    test_audio_path = os.path.join("output", "audio", "2024-05-01", "English-0800-0810-1.mp3")
    backup_path = test_audio_path + ".testbak"
    had_original = os.path.exists(test_audio_path)
    if had_original:
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(test_audio_path, backup_path)

    try:
        r_file_missing = client.get("/audio/0")
        print(f"Existing ID 0 without audio file: status={r_file_missing.status_code}, body={r_file_missing.json()}")
        assert r_file_missing.status_code == 404
        assert "error" in r_file_missing.json()

        # 3. Create a temporary MP3 file for segment 0
        os.makedirs(os.path.dirname(test_audio_path), exist_ok=True)
        # Write 1000 bytes dummy MP3 content
        dummy_mp3 = bytes([0xFF, 0xFB, 0x90, 0x64]) + bytes(996)
        with open(test_audio_path, "wb") as f:
            f.write(dummy_mp3)

        # 3a. Full file request
        r_ok = client.get("/audio/0")
        print(f"Existing ID 0 with audio: status={r_ok.status_code}, content-type={r_ok.headers.get('content-type')}")
        assert r_ok.status_code == 200
        assert "audio/mpeg" in r_ok.headers.get("content-type", "")
        assert len(r_ok.content) == 1000

        # 4. HTTP Range request (seeking/scrubbing support)
        r_range = client.get("/audio/0", headers={"Range": "bytes=0-99"})
        print(f"Range request: status={r_range.status_code}, content-range={r_range.headers.get('content-range')}")
        assert r_range.status_code == 206
        assert r_range.headers.get("content-range") == "bytes 0-99/1000"
        assert len(r_range.content) == 100
        assert r_range.headers.get("accept-ranges") == "bytes"
    finally:
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
        if had_original and os.path.exists(backup_path):
            os.rename(backup_path, test_audio_path)

    # 5. Malicious crafted ID / boundary check
    # Try invalid type (e.g. /audio/evil)
    r_bad_id = client.get("/audio/evil")
    print(f"Invalid ID format: status={r_bad_id.status_code}")
    assert r_bad_id.status_code in (404, 422)

    print("\nAll GET /audio/{faiss_id} tests PASS (exit code 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
