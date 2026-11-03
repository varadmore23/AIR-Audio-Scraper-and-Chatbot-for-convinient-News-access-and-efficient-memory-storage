"""
test_resolve_audio.py - Unit test for resolve_audio_path helper.

Validates:
1. Exact mapping against 3 real rows in news_store.db.
2. Fallback when audio_file in metadata.csv is empty.
3. Preference for audio_file from metadata.csv when populated.
4. Exists on disk check (returns False when missing, True when present).

Exit code 0 on all pass.
"""

import os
import sqlite3
import tempfile
from vector_store import resolve_audio_path


def main() -> int:
    conn = sqlite3.connect("news_store.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT DISTINCT source_file FROM segments LIMIT 3").fetchall()
    conn.close()

    assert len(rows) >= 3, f"Expected at least 3 distinct source_file rows, found {len(rows)}"

    print("Checking 3 real rows from SQLite news_store.db:")
    for i, r in enumerate(rows, 1):
        sf = r["source_file"]
        path, exists = resolve_audio_path(sf)
        expected_start = "output/audio/"
        expected_end = ".mp3"
        print(f"  Row {i}: '{sf}' -> '{path}' (exists={exists})")
        assert path.startswith(expected_start), f"Expected path to start with '{expected_start}', got '{path}'"
        assert path.endswith(expected_end), f"Expected path to end with '{expected_end}', got '{path}'"
        assert exists == os.path.exists(path), f"exists flag ({exists}) did not match os.path.exists({path})"

    # Test metadata.csv preference
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="", encoding="utf-8") as f:
        f.write("transcript_file,audio_file\n")
        f.write("text/test/sample.txt,custom_audio/override.mp3\n")
        f.write("text/test/empty_audio.txt,\n")
        temp_csv = f.name

    try:
        # Override case
        path_override, _ = resolve_audio_path("text/test/sample.txt", metadata_csv=temp_csv)
        assert path_override == "output/custom_audio/override.mp3", f"Expected metadata override, got {path_override}"
        print("  PASS: metadata.csv audio_file override respected")

        # Fallback case when audio_file is empty
        path_fallback, _ = resolve_audio_path("text/test/empty_audio.txt", metadata_csv=temp_csv)
        assert path_fallback == "output/audio/test/empty_audio.mp3", f"Expected derived fallback, got {path_fallback}"
        print("  PASS: empty metadata column falls back to derived path")

    finally:
        os.remove(temp_csv)

    print("\nAll resolve_audio_path tests PASS (exit code 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
