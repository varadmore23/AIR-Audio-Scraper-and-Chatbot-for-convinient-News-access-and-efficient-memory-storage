"""
integration_test_live.py - End-to-end live ingest integration test.

Uses a local test_news.mp3 as a stand-in for a live stream chunk, drives the
full ingest chain, and reports:
  1. FAISS + SQLite segment count BEFORE
  2. Actual transcript text produced
  3. FAISS + SQLite segment count AFTER
  4. No ffmpeg processes left running after stop
  5. Audio snapshot and transcript file presence on disk
"""

import datetime as dt
import os
import subprocess
import sys
import sqlite3
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_faiss_count():
    import faiss
    index_path = os.path.join("faiss_index.bin")
    if not os.path.isfile(index_path):
        return 0
    index = faiss.read_index(index_path)
    return int(index.ntotal)

def get_sqlite_count():
    db_path = os.path.join("news_store.db")
    if not os.path.isfile(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    conn.close()
    return count

def count_ffmpeg_procs():
    result = subprocess.run(
        ["tasklist", "/FO", "CSV", "/FI", "IMAGENAME eq ffmpeg.exe"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.splitlines() if "ffmpeg" in l.lower()]
    return len(lines)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# --------------------------------------------------------------------------
# Step 0: Verify test audio exists
# --------------------------------------------------------------------------

section("STEP 0: Setup")

test_mp3 = os.path.join("test_news.mp3")
if not os.path.isfile(test_mp3):
    print(f"ERROR: {test_mp3} not found - expected TTS audio file", file=sys.stderr)
    sys.exit(1)
print(f"[OK] Test audio: {test_mp3} ({os.path.getsize(test_mp3):,} bytes)")

# --------------------------------------------------------------------------
# Step 1: Baseline counts
# --------------------------------------------------------------------------

section("STEP 1: Baseline FAISS + SQLite segment counts")
before_faiss = get_faiss_count()
before_sqlite = get_sqlite_count()
print(f"  FAISS  segments BEFORE: {before_faiss}")
print(f"  SQLite segments BEFORE: {before_sqlite}")

# --------------------------------------------------------------------------
# Step 2: Read raw PCM via ffmpeg (same path live_stream_ingest uses)
# --------------------------------------------------------------------------

section("STEP 2: Decode test_news.mp3 to 16kHz mono PCM (ffmpeg)")
cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-i", test_mp3,
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    "-f", "s16le", "-"
]
proc = subprocess.run(cmd, capture_output=True)
pcm_bytes = proc.stdout
print(f"  [OK] PCM decoded: {len(pcm_bytes):,} bytes  ({len(pcm_bytes)/32000:.2f}s at 16kHz mono)")
if not pcm_bytes:
    print("  ERROR: ffmpeg produced no PCM output", file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------------------------------
# Step 3: Save as MP3 snapshot (same path as live_stream_ingest.save_pcm_as_mp3)
# --------------------------------------------------------------------------

section("STEP 3: Save rolling window as MP3 snapshot")

from live_stream_ingest import save_pcm_as_mp3

now = dt.datetime.utcnow()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H%M")
slug = f"Live-Integration-{time_str}-0001"
out_dir = "output"
audio_dir = os.path.join(out_dir, "audio", date_str)
audio_path = os.path.join(audio_dir, f"{slug}.mp3")
save_pcm_as_mp3(pcm_bytes, audio_path, sample_rate=16000)
print(f"  [OK] Audio snapshot: {audio_path}")
print(f"       Size: {os.path.getsize(audio_path):,} bytes" if os.path.isfile(audio_path) else "  ERROR: snapshot not saved")

# --------------------------------------------------------------------------
# Step 4: Transcribe with faster-whisper (VAD=True)
# --------------------------------------------------------------------------

section("STEP 4: Transcribe with faster-whisper (vad_filter=True)")

from faster_whisper import WhisperModel
print("  Loading faster-whisper 'base' model on cpu...")
model = WhisperModel("base", device="cpu", compute_type="int8")
segments_gen, info = model.transcribe(audio_path, vad_filter=True)
transcript_raw = " ".join(seg.text.strip() for seg in segments_gen).strip()

print(f"\n  *** ACTUAL TRANSCRIPT TEXT ***")
print(f"  {transcript_raw!r}")
print(f"\n  Language detected : {info.language} (prob={info.language_probability:.2f})")
print(f"  Duration          : {info.duration:.2f}s")

if not transcript_raw:
    print("  WARNING: Empty transcript — VAD may have filtered all audio.")

# --------------------------------------------------------------------------
# Step 5: strip_overlap (simulate chunk stitching)
# --------------------------------------------------------------------------

section("STEP 5: strip_overlap on chunk boundary")

from air_bulletins_scraper import strip_overlap

prev_text = ""
transcript_clean = strip_overlap(prev_text, transcript_raw, min_overlap=80)
print(f"  prev_text (empty for first chunk): {prev_text!r}")
print(f"  After strip_overlap: {transcript_clean!r}")

# --------------------------------------------------------------------------
# Step 6: Write transcript to disk
# --------------------------------------------------------------------------

section("STEP 6: Persist transcript")

text_dir = os.path.join(out_dir, "text", date_str)
os.makedirs(text_dir, exist_ok=True)
transcript_path = os.path.join(text_dir, f"{slug}.txt")
with open(transcript_path, "w", encoding="utf-8") as f:
    f.write(transcript_clean)
rel_transcript = os.path.relpath(transcript_path, out_dir)
print(f"  [OK] Transcript file: {transcript_path}")
print(f"       Relative path  : {rel_transcript}")

# Append to metadata.csv
from air_bulletins_scraper import write_csv_rows, sha256_file

file_hash = sha256_file(audio_path) if os.path.isfile(audio_path) else ""
row = {
    "title": f"Integration Test Live ({date_str} {time_str})",
    "date": date_str,
    "language": "english",
    "category": "hourly-news",
    "audio_url": "test://live-integration",
    "audio_file": os.path.relpath(audio_path, out_dir),
    "transcript_file": rel_transcript,
    "sha256": file_hash,
    "downloaded_at": now.isoformat(timespec="seconds") + "Z",
}
write_csv_rows(os.path.join(out_dir, "metadata.csv"), [row])
print(f"  [OK] Row appended to output/metadata.csv")

# --------------------------------------------------------------------------
# Step 7: NLP pipeline + vector store ingestion
# --------------------------------------------------------------------------

section("STEP 7: nlp_pipeline.process_transcript + vector_store.add_segments")

import nlp_pipeline
import vector_store

existing_embeddings = vector_store.get_all_embeddings()
existing_texts = vector_store.get_all_texts()
print(f"  Existing embeddings in index: {existing_embeddings.shape if existing_embeddings is not None else 'None'}")

if transcript_clean:
    segments = nlp_pipeline.process_transcript(
        transcript_clean,
        existing_embeddings=existing_embeddings,
        existing_texts=existing_texts,
        fast=True,
        mmr_threshold=0.85,
        detailed=False,
    )
    print(f"\n  Novel segments produced : {len(segments)}")
    for i, seg in enumerate(segments, 1):
        print(f"  [{i}] genre={seg['genre']!r:<28} text={seg['text'][:80]!r}")

    if segments:
        hour_num = now.hour
        added = vector_store.add_segments(segments, date_str, hour_num, rel_transcript)
        print(f"\n  [OK] Segments added to FAISS + SQLite: {added}")
    else:
        print("  [NOTE] All sentences were MMR-deduplicated — no new segments added.")
else:
    print("  [SKIP] Empty transcript — skipping vector store ingestion")
    segments = []

# --------------------------------------------------------------------------
# Step 8: After counts
# --------------------------------------------------------------------------

section("STEP 8: FAISS + SQLite segment counts AFTER")
after_faiss = get_faiss_count()
after_sqlite = get_sqlite_count()
print(f"  FAISS  segments AFTER  : {after_faiss}  (was {before_faiss}, +{after_faiss - before_faiss})")
print(f"  SQLite segments AFTER  : {after_sqlite}  (was {before_sqlite}, +{after_sqlite - before_sqlite})")

# --------------------------------------------------------------------------
# Step 9: Verify ffmpeg cleanup (simulate stop)
# --------------------------------------------------------------------------

section("STEP 9: ffmpeg process cleanup after stream stop")

from live_stream_ingest import LiveStreamManager
manager = LiveStreamManager()

ffmpeg_before_stop = count_ffmpeg_procs()
print(f"  ffmpeg processes BEFORE stop: {ffmpeg_before_stop}")
result = manager.stop()
time.sleep(0.5)
ffmpeg_after_stop = count_ffmpeg_procs()
print(f"  stop() result              : {result}")
print(f"  ffmpeg processes AFTER stop: {ffmpeg_after_stop}")

if ffmpeg_after_stop == 0:
    print("  [OK] No ffmpeg processes left running")
else:
    print(f"  [WARN] {ffmpeg_after_stop} ffmpeg process(es) still running")

# --------------------------------------------------------------------------
# Step 10: Summary
# --------------------------------------------------------------------------

section("FINAL SUMMARY")
print(f"  Test audio             : test_news.mp3 ({os.path.getsize(test_mp3):,} bytes)")
print(f"  Audio snapshot on disk : {audio_path}")
print(f"  Transcript on disk     : {transcript_path}")
print(f"  Transcript text        : {transcript_clean!r}")
print(f"  Novel segments added   : {len(segments)}")
print(f"  FAISS  : {before_faiss} -> {after_faiss}  (+{after_faiss - before_faiss})")
print(f"  SQLite : {before_sqlite} -> {after_sqlite}  (+{after_sqlite - before_sqlite})")
print(f"  ffmpeg after stop      : {ffmpeg_after_stop}")
print()

all_ok = (
    os.path.isfile(audio_path)
    and os.path.isfile(transcript_path)
    and ffmpeg_after_stop == 0
)
print("INTEGRATION TEST PASS" if all_ok else "INTEGRATION TEST PARTIAL (see warnings)")
sys.exit(0 if all_ok else 1)
