# VaniStream — Evaluation Demo Guide

This guide contains everything needed to spin up the system from a clean checkout and run an end-to-end evaluation demo.

---

## 1. Clean Checkout Startup

Run these commands in order from the repository root:

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Ingest transcripts, run MMR deduplication, and build vector store
python ingest_pipeline.py --reindex --fast

# 4. Start the FastAPI server & Chat Web UI
uvicorn chatbot_api:app --reload --port 8000
```

Open your browser at: **[http://localhost:8000](http://localhost:8000)**

---

## 2. Five Verified Demo Queries

These 5 queries are verified against the indexed AIR news corpus:

### Query 1: Relative Date Search
- **Query:** `What happened yesterday?`
- **What it demonstrates:** Dynamic date anchoring (`_get_corpus_anchor()`) resolves relative date queries against the newest date in the corpus (`2024-05-02`), extracting `2024-05-01` bulletins instead of failing on the real-world calendar date.
- **Expected result:** 5 news segments retrieved from `2024-05-01`, including the India-Australia ICC T20 match victory, Gujarat semiconductor plant inauguration, RBI repo rate decision, and ISRO satellite launch.

### Query 2: Science & Technology Search
- **Query:** `Tell us about the ISRO satellite launch`
- **What it demonstrates:** Word-boundary keyword matching correctly classifies query genre as `Science & Technology` (without false-positive matching on "us").
- **Expected result:** Retrieves the ISRO earth observation satellite launch segment.

### Query 3: Defence & Security Search
- **Query:** `What are the latest defence and military updates?`
- **What it demonstrates:** Genre pre-filtering restricts the search space to `Defence & Security` in SQLite before FAISS similarity calculation.
- **Expected result:** Retrieves: *"Defence forces completed joint military exercises demonstrating multi-domain operational readiness."*

### Query 4: Politics Search
- **Query:** `What happened in politics?`
- **What it demonstrates:** Two-tier RAG filtering on `Politics`.
- **Expected result:** Retrieves: *"The Supreme Court reserved its judgment on important constitutional petitions regarding electoral reforms."*

### Query 5: Environmental & Weather Search
- **Query:** `What is the weather report?`
- **What it demonstrates:** Genre extraction mapping "weather" to `Environment`.
- **Expected result:** Retrieves: *"Heavy rainfall lashed coastal regions bringing relief from intense summer heatwaves."*

---

## 3. Demo Safety & Out-of-Range Handling

To demonstrate that the system **never** silently performs global search when a filter finds no results:

- **Query:** `sports news on 2099-01-01`
- **Expected result:** Explicit message stating:
  > *"No news segments found for 2099-01-01 to 2099-01-01 in genre 'Sports'. My corpus covers 2024-05-01 to 2024-05-02."*

---

## 4. Key Demo Features

### A. Deduplication Proof
- **Endpoint:** `GET http://localhost:8000/dedup`
- **UI Element:** View the green summary badge in the UI header:
  > *"Redundancy removed: 20 → 12 sentences, 40.0%"*
- Click **"View deduplication examples"** in the header to expand the top 3 rejected duplicate pairs with their cosine similarity scores.

### B. Story Evolution Timeline
- **Endpoint:** `GET http://localhost:8000/timeline?q=politics`
- **UI Element:** Enter any query into the chat input and click the **"Timeline"** button.
- Displays chronological story evolution grouped by broadcast date and ordered by broadcast hour (`08:00 hrs`, `09:00 hrs`, `10:00 hrs`).

### C. Generation Mode & Health Check
- **Endpoint:** `GET http://localhost:8000/health`
- Returns:
  ```json
  {"status": "ok", "generation_mode": "FALLBACK (no API key)"}
  ```
  *(Or `"LIVE"` if `GEMINI_API_KEY` is exported).*

### D. Click-to-Play Audio Citations
- **Audio File on Disk:** `output/audio/2024-05-01/English-0800-0810-1.mp3`
- **What it demonstrates:** Proves the end-to-end provenance chain from raw radio broadcast to transcript segment and RAG citation.
- **Testing in Chat UI:**
  1. Submit query: `Tell me about the semiconductor facility in Gujarat`
  2. The response renders source chips. Citations backed by an on-disk audio file display a clickable `▶ Audio` button.
  3. Clicking `▶ Audio` activates the shared bottom audio player bar (`#audioPlayerBar`) and begins playback with track metadata:
     > *"AIR Bulletin (2024-05-01 800) (plays full bulletin — sentence-level alignment is future work)"*
  4. The button transitions to `⏸ Stop`. Clicking it or the close button (`✕`) halts playback and hides/resets the player bar.
- **Testing in Timeline UI:**
  1. Enter `semiconductor` and click **"Timeline"**.
  2. The chronological card for 08:00 hrs displays the `▶ Audio` button alongside relevance score and genre badge.
- **Streaming & HTTP Range Endpoint:**
  - **Endpoint:** `GET http://localhost:8000/audio/0`
  - Returns `200 OK` (`Content-Type: audio/mpeg`) or `206 Partial Content` with `Accept-Ranges: bytes` when scrubbing in the browser.
- **Demo Safety & Graceful Degradation:**
  - If `output/audio` is deleted or missing, the system does not crash or throw errors: `/chat` and `/timeline` seamlessly return `has_audio: false`, and audio buttons are cleanly hidden in the UI.

### E. Live Streaming & Real-Time Ingestion
- **What it demonstrates:** Continuous transcription of live radio streams (Icecast MP3 / HLS) via rolling PCM ffmpeg decoding, faster-whisper VAD, MMR deduplication, and immediate indexing into FAISS + SQLite.
- **Standalone CLI Command:**
  ```bash
  python live_stream_ingest.py --url "http://air-stream-url.example/live.mp3" --chunk-seconds 30 --overlap-seconds 5 --fast
  ```
- **REST Endpoints & curl Examples:**
  1. **Start Stream:**
     ```bash
     curl -X POST http://localhost:8000/stream/start \
       -H "Content-Type: application/json" \
       -d '{"url": "http://air-stream-url.example/live.mp3", "language": "english", "chunk_seconds": 30}'
     ```
     *Response:*
     ```json
     {"stream_id": "a1b2c3d4", "status": "started", "url": "http://air-stream-url.example/live.mp3"}
     ```

  2. **Check Stream Status:**
     ```bash
     curl http://localhost:8000/stream/status
     ```
     *Response:*
     ```json
     {
       "running": true,
       "url": "http://air-stream-url.example/live.mp3",
       "started_at": "2026-09-22T14:30:00Z",
       "segments_ingested": 4,
       "last_chunk_at": "2026-09-22T14:31:00Z"
     }
     ```

  3. **Stop Stream (Clean Buffer Flush):**
     ```bash
     curl -X POST http://localhost:8000/stream/stop \
       -H "Content-Type: application/json" \
       -d '{}'
     ```
     *Response:*
     ```json
     {"status": "stopped", "stream_id": "a1b2c3d4"}
     ```

- **UI LIVE Pill:**
  - Located in the header next to the status dot.
  - Automatically polls `/stream/status` every ~5s.
  - Glows and pulses green when an active stream is running, updating its tooltip with stream URL and ingested segment count.

---

## 5. Architecture Defense Q&A

**Question:** *"Why was zero-shot the initial choice, and when should you switch to the fine-tuned model?"*

**Answer — Why zero-shot first:**
> Zero-shot NLI (BART-large-MNLI / MiniLM NLI) generalises directly from entailment pretraining without requiring any labeled AIR-domain data. At project inception no labeled corpus existed, so zero-shot was the only principled option. Committing to a fine-tuned model without labeled data would have introduced silent label noise with no way to measure it — a risk explicitly identified in `DEMO.md` and `build_genre_training_data.py`. This matches the project-wide graceful degradation philosophy: use the best available capability, not the most complex one.

**Answer — When to switch to fine-tuned:**
> The fine-tuned DistilBERT classifier (`models/air-genre-classifier/`) should be preferred when **both** conditions are met:
> 1. **Corpus size**: at least ~100 human-reviewed training examples per class (1,000 rows total across 10 genres), obtainable by running `python build_genre_training_data.py --augment` and filling the `true_genre` column.
> 2. **Measured accuracy gain**: `python eval_genre.py --compare` shows a delta of **+5 pp or more** over zero-shot on a held-out `eval_labels.csv` set. Below that threshold the improvement is within noise and does not justify the inference overhead.
>
> The fine-tuned path is a **selectable backend** (`--classifier finetuned`), not a replacement. If the checkpoint is absent, the system automatically falls back to zero-shot fast — the same graceful degradation pattern used for the Gemini LIVE/FALLBACK probe in `rag_engine.py`.

**Question:** *"What domain terms does zero-shot miss?"*

**Answer:**
> Zero-shot NLI struggles with opaque domain tokens like **"Rajya Sabha"** (misclassified as Society rather than Politics), **"ISRO"** (sometimes Economy or Other), and **"repo rate"** (sometimes International). The fine-tuned model learns these associations directly from AIR corpus examples. `build_genre_training_data.py --augment` pre-seeds the training set with template sentences combining these keywords to bootstrap underrepresented classes.
