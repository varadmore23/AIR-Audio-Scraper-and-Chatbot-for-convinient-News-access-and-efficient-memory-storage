# VaniStream — AIR News Intelligence System

VaniStream turns All India Radio broadcasts into a searchable, conversational knowledge base. It ingests archived bulletins and a **live radio stream**, transcribes them, removes repeated stories, classifies them by genre, and answers questions with **citations that point to the exact broadcast, time and audio segment**.

```
ARCHIVE MODE                       LIVE MODE
air_bulletins_scraper.py           LIVE_STREAM_URL -> ffmpeg (16 kHz PCM)
  MP3 bulletins                      30 s windows, 5 s overlap
        \                           /
         transcriber.py  (faster-whisper, VAD, word timestamps)
                         |
         nlp_pipeline.py (sentences -> MiniLM embeddings -> MMR dedup -> genre)
                         |
         vector_store.py (SQLite metadata + FAISS vectors + FTS5 keyword index)
                         |
         rag_engine.py   (query parsing -> SQL prefilter -> hybrid search -> cited answer)
                         |
         chatbot_api.py  (FastAPI) + static/index.html (web UI)
```

Archive and live input share **one** processing, storage and RAG path.

## What is implemented

| Area | Feature | Where |
|---|---|---|
| Ingestion | AIR archive scraper (pagination, robots.txt, retries) | `air_bulletins_scraper.py` |
| | Continuous live-stream ingestion with reconnect/backoff, restart-safe | `live_stream_ingest.py` |
| ASR | faster-whisper with audio preprocessing, VAD, word timestamps, cached model | `transcriber.py` |
| NLP | Sentence segmentation, `all-MiniLM-L6-v2` embeddings, MMR novelty dedup, zero-shot / fine-tuned genre | `nlp_pipeline.py` |
| Storage | SQLite (authoritative history) + FAISS (cosine) + FTS5 (BM25), safe schema migration | `vector_store.py` |
| Retrieval | Date / time-of-day / genre parsing, SQL prefilter, dense + BM25 fusion (RRF), relevance gate, sentence-window context | `rag_engine.py` |
| Generation | Gemini with mandatory `[S#]` citations, citation validation, explicit insufficient-context answer; extractive fallback without a key | `rag_engine.py` |
| Traceability | Every request/ingestion has a trace ID; logs show filters, hits, scores, sources, audio ranges, models and timings | `tracing.py` |
| API / UI | Chat with source cards, transcript viewer with highlighted segment, audio playback at the cited second, live status | `chatbot_api.py`, `static/index.html` |
| Newsroom | 11 long-form demo articles (headline, genre, summary, body, background, key developments, people, implications, takeaways), landing-page previews, genre filter, search, full reading view with related stories | `data/demo_articles.json`, `news_articles.py`, `static/index.html` |
| Detailed answers | Question typing (simple / standard / explain / overview / compare / gap) sets length (a few sentences up to ~300-600 words) and structure; whole articles as context; per-claim attribution | `rag_engine.py` |
| Evaluation | Hit rate, MRR, citation validity/correctness, groundedness, refusal accuracy, answer length and multi-article coverage | `eval_rag.py`, `eval/rag_eval_set.json`, `eval/rag_reasoning_set.json` |
| Broadcasts | Day-wise and per-programme browsing (Midday, News at Nine, Hourly, Morning, Evening, Other / Unclassified), combined filters, transcript search, date pagination | `vector_store.py`, `static/index.html` |
| Date-range ingestion | Start/end date + multi-select broadcast types, validation, discovery preview, confirmation, live stage progress, cancel, retry failed | `newsonair_ingest.py`, `chatbot_api.py` |
| Taxonomy | One genre configuration (18 genres, ids + names), rule-based classification with secondary tags, broadcast types from titles | `taxonomy.py` |
| Coverage | Actual coverage from indexed records, kept apart from the configured reporting window; missing dates classified | `analytics.py` |
| Analytics | Dashboard with dataset / coverage / ingestion / RAG KPIs and charts, filters, table views, dark mode | `analytics.py`, `static/analytics.html`, `static/charts.js` |
| Tests | 116 offline tests (live pipeline via ffmpeg or PyAV, RAG, articles, storage, API, broadcasts, date-range ingestion, taxonomy, coverage, analytics, live controls) | `tests/` |

Not yet implemented: multilingual (Hindi/regional) RAG — planned as the next phase, after the realtime pipeline is stable on a real stream.

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

FFmpeg must be on your PATH (`winget install Gyan.FFmpeg`, `sudo apt install ffmpeg` or `brew install ffmpeg`; check with `ffmpeg -version`).

Configuration lives in `config.py`; override anything with environment variables or a `.env` file. Start from the template:

```bash
copy .env.example .env      # Windows   (cp .env.example .env on Linux/macOS)
```

Models are downloaded from Hugging Face on first use (faster-whisper `small` is about 470 MB, MiniLM about 90 MB) and cached outside the repo.

## Running

### Quick start: real NewsOnAIR data

```bash
python newsonair_ingest.py --articles 10 --broadcasts 1   # small test batch
python newsonair_ingest.py --articles 80 --broadcasts 8   # initial dataset
python newsonair_ingest.py --status                       # what is indexed + recent jobs
uvicorn chatbot_api:app --port 8000                       # then open http://localhost:8000
```

Pages: `/` home, `/newsroom` (stories, **Broadcasts**, Ask, **Data**), `/analytics` (dashboard).

Date-range ingestion from the command line (dates are calendar dates in `VANI_TIMEZONE`, Asia/Kolkata):

```bash
python newsonair_ingest.py --from 2026-10-01 --to 2026-10-15 --articles 0 --types midday,news-at-nine
python newsonair_ingest.py --from 2026-10-01 --to 2026-10-15 --no-audio        # articles only
python newsonair_ingest.py --retry-failed                                      # retry every failed document
python newsonair_ingest.py --cancel NOA-20261103-010737                        # stop a running job between documents
python newsonair_ingest.py --recover-meta     # fill bulletin listings from the log + source availability from past jobs
python taxonomy.py --backfill                                                  # re-label genres in place (no re-embedding)
```

Re-running a command is incremental. Documents already indexed are skipped without re-embedding, documents the publisher changed are re-indexed (their old chunks are retired), and failed ones are retried, up to 3 attempts.

### 0. Real data: NewsOnAIR ingestion (`newsonair_ingest.py`)

**Sources.** Both were verified against the live site (November 2026). `robots.txt` allows all paths.

| What | Mechanism | Notes |
|---|---|---|
| Articles | The site's public WordPress REST API: `GET /wp-json/wp/v2/posts?lang=en&categories=<id>` (category ids from `/wp-json/wp/v2/categories?lang=en`) | The full article HTML is in the API response; it is converted to paragraphs. English only (`lang=en`). |
| Audio bulletins | The server-rendered archive `GET /national-bulletins/page/<n>/` | Each bulletin has a stable id, title, programme, language, broadcast date and time, an MP3 under `/wp-content/uploads/` and a `/listen-news/` page. English bulletins are downloaded and transcribed. |

The older audio-archive endpoint used by `air_bulletins_scraper.py` (`/wp-admin/admin-ajax.php`) now returns **403** to non-browser clients. The collector does not try to get around this. It uses the public pages above, sends an honest User-Agent, makes one request at a time with `NEWSONAIR_DELAY_SECONDS` (1.5 s) between requests, backs off on errors, and downloads each MP3 only once (cached under `output/audio/`).

**Pipeline.** Each stage reuses the existing modules:

```
NewsOnAIR -> collect (REST API / bulletin pages) -> text extraction | faster-whisper transcription
          -> genre (publisher category, else the existing classifier) -> chunking -> MiniLM embeddings
          -> FAISS + SQLite + FTS5 -> RAG retrieval -> cited answer
```

- **Articles** are chunked by `news_articles.chunk_article` (paragraphs packed into chunks of at most 150 words; a check rejects any article whose chunks do not contain every word). The whole article is stored in the `articles` table and used as answer context.
- **Bulletins** go through `transcriber.py` (faster-whisper with word timestamps), then `nlp_pipeline.process_transcript` (sentence chunks, MMR removal of repeated headlines across bulletins, genre classifier). Every sentence keeps its start and end second in the MP3.
- **Genres.** NewsOnAIR categories map to the existing AIR genres: Business→Economy, Sports→Sports, International→International, Health→Health, Science and Technology→Science & Technology, Entertainment→Society & Culture, Elections→Politics, Weather→Environment. National, Regional and Miscellaneous cover many topics, so their genre comes from the existing classifier. The publisher category is also kept (`source_category`), so "national news" filters to NewsOnAIR's National section. The genre is stored with the document and never re-classified.
- **Metadata on every chunk:** `doc_id`, chunk id (`faiss_id`) and `chunk_index`, `source_url`, `source_name`, `headline`, `genre`, `source_category`, `language`, `published_at`, `indexed_at`, `content_type` (`article` | `audio_transcript`), and `audio_start`/`audio_end` for transcripts.
- **Incremental state** lives in the existing SQLite database. The `ingestion_state` table holds one row per document (stable id `noa-post-<wp id>` / `noa-bulletin-<id>`, content hash, status, attempts, last error). The `ingestion_jobs` table holds one row per run with all counts. Headlines are never used to detect duplicates.
- **Short articles.** When the publisher's text is short (< 60 words), it is stored as-is with `content_limited: true`. Nothing is expanded or invented.

**ffmpeg.** If the `ffmpeg` binary is missing, `transcriber.py` decodes the audio with PyAV (bundled with faster-whisper) and skips the high-pass and loudness filters, logging a warning.

**Data tab.** The Newsroom's **Data** tab shows the database counts, the coverage panel, genres, NewsOnAIR categories, the date-range ingestion form, running jobs with live progress and the job history. Every control that starts work is disabled unless `INGEST_ADMIN_TOKEN` is set in `.env`; even then requests are accepted only from the server machine with that token (`X-Admin-Token`), batch sizes are capped and overlapping jobs are refused.

**Current dataset**

Read from the database on 2026-11-03 (`python newsonair_ingest.py --status`, `GET /coverage`):

| | |
|---|---|
| Articles (full text, English) | 2,075. October 2026: 1,936 (every English post of 2026-10-01..11-01). November 2026: 128. The other 11 are older posts from the first category-sampled runs. |
| Audio bulletins (English, transcribed) | 73, about 16.8 h of audio: Midday News 33, Evening News 30, Hourly News 7, News at Nine 2, Morning News 1. Every date from 2026-10-01 to 2026-11-01 has at least two main bulletins. |
| Indexed chunks | 8,858 (3,465 article chunks, 5,389 transcript sentences with audio time ranges). FAISS and SQLite agree. |
| Failed documents | 0. The 9 bulletins whose downloads dropped were re-fetched by the top-up job, and the last one by a targeted job. |
| Actual broadcast coverage | **2026-10-01 to 2026-11-02** (33 dates) |
| Actual article coverage | 2025-11-02 to 2026-11-02, but only 42 dates have articles: nearly all of them are in October and November 2026. |
| Reporting window (configured) | 2025-11-02 to 2026-11-02. 42 of its 366 dates have any indexed record. Article listings were checked for 2026-10-01..11-01; the other dates are "not checked", not "no news". |

**October run.**
- *Articles:* job `NOA-20261102-215447` indexed 1,985 posts, skipped 24 already indexed, 0 failed.
- *Bulletins, first pass:* `NOA-20261102-222110` (Midday + News at Nine, 2 per day) indexed 31. 9 failed on dropped connections, which led to the download retry.
- *Bulletins, top-up:* `NOA-20261103-002202` added Evening News for days without News at Nine, which NewsOnAIR does not publish every day, and retried the failures: 32 indexed, 0 failed.
- *Last fix:* `NOA-20261103-022306` added 15 Oct (the source has only Morning and Midday News that day) and the last failed Midday bulletin.

**Real-data evaluation.** `eval/newsonair_eval_set.json` has 12 questions taken from the indexed text: facts, a why-question, national/international overviews, a broadcast question that needs audio timestamps, and 2 off-topic questions. Results:

| Mode | Hit rate | Citation validity | Correct citation | Detail checks (doc cited, URLs, timestamps, category/genre filter, phrases) | Refusals |
|---|---|---|---|---|---|
| Extractive (no LLM) | 1.0 | 1.0 | 1.0 | 10/10 | 2/2 |
| Gemini (`gemini-flash-lite-latest`, via quota fallback) | 1.0 | 1.0 | 1.0 | 10/10 | 2/2 |

The reports are in `eval/newsonair_report*.json`, with every answer included. Specific details in the Gemini answers (for example "around 150,000 people have been evacuated", "100 to 110 km per hour") were checked by hand against the bulletin transcripts.

```bash
python eval_rag.py --set eval/newsonair_eval_set.json --out eval/newsonair_report.json
```

**Known limitations**

- **ASR errors.** Whisper `small` on CPU mishears some names ("Mitch Shaw" for Amit Shah, "Sita Raman"). Use `VANI_ASR_MODEL=medium` or `large-v3` (GPU) for better transcripts.
- **Genre classifier.** The fast zero-shot classifier labels some National/Regional stories loosely (for example, an extradition story as International). The publisher category is stored too and is what "national news" uses. The fine-tuned classifier (`--classifier finetuned`) can be trained on these labels.
- **English only.** Hindi and regional-language bulletins and posts are discovered but not ingested. Multilingual RAG is the next phase.
- **Missing author names.** Posts expose only a numeric WordPress author id, not author names, so names are not stored.
- **Short source articles.** Some Business updates are very short (30–70 words). They are stored as published and flagged `content_limited`.
- **Not a live feed.** Ingestion runs when started (CLI or dashboard). The public live HLS stream on the NewsOnAIR homepage was offline when checked on 2026-11-03: its playlist was frozen since 2026-10-09 and its segments returned 404. Live ingestion was verified on real AIR audio played through the same pipeline, not on a live broadcast.
- **Reconnect on finite inputs.** A healthy connection resets the retry counter, so a finite file or recording used as a "stream" is replayed until it is stopped. Every replayed sentence is rejected as a duplicate, so nothing is stored twice.
- **Transcript genres.** About 45% of bulletin sentences have no unambiguous topic word (sign-offs, fragments) and stay Other / Unclassified at sentence level. Bulletin-level genres use the classified sentences.
- **Sparse older coverage.** Before October 2026 only a handful of articles and no bulletins are indexed. The coverage displays show this, so the reporting window is never presented as covered.

## Broadcast browser

Broadcasts live in the Newsroom's **Broadcasts** tab (`/newsroom#broadcasts`). The old left sidebar is gone; it held the live-ingestion panel and a long unfiltered list of bulletins. The page now has, top to bottom:

1. A one-line **live ingestion** control (see *Live ingestion* below).
2. The actual **broadcast coverage**, kept apart from the reporting window.
3. The **mode** tabs.
4. A **search** field and a collapsible **Filters** panel, open on desktop and closed on phones, showing how many filters are active.
5. Compact one-line rows.

- **Modes:**
  - *All broadcasts:* one list, newest first, 30 rows per request.
  - *Day-wise:* grouped by date. Each date shows a one-line summary, for example "6 broadcasts · News at Nine · Hourly News ×4 · Midday News". Only the newest date starts open.
  - *Midday*, *News at Nine*, *Hourly News* and *Other / Unclassified:* that programme only, grouped by date. Evening and Morning News are under the Type filter.
  - Each tab shows its count.
- **Types** come from the title NewsOnAIR publishes (`taxonomy.BROADCAST_TYPES`). A title that matches none of them is shown as Other / Unclassified, never guessed.
- **Filters** combine freely: type, genre (a bulletin matches when any of its passages has that genre), language, date range, order and search. Search covers the title, the date and the words spoken (full-text index over the transcript).
- **Each row** shows date/time, title, type, language, segment count and duration (read from the MP3). **Open** loads the timestamped transcript (with the NewsOnAIR link and genres) only when clicked. **Play** loads the audio only when pressed.
- **Pagination:** rows (All) or dates (7 per request), with "Load more" and loading on scroll. List rows carry no transcript text, and rows have no entrance animation, so long lists stay fast.
- **Stored per bulletin** (`broadcasts` table, rebuilt from the chunks by `vector_store.sync_broadcasts`): bulletin id, title, date, time, type, language, audio URL, page URL, transcript source file, segment and timed-segment counts, duration, genres and ingestion status.

## Date-range ingestion

In the Data tab, or with `--from/--to/--types` on the command line:

1. **Validate:** start ≤ end, no future end date, not before `INGEST_EARLIEST_DATE` (2025-11-02), at most `INGEST_MAX_RANGE_DAYS` (31) days, known types, at least one type or articles. Each problem gets its own message, and dates are never adjusted silently.
2. **Discover (preview):** the bulletin archive is sorted newest first. The page where the range starts is found by doubling and then bisecting the page number, and only the pages covering the range are read. For 2026-10-01..03 that was 28 requests instead of about 140. Articles are counted with one API request (`X-WP-Total`). Listings go into `bulletin_catalog`, so the same range is not scanned again for `CATALOG_MAX_AGE_HOURS`.
3. **Show counts:** per type, discovered / already indexed / to process, plus documents that failed before. Articles show published / already indexed / at most new. A request over `INGEST_MAX_JOB_BROADCASTS` is refused with a message, not trimmed.
4. **Confirm**, then the job processes exactly the previewed bulletins: download → transcribe → chunk → embed → index.
5. **Progress:** stage and counts are written to `ingestion_jobs` during the job (stage, discovered, already indexed, downloaded, transcribed, chunks, indexed, failed, elapsed). The dashboard shows only the stages that apply; article-only jobs skip download and transcription.

**Safety and incrementality**

- Stable ids (`noa-bulletin-<id>`, `noa-post-<id>`) and content hashes. Indexed, unchanged documents are skipped; "reprocess" must be asked for explicitly.
- Failed documents keep their listing (`ingestion_state.meta_json`). **Retry failed** re-runs them without the attempt limit.
- Interrupted MP3 downloads are retried (whole file, bounded).
- **Cancel** is honoured between documents; documents already indexed stay.
- A job whose dates and types overlap a running job, in any process, is refused. A job whose process is gone is detected by its process id and marked failed. Jobs started by older code have no pid and get 24 hours.

## Genre taxonomy

`taxonomy.py` is the single genre configuration: 18 genres, each with an id and a display name (National News, International News, Politics & Governance, Business & Economy, Finance & Markets, Technology & AI, Science & Research, Health & Medicine, Education, Environment & Climate, Agriculture & Rural Development, Infrastructure & Transport, Defence & Security, Sports, Entertainment & Culture, Regional News, Social Issues, Other / Unclassified). Every one of them has indexed documents, so none is empty.

Classification is deterministic and makes no LLM calls:
1. A reliable publisher category decides first (Sports, Business, International, Health, ...). Keywords can refine it, for example Business → Finance & Markets.
2. Otherwise, keyword rules over the headline and opening text.
3. Otherwise, the zero-shot model label when its confidence is at least 0.5.
4. Otherwise, the publisher's National or Regional desk.
5. Otherwise, Other / Unclassified.

Bulletin passages are classified one by one: a sentence needs one unambiguous topic word, and ties stay unclassified. Up to three secondary tags are stored (`genre_tags`), and genre filters match the primary genre or a tag. The publisher category (`source_category`) and the original model label (`genre`) are preserved. Changing the taxonomy only rewrites metadata columns: `python taxonomy.py --backfill` re-labels every chunk in seconds without re-embedding.

## Coverage

The **reporting window** is configuration: `VANI_REPORT_START` / `VANI_REPORT_END`, default 2025-11-02 to 2026-11-02.

**Actual coverage** is always computed:
- It is the earliest and latest date that has at least one *active indexed chunk*. That date is the article's publication date or the bulletin's broadcast date, both in Asia/Kolkata.
- Failed or never-indexed documents have no active chunk, and superseded versions are inactive, so neither counts. Ingestion timestamps are never used.
- Rows with invalid dates are ignored.

The home page, the newsroom bar, the Data tab and the dashboard show both the window and the actual coverage. They also show how many dates in the window have records. Dates without records are split into:
- **no source content:** the source listing for that day was checked and had nothing;
- **source content not indexed:** the source had content that is not ingested yet;
- **not checked:** nobody has looked at the source for that day.

So a gap is never presented as a source outage.

## Analytics dashboard (`/analytics`)

All numbers come from SQLite on request. Results are cached per filter set and invalidated whenever the data changes. A metric with no recorded data shows "Not yet measured". The default period is the reporting window, extended to today so recent jobs and questions show up.

- **Filters** (one row, kept in the URL): date range, content (all / articles / broadcasts), broadcast type, genre, language, job status, interval (daily / weekly / monthly). A daily view longer than 120 days is drawn weekly, and the chart says so.
- **KPIs.**
  - **Dataset:** articles, broadcasts, chunks, unique documents, audio files processed (hours), broadcasts transcribed, genres, languages.
  - **Coverage:** earliest and latest indexed date, configured start and end, dates with records and %, dates without records split by cause.
  - **Ingestion:** jobs total / successful / with errors / failed / cancelled / running, documents discovered / skipped as duplicates / processed / failed, average job duration, latest successful ingestion.
  - **RAG:** queries, average retrieval / generation / end-to-end latency, average chunks, retrieval failures, no-context answers, citation coverage, queries by genre.
- **Charts.**
  - Indexed documents over time (articles vs broadcasts).
  - Broadcasts by type; genres (documents / articles / broadcasts / chunks); languages; NewsOnAIR categories.
  - Job history by outcome; daily ingestion activity (processed / duplicates / failed).
  - Processing funnel for articles and broadcasts. Document stages and passage stages are on separate scales, and transcription is marked "does not apply" for articles.
  - Query volume; latency; recent jobs table.
- **Presentation.** Every chart has hover tooltips, a legend and a **Table** view. The colours are a fixed categorical order validated for colour-vision deficiency in light and dark mode. The page has empty, loading and error states, a "Data as of" timestamp, a Refresh button and a responsive layout.

**Instrumentation**
- `ingestion_jobs` stores kind, date range, types, pid, stage, live progress counts, per-kind chunk/embedding counts, cancel flag and safely truncated errors.
- `query_log` stores one row per question: request id, time, genre scope or detected genre, dates, question type, mode, model, retrieved document ids, chunk/source counts, retrieval/generation/total ms, citation count and the no-context flag. It never stores the question text. Turn it off with `VANI_QUERY_LOG=0`.

## Performance

Measured with `python eval/perf_bench.py --label <name> --compare <other>` on the real index. The table shows the median of 5 requests on the same machine while a transcription job was running.

| Request | Before: median / payload | After: median / payload |
|---|---|---|
| articles (first page) (`/articles`) | 276 ms / 1,529.6 KB | 74 ms / 21.0 KB |
| articles genre=Sports (`/articles?genre=Sports`) | 187 ms / 138.8 KB | 65 ms / 20.8 KB |
| articles search 'india' (`/articles?q=india`) | 265 ms / 877.6 KB | 85 ms / 21.6 KB |
| broadcast list (`/sources`) | 41 ms / 15.6 KB | 50 ms / 19.8 KB |
| broadcasts day-wise (`/broadcasts?mode=day`) | did not exist | 11 ms / 15.6 KB |
| broadcasts Midday (`/broadcasts?type=midday`) | did not exist | 31 ms / 5.8 KB |
| corpus stats (`/stats`) | 79 ms / 0.4 KB | 86 ms / 0.4 KB |
| ingestion status (`/ingest/status`) | 162 ms / 4.5 KB | 173 ms / 8.3 KB |
| coverage (`/coverage`) | did not exist | 27 ms / 1.2 KB |
| analytics summary (`/analytics/summary`) | did not exist | 92 ms / 2.3 KB |
| analytics timeseries (`/analytics/timeseries?interval=day`) | did not exist | 38 ms / 6.2 KB |
| retrieval: "What did the Prime Minister say about renewable energy?" | 108 ms | 99 ms |
| retrieval: "Summarize the latest sports news" | 149 ms | 142 ms |
| retrieval: "What happened in the economy in October 2026?" | 113 ms | 120 ms |
| retrieval: "cyclone warning coastal districts" | 99 ms | 87 ms |
| retrieval: "India foreign policy developments" | 102 ms | 91 ms |

Raw results: `eval/perf_before.json`, `eval/perf_after.json`. Endpoints marked "did not exist" are new.

What changed:
- **Articles** are paginated in SQL (24 per page; search, genre and date filtering in the query). The newsroom used to download every article preview at once.
- **Broadcasts** are paginated by date from the `broadcasts` table, and transcripts load only when opened.
- **Schema checks** run once per process instead of on every connection. Each connection used to count the whole keyword index.
- **Keyword index** no longer re-adds retired chunks.
- **Indexes** were added for broadcast date and type, genre, content type and date, language, ingestion status, job time and query time.
- **Retrieval** drops near-identical passages repeated across bulletins before building the context.
- **Embedding model** is warmed up when the server starts. The first question used to wait about 50 s for it.
- **Dashboard** aggregates are cached, client requests are memoised, search inputs are debounced, and superseded responses are ignored.
- **Retrieval latency** is unchanged (within noise). The embedding model and FAISS index are the same, so no re-embedding was needed.


**Previous demo data.** The demo index (11 illustrative stories, 5 sample broadcasts, 95 chunks) was moved to `data/demo_index_backup/`. To look at it again, stop the server and copy those files back to the project root.

### 1. Archive pipeline (legacy scraper)

Scrape and transcribe bulletins (note: the AJAX endpoint this script uses currently answers 403; see above):

#### List categories
```bash
python air_bulletins_scraper.py --language english --list-categories
```

#### Download (and optionally transcribe)
```bash
# By month:
python air_bulletins_scraper.py --language english --category hourly-news --month 2024-05

# By date range:
python air_bulletins_scraper.py --language english --category hourly-news --from 2024-05-01 --to 2024-05-31

# With local Whisper transcription:
python air_bulletins_scraper.py --language english --category hourly-news --month 2024-05 --transcribe
```

Outputs:
- Audio: `output/audio/YYYY-MM-DD/*.mp3`
- Transcripts: `output/text/YYYY-MM-DD/*.txt`
- Metadata: `output/metadata.csv`

Each transcript also gets a `.segments.json` with word timestamps, so every stored sentence knows where it is in the audio.

Then index the transcripts:

```bash
python ingest_pipeline.py --fast          # only new transcripts
python ingest_pipeline.py --reindex --fast   # wipe and rebuild (explicit only)
```

### 2. API and web UI

```bash
uvicorn chatbot_api:app --port 8000
```

Open http://localhost:8000 for the **home page**: the project overview, the audio-to-knowledge pipeline, dataset statistics, the NewsOnAIR source section and a live preview of the stories. **Enter Newsroom** (or http://localhost:8000/newsroom) opens the app: stories, broadcasts, transcripts, live ingestion and the assistant. Ask a question, click a citation (`S1`) to open the article or the transcript at that sentence, or press **Play** to hear the exact audio range. The **Genre** selector under the question box limits the assistant to one genre, and choosing a genre chip in the Newsroom sets it too.

The VaniStream brand at the top left always returns to the home page. The sun/moon button switches between light and dark mode. The choice is saved in the browser and applied before the page paints. Both pages share `static/theme.css` (colour tokens, type, motion) and `static/theme.js` (theme, scroll reveals, count-up). Animations respect `prefers-reduced-motion`. Newsroom deep links: `/newsroom#article=<id>`, `/newsroom#ask`.

### 3. Live ingestion

Find the station's direct stream URL (newsonair.gov.in → Live Radio → browser DevTools → Network tab → filter `media` / `.m3u8` / `.mp3` → copy the URL), then set it in `.env`:

```
LIVE_STREAM_URL=https://<the-stream-endpoint-you-copied>
```

Then either:

```bash
python live_stream_ingest.py                 # standalone, Ctrl+C to stop
# or start/stop it from the web UI, or have the API start it automatically:
#   LIVE_AUTOSTART=1 in .env, then: uvicorn chatbot_api:app --port 8000
```

The project does not ship a stream URL. The live player on newsonair.gov.in embeds `https://airhlspush.pc.cdn.bitgravity.com/httppush/hlspbaudio233/hlspbaudio233_Auto.m3u8`. On 2026-11-03 its playlist had not changed since 2026-10-09 and its newest segment returned HTTP 404, so it is offline. The stream check reports exactly that and refuses to start. Without a URL, `/stream/start` and the CLI fail with a clear error.

**From the web UI** (Newsroom → Broadcasts, the live bar at the top):

1. When live ingestion is off, the bar shows only its state, which stream it would use and **Start live ingestion**.
2. **Settings** opens the stream URL field. Leave it empty to use `LIVE_STREAM_URL`; the page shows which one applies, by host and file name only. Settings also has an optional "On air" programme, which labels the windows with that broadcast type (otherwise they are Other / Unclassified), and **Check stream**.
3. **Start** validates the URL in the browser, then on the server (`probe_stream`):
   - the URL syntax;
   - that the host answers;
   - that the URL serves audio, not a web page;
   - for HLS, that the playlist is still updating and its newest segment downloads.

   The real reason is shown, for example "The stream at … appears to be offline: the playlist has not been updated since 2026-10-09 06:05 UTC; its newest audio segment answers HTTP 404".
4. The bar reads only the state the ingestion thread reports: *Connecting…* until the first audio arrives, then *On air*, *Reconnecting* after a drop, or *Stopped* with the error and a **Retry** button. While running it shows connection, last audio, segments stored, duplicates skipped, reconnects and running time. **Stop** flushes the last window and stops.

Control is limited to the server machine, and needs `X-Admin-Token` when `INGEST_ADMIN_TOKEN` is set. Status and stored chunks never contain the raw stream URL: credentials, query tokens and path are removed (`https://host/.../file.m3u8`).

**Without the ffmpeg binary**, live audio is decoded and windows are saved as MP3 through PyAV (the FFmpeg libraries bundled with faster-whisper). PyAV has an open/read timeout, so a dead stream ends and is retried instead of hanging. The 11 live-pipeline tests (`tests/test_live.py`) run on this fallback.

**How live ingestion behaves**

- Audio is decoded in 30 s windows with 5 s overlap. Words in the overlap tail are deferred to the next window, words already committed are dropped, and an unfinished sentence is carried over, so nothing is lost or doubled at boundaries.
- Each window with speech is kept as a small MP3 in `output/audio/<date>/Live-…mp3` (48 kbps, about 0.5 GB per 24 h). Every sentence stores that file, its start/end second inside it, and its absolute IST time (`ts_start`, `ts_end`, `date`, `hour`).
- The existing MMR gate compares new sentences with the last `LIVE_DEDUP_WINDOW_HOURS` (24) of segments. That merges a headline repeated at 09:00, 09:30 and 10:00, but doesn't block a genuinely new development days later.
- Disconnects, stalls (`LIVE_READ_TIMEOUT_SECONDS`) and ffmpeg exits trigger a reconnect with exponential backoff (5 s → 5 min cap). The default is to retry forever. A failing window (ASR, DB or FAISS error) is logged and skipped.
- Restarting never wipes data. FAISS is saved before SQLite commits, and `check_consistency()` removes any half-written rows after a crash.
- Segments are searchable as soon as they are stored.

**Checking that it works**

```bash
curl http://localhost:8000/health        # stream_connected, last_audio_received, last_transcription, segments_today, last_segment_id
python vector_store.py --stats           # by_mode.live grows
python vector_store.py --check           # FAISS and SQLite counts agree
python rag_engine.py "What happened today?"
```

The next morning, `python rag_engine.py "What happened yesterday?"` (or the UI) answers from the overnight records. "Yesterday" means the real previous IST date.

### 4. Gemini (optional)

**Provider:** Google Gemini API (`generativelanguage.googleapis.com`) through the `google-generativeai` SDK (0.8.x). Google has deprecated this SDK in favour of `google-genai`; it still works, and migrating is a separate change.

**Configuration.** Put the key in the environment, or copy `.env.example` to `.env` in the project root and fill it in. Never commit `.env`; the key is read only on the server and is never sent to the browser, written to logs or returned by any endpoint.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | empty | API key. Empty means extractive answers only. |
| `GEMINI_MODEL` | `gemini-flash-latest` | First model tried. |
| `GEMINI_FALLBACK_MODELS` | `gemini-flash-lite-latest,gemini-3.5-flash-lite` | Tried in order when a model is out of quota (429) or not found (404). |
| `GEMINI_COOLDOWN_SECONDS` | 300 | How long a 429/404 model is skipped. |
| `GEMINI_TIMEOUT_SECONDS` | 90 | Per-request timeout. SDK retries are disabled, so a 429 fails over in under a second instead of retrying for minutes. |
| `VANI_RAG_CONTEXT_MAX_WORDS` | 7000 | Total words of source text sent to the model. |
| `VANI_RAG_CONTEXT_MAX_WORDS_PER_SOURCE` | 1800 | Longer sources are cut with a visible `[... N more words of this source not shown]` marker. |

The fallback is explicit, not silent: the chat response's `model` field, the server log ("Answer generated by fallback model ...") and the API status card all name the model that answered. Without a working model, answers are assembled from cited source passages (extractive mode, `mode: "extractive"`).

**Connection check.** A key being set is not treated as "connected".

- `GET /ai/status` reports the provider, the configured and fallback models, whether a key is set (never its value), the last successful and last failed check per model, and the latest chat generation result. It makes no API call.
- `POST /ai/check` sends one minimal real request ("Reply with the single word OK", 5 output tokens) to each configured model, measures latency, stores the result in the `ai_health_checks` table and puts a 429/404 model on cooldown. It is accepted only from the server machine (plus `X-Admin-Token` when `INGEST_ADMIN_TOKEN` is set) and runs at most once every 30 s.
- The **Data** tab has an *Answer model (API)* card with a **Test connection** button that calls `/ai/check`.

States: `not_configured`, `unverified` (key set, never checked), `connected`, `auth_failed`, `model_unavailable`, `quota_exceeded`, `network_error`, `timeout`, `invalid_response`, `generation_failing` (the check passed but the latest chat generation failed), `sdk_missing`. From the command line:

```bash
python -c "import ai_health, json; print(json.dumps(ai_health.check_connection(force=True), indent=1, default=str))"
```

**Answer checks.** A generated answer with no `[S#]` citation, or with headings but no content, is rejected and regenerated once with a stricter instruction; if the retry is also unusable the extractive answer is returned. Every source in the prompt carries its title, programme, taxonomy genre, date and time, URL and (for broadcasts) the audio range.

**Fine-tuning.** Not available for this provider and account: no model returned by `list_models()` supports `createTunedModel`, and `list_tuned_models()` returns 501 (MethodNotImplemented). No fine-tuning job was run and the chatbot is not fine-tuned. Answer quality is improved through retrieval, the prompt and the answer checks above, and measured with the evaluation below. (The separate genre classifier in *Domain Fine-tuned Genre Classifier* is a local model and is unrelated to the chatbot.)

## Newsroom articles (demo content)

`data/demo_articles.json` holds 11 detailed articles (about 430-590 words each) across Environment, Economy, Science & Technology, Health, Sports, Politics and Society & Culture. **They are illustrative demo content**: the organisations, people and figures are fictional, and they are credited to "VaniStream Demo Desk", never to AIR or another real outlet. The UI and every API response label them as demo content. Replace the file with real, sourced articles for production use, keeping the same structure.

- The **Newsroom** tab (landing page) shows previews only: headline, genre, date, summary and reading time. Filter by genre or search the full text.
- Opening a story shows the full article: metadata, lede, body, background, key developments, people and organisations, implications, key takeaways and related stories in the same genre. "Ask for a detailed explanation" sends a question about the story to the Ask tab.
- Ingestion (`python ingest_pipeline.py`, or `--no-articles` to skip) chunks each article by section. Prose sections are packed paragraph by paragraph into chunks of at most 150 words; paragraphs are never split. `verify_coverage()` refuses to store an article whose chunks do not contain every word. Each chunk keeps `article_id`, `section`, genre, date and time. The complete article is stored in the `articles` table. Articles are not passed through MMR dedup, so no article content is dropped.
- `python news_articles.py --list` lists the articles; `--chunks <id>` shows how one is chunked.

API: `GET /articles?genre=&q=` (previews), `GET /articles/{id}` (full article + related).

## Speech recognition: why faster-whisper stays

The brief asked whether to replace Whisper with a free alternative. The candidates were compared on what this project needs: punctuation (sentence segmentation and MMR depend on it), word timestamps, CPU operation on Windows, a permissive licence, and Indian languages for the planned multilingual phase.

| Model | Licence | Strengths | Blocking issues here |
|---|---|---|---|
| **faster-whisper (Whisper, CTranslate2)** | MIT | Punctuated, cased output; word timestamps; built-in VAD; int8 on CPU; Hindi and other Indian languages | Slower than Parakeet on GPU |
| NVIDIA Parakeet TDT 0.6B (v2/v3) | CC-BY-4.0 | Very strong English accuracy and speed | Needs the NeMo toolkit (heavy, GPU/Linux-oriented, awkward on Windows); no Indian languages, which blocks the multilingual phase |
| Meta MMS-1B-all | CC-BY-NC-4.0 | 1,000+ languages | Non-commercial licence; CTC output without punctuation or casing breaks sentence segmentation; weaker English accuracy than Whisper |

Decision: keep faster-whisper and improve how it is used:

- one shared, cached model (the scraper used to reload it for every bulletin)
- ffmpeg preprocessing (high-pass filter and dynamic loudness)
- VAD and `condition_on_previous_text=False` against repetition loops on long audio
- word timestamps
- explicit handling of empty, silent and corrupt audio
- the default model raised from `base` to `small`

`transcriber.py` is the single interface, so a different backend can be added there without touching the rest.

Hardware guide: `small` int8 on a modern 4-core CPU transcribes a 30 s window in roughly 5–10 s, which keeps up with a live stream. Use `base` on weak machines. With an NVIDIA GPU, set `VANI_ASR_DEVICE=cuda` and `VANI_ASR_MODEL=large-v3` or `distil-large-v3`.

## Retrieval and answers

1. **Query understanding.** Handles "yesterday", "last week", "on 20 September", "around 7 PM", "between 6 and 8 PM", "this morning", "latest", and an explicit genre ("sports news", "in politics"). A topic word like "minister" or "cricket" is only a ranking hint, never a hard filter. If an explicit genre filter returns nothing, the search retries without it.
2. **SQLite prefilter** on date, HHMM hour and genre.
3. **Retrieval.** Topical questions use FAISS cosine plus FTS5 BM25, fused with Reciprocal Rank Fusion. A relevance gate (minimum cosine, relative to the best hit) drops unrelated chunks. Questions with no topic ("What happened yesterday?") get a chronological sample of the period.
4. **Context.** Each hit is widened with its neighbouring sentences from the same broadcast and labelled `[S1]…` with source, date, time and audio range.
5. **Answer.** Every sentence must cite `[S#]`. Inferences are prefixed "Inference:". Citations to unknown ids are removed and logged. If nothing relevant is found, the answer says so and reports the corpus coverage.

### Answer depth and grounding

When a hit comes from an article, all of its matching chunks are merged into one source, and the **whole article** is passed as context, so the model can draw on background, developments and implications rather than one passage.

Each question is typed, and the type sets the answer guidance sent to the model:

| Type | Example | Length / structure |
|---|---|---|
| simple | "Who is the chief engineer of the water board?" | 1-3 sentences, no headings |
| standard | "Tell me about the port pilot" | about 150-300 words |
| explain | "Why did dengue cases rise?" | about 300-600 words: Direct answer, Detailed explanation, Background and context, Key developments, Implications, Sources |
| overview | "Summarise the major health developments" | every story in the period or genre, same structure |
| compare | "Compare the battery pilot with the evening demand story" | adds "How the stories compare or connect"; every claim names its article |
| gap | "What is missing about the battery pilot's costs?" | adds "What the sources do not say" |

Longer answers must not invent anything. The prompt forbids statistics, quotes, dates or events not in the sources, and it requires a citation on every factual sentence and an "Inference:" label on conclusions the sources do not state. Repeating a fact to fill length is also forbidden. A model reply counts as a refusal only when the whole reply is the refusal sentence, so a partial answer that notes a gap is kept. Retrieval drops a topical question as not covered when its best match has cosine < `VANI_RAG_MIN_TOP_SCORE` (0.40). Without that floor, off-topic questions matched unrelated articles on the larger corpus.

The extractive fallback (no Gemini) follows the same structure, but every paragraph and bullet is copied from an article section and cited. Bullets whose facts already appear in the answer are skipped, and secondary articles are added only when their match is close to the best one.

## Terminal logs

Each request, archive file and live stream has a trace ID (`REQ-0007`, `ING-0003`, `LIVE-<id>`). Example from a real test run:

```
[2026-11-02 12:49:28] [INFO ] [LIVE-f319f1b6] [live        ] Audio window #1 received: 30.0s (18:19:28-18:19:58)
[2026-11-02 12:49:28] [INFO ] [LIVE-f319f1b6] [live        ] Transcription completed: 51 words heard, 43 new, 8 deferred to next window
[2026-11-02 12:49:28] [INFO ] [LIVE-f319f1b6] [nlp         ] MMR filter: 4 novel / 4 total (threshold=0.85)
[2026-11-02 12:49:28] [INFO ] [LIVE-f319f1b6] [store       ] Stored 4 segments -> FAISS ids 0..3 (date=2026-11-02, source=Live-En-181928-0001.txt)
[2026-11-02 12:49:28] [INFO ] [LIVE-f319f1b6] [live        ] Segment accepted: FAISS ID=1 | genre=Politics | 00:06-00:12 in Live-En-181928-0001.mp3 | "The Prime Minister inaugurated..."
[2026-11-02 12:49:28] [INFO ] [LIVE-f319f1b6] [live        ] Duplicate rejected (cos=1.00): "The Prime Minister inaugurated..."
[2026-11-02 12:51:30] [INFO ] [EVAL-0012] [rag         ] Query understanding: topic='solar corridors announced renewable energy ministry'
[2026-11-02 12:51:30] [INFO ] [EVAL-0012] [rag         ] Retrieval: dense=14 hits (top cos=0.816), keyword/BM25=4 hits (0.00s)
[2026-11-02 12:51:30] [INFO ] [EVAL-0012] [rag         ] Relevance gate: 1/14 candidates kept (min cos=0.25, top cos=0.816)
```

Logs also go to `logs/vanistream.log` (rotated). No API keys or full transcripts are logged.

## Tests and evaluation

```bash
pytest                      # offline tests (no network; provider responses are mocked)
python eval_rag.py          # retrieval and answer metrics on the current corpus
python eval_rag.py --set eval/rag_reasoning_set.json --out eval/rag_reasoning_report.json
```

The tests replace the embedding model, genre classifier and ASR with deterministic stand-ins, so they run without downloads. ffmpeg, SQLite, FAISS, the live windowing/stitching, MMR, retrieval and the API are the real code. The live tests stream a generated WAV through ffmpeg and cover sentences split across window boundaries, duplicate headlines, a crashing window, reconnection and restart.

`eval_rag.py` runs `eval/rag_eval_set.json` (15 hand-written questions over the sample transcripts in `output/text`, including 3 that must be refused). It reports hit rate, MRR, citation validity and correctness, groundedness, and refusal accuracy, and writes `eval/rag_eval_report.json`. To evaluate the real models, ingest the samples first (`python ingest_pipeline.py --reindex --fast`), then run it. Add your own cases to the JSON as the corpus grows.

`eval/rag_reasoning_set.json` holds 10 questions that need detailed reasoning over the articles: background and implications, comparisons, causes, a genre overview, missing information, relationships between events, one simple fact, and one that must be refused. Each case can also check `min_words` / `max_words`, `require_all_expected` (every expected article cited) and `must_mention`. Every answer is saved in the report for review by hand.

### Chatbot evaluation on real indexed documents

```bash
python eval/build_chatbot_eval.py      # -> eval/chatbot_eval_set.json (19 cases, fixed seed)
python eval_rag.py --set eval/chatbot_eval_set.json --out eval/chatbot_report_improved.json
```

The set is built from documents in `news_store.db`: 5 factual, 3 analytical, 3 genre-scoped, 1 cross-genre, 2 date summaries, 3 broadcast-transcript questions (audio timestamps required), and 2 questions whose topic words were verified absent from the index. It stores the expected documents and the source evidence; no expected answer text is written. Results on 3 Nov 2026 (Gemini `gemini-flash-lite-latest`, because the primary model was out of quota), same questions before and after the changes:

| Metric | Baseline | Improved |
|---|---|---|
| Hit rate / MRR | 1.0 / 0.843 | 1.0 / 0.843 |
| Citation validity / correctness | 1.0 / 1.0 | 1.0 / 1.0 |
| Groundedness (lexical proxy) | 0.711 | 0.783 |
| Refusal accuracy (2 unanswerable) | 0.5 | 1.0 |
| False refusals (17 answerable) | 0 | 0 |
| Median / max latency | 2.9 s / 30.0 s | 2.8 s / 35.6 s |
| API failures | 0 | 0 |

Reports: `eval/chatbot_report_baseline.json`, `eval/chatbot_report_improved.json`. The sample is small and single-run, so small differences are not significant.

The older root scripts (`test_genre.py`, `test_live_stream.py`, `test_audio_endpoint.py`, `test_resolve_audio.py`, `integration_test_live.py`) still work as manual checks against your local database and real models.

## Genre classifier tools

### Genre Classification Evaluation
Sample segments into an evaluation CSV and compute classification accuracy and confusion matrix:
```bash
# 1. Sample N random segments:
python eval_genre.py --sample 10

# 2. Fill in ground-truth labels in 'true_genre' column of eval_labels.csv

# 3. Evaluate accuracy and print confusion matrix:
python eval_genre.py --evaluate

# 4. Compare zero-shot vs fine-tuned (side-by-side matrices + accuracy delta):
python eval_genre.py --compare
```

### Domain Fine-tuned Genre Classifier
The fine-tuned classifier is a selectable alternative to zero-shot NLI. Zero-shot remains the default.

**Step 1 — Bootstrap training data:**
```bash
# Silver-label existing DB segments + transcript files + synthetic augmentation:
python build_genre_training_data.py --augment
```
This writes `genre_training_data.csv` with columns: `text, predicted_genre, confidence, true_genre`.
> ⚠️ **Human review required**: Open `genre_training_data.csv` and fill in the `true_genre` column before training. The `predicted_genre` values are zero-shot silver labels and may be incorrect.

**Step 2 — Train the classifier:**
```bash
python train_genre_classifier.py
# Optional flags:
python train_genre_classifier.py --epochs 10 --batch-size 32 --model distilbert-base-uncased
```
Saves the best checkpoint (by val accuracy) to `models/air-genre-classifier/` and prints per-class F1.

**Step 3 — Use the fine-tuned classifier:**
```bash
# Ingest pipeline:
python ingest_pipeline.py --classifier finetuned

# Standalone NLP pipeline:
python nlp_pipeline.py --transcript output/text/2026-05-01/file.txt --date 2026-05-01 --hour 8 --classifier finetuned

# Compare vs zero-shot:
python eval_genre.py --compare
```
> **Graceful degradation**: If `models/air-genre-classifier/` does not exist, `--classifier finetuned` logs a WARNING and automatically falls back to the zero-shot fast model — no crash, no exception.
