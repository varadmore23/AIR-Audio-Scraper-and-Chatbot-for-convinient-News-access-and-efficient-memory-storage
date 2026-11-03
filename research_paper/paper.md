---
title: "VaniStream: Source-Grounded Conversational Retrieval over Transcribed All India Radio Bulletins and NewsOnAIR Articles"
---

**Authors**

Varad More^1^, Manav Jadhav^1^, Soham Jadhav^1^, Anant Nimkar^1^

**Affiliations**

^1^ [Department, Institution, City, Country — to be confirmed by the authors]

**Corresponding author**

[Name and e-mail address — to be confirmed by the authors]

# Abstract {.unnumbered}

**Background.** Public-service radio news is published as long audio bulletins and short web articles. Listeners who want a specific fact must replay whole bulletins or search article text by keyword, and nothing links an answer back to the moment it was broadcast. Retrieval-augmented generation (RAG) can answer questions from a document collection, but broadcast news is repetitive across bulletins and time-sensitive, so similarity search alone tends to return the right story from the wrong day.

**Methods.** We built and evaluated VaniStream, a system that collects English news articles and audio bulletins from NewsOnAIR (newsonair.gov.in), transcribes the audio with faster-whisper (Whisper *small*, int8, CPU) with word timestamps, removes sentences that repeat earlier bulletins with a cosine novelty threshold, labels every item with an 18-genre rule-based taxonomy, and indexes 384-dimensional all-MiniLM-L6-v2 embeddings in FAISS alongside an SQLite/FTS5 metadata and keyword store. Questions are answered by a two-tier retriever: a structured SQL prefilter on date, time of day, genre and programme, followed by dense and BM25 search fused with reciprocal-rank fusion and gated for relevance. A Gemini model writes the answer, and the server checks that every citation points to a retrieved source. We measured the dataset, the ingestion jobs, a novelty-threshold sweep, a two-tier versus whole-index retrieval comparison on 150 date-anchored questions, answer quality on a 12-question evaluation set, and latency.

**Results.** The index holds 2,075 articles and 73 bulletins (16.78 h of audio), 8,854 active chunks, all in English. Broadcast coverage spans 2026-10-01 to 2026-11-02 (33 dates); article coverage spans 2025-11-02 to 2026-11-02 but only 42 dates have articles, 1,936 of them in October 2026. The novelty threshold λ = 0.85 stores 80.8% of 6,524 bulletin sentences while keeping 97.4% of their vocabulary. With the date prefilter, 96.7% of the top-5 passages come from the date named in the question, against 44.7% for the same hybrid search over the whole index. On the evaluation set, hit rate was 1.0 and citation validity and correctness were 1.0, but one of two unanswerable questions was answered on the larger corpus. Median warm retrieval took 32.5 ms (n = 162) and median warm end-to-end answer time 2.1 s (n = 10).

**Conclusions.** Structured temporal filtering before vector search, combined with server-side citation checking, gives grounded and correctly dated answers over a month of real broadcast news on commodity hardware. The evaluation set is small, transcription accuracy was not measured, and the findings should be read as a system study rather than a benchmark comparison.

**Keywords:** Audio information retrieval; Retrieval-augmented generation; News intelligence; Speech-to-text; Semantic search; Vector embeddings; Hybrid retrieval; Conversational AI; NewsOnAIR

# Introduction

Radio remains a primary news medium in India, and All India Radio publishes its bulletins on the NewsOnAIR portal both as audio recordings and as short text articles. A single day carries dozens of bulletins: hourly summaries, a midday bulletin, evening and late-evening bulletins, each roughly fifteen minutes long. The audio is the most complete record of what was broadcast, but it is the least accessible. A listener who wants to know what was said about a cyclone warning or a policy announcement must find the right bulletin and listen through it; keyword search over article text finds only the stories that were also written up, and it cannot say *when* a fact was broadcast.

Three properties of broadcast news make automated question answering harder than question answering over a static document collection. First, the content is spoken, so it must be transcribed, and every transcription error propagates into retrieval. Second, it is repetitive: the same headline is read in several consecutive bulletins, so a naive index stores many near-identical passages from different hours and days. Third, it is time-sensitive: "what did the news say on 12 October" has a correct answer that differs from "what did the news say about the same topic a week later", yet semantic similarity cannot distinguish the two. A retriever that ignores time will often return a correct-sounding passage from the wrong date, and a generator will then present it as the answer.

Retrieval-augmented generation [CITATION PLACEHOLDER: original RAG formulation] addresses hallucination by conditioning a language model on retrieved passages, and dense passage retrieval [CITATION PLACEHOLDER: dense passage retrieval for open-domain QA] made semantic retrieval practical. These techniques are established; this paper does not claim new retrieval or generation algorithms. Our contribution is a complete, measured system that combines them for broadcast news and the specific evidence we collected on real data:

1. **An implemented pipeline** from the NewsOnAIR website to cited answers, including polite incremental collection, word-timestamped transcription, cross-bulletin novelty filtering, an 18-genre taxonomy and a two-tier retriever whose first tier is a structured SQL filter on date, time of day, genre and programme (Section 4 and Section 6).
2. **A real-data dataset description** that keeps actual coverage separate from the configured reporting window (Section 5).
3. **Two controlled experiments on the indexed data**: a sweep of the novelty threshold λ over 6,524 bulletin sentences, and a comparison of two-tier retrieval against whole-index hybrid retrieval on 150 date-anchored questions (Section 10).
4. **An honest evaluation of answer quality and latency**, including the failures we observed (Section 10 and Section 11).

The rest of the paper is organised as follows. Section 2 defines the problem and the objectives. Section 3 reviews related work. Sections 4 to 6 describe the architecture, the dataset and the methods, and Section 7 gives the mathematical formulation. Section 8 works through the main calculations with real values, Section 9 describes the experimental setup, and Section 10 reports the results. Section 11 discusses limitations and Section 12 concludes.

# Problem Statement and Objectives

## Problem definition

Let $\mathcal{S}$ be a news source that publishes, over a period of days $\mathcal{T}$, a set of text articles $\mathcal{A}$ and a set of audio bulletins $\mathcal{B}$. Each article $a \in \mathcal{A}$ has a text, a publication time and a publisher category. Each bulletin $b \in \mathcal{B}$ has an audio recording, a programme title, a broadcast time and a language. The task is to build an index $\mathcal{D}$ of passages derived from $\mathcal{A}$ and from transcripts of $\mathcal{B}$ such that, for a natural-language question $q$, the system returns an answer $\hat{y}$ together with a set of cited passages $C(\hat{y}) \subseteq \mathcal{D}$ that

- support every factual statement in $\hat{y}$ (groundedness);
- satisfy the temporal, genre and programme constraints expressed in $q$ (constraint satisfaction);
- identify each source precisely: for an article its URL and publication time, for a bulletin the recording and the start and end second of the cited sentence (traceability);
- and, when no passage in $\mathcal{D}$ supports an answer, are empty, with $\hat{y}$ stating that the indexed news does not cover the question (calibrated refusal).

## Requirements

The implemented system addresses the following requirements.

- **R1 Collection.** Collect English articles and bulletins from NewsOnAIR without bypassing access restrictions, respecting robots.txt and rate limits, and record the provenance of every item.
- **R2 Audio to text.** Convert each bulletin into text with per-sentence audio timestamps.
- **R3 Metadata preservation.** Keep publication or broadcast date and time, programme, publisher category, language and source URL with every passage.
- **R4 Retrieval.** Retrieve passages that match both the meaning of the question and its explicit constraints.
- **R5 Grounded generation.** Generate answers only from retrieved passages, cite them, and refuse when evidence is insufficient.
- **R6 Genre-aware access.** Support restricting retrieval to a genre, chosen by the user or named in the question.
- **R7 Freshness.** Update the index incrementally, without re-processing or duplicating documents that were already indexed.

## Objectives and evaluation criteria

Table 1 lists the objectives and how each is measured in this paper.

Table: Table 1. Objectives and evaluation criteria.

| Objective | Measured by | Section |
|---|---|---|
| O1 Build a working pipeline from source to cited answer (R1–R5) | end-to-end runs on real data; automated test suite | 9, 10 |
| O2 Describe the collected dataset faithfully (R1, R3) | counts, coverage and distributions computed from the database | 5, 8 |
| O3 Remove cross-bulletin repetition without losing information (R2) | storage retained and vocabulary retained versus λ | 10.2 |
| O4 Retrieve passages from the requested date (R4) | date precision@5, date correctness@1 and source-sentence hit@5, two-tier versus whole-index | 10.3 |
| O5 Produce grounded, correctly cited answers and refuse otherwise (R5) | hit rate, MRR, citation validity and correctness, groundedness, refusal accuracy | 10.4 |
| O6 Answer at interactive speed | retrieval and end-to-end latency | 10.5 |
| O7 Ingest incrementally and recover from failures (R7) | per-job discovered, indexed, skipped and failed counts | 10.1 |

Transcription accuracy (word error rate) is a natural further criterion, but it was not measured because no reference transcripts exist for these bulletins (Section 11).

# Related Work

**Speech recognition.** Large weakly supervised sequence-to-sequence models made robust, multilingual transcription widely available [CITATION PLACEHOLDER: Whisper model paper]. VaniStream uses faster-whisper [CITATION PLACEHOLDER: faster-whisper / CTranslate2 implementation], a re-implementation of Whisper for efficient CPU inference with quantisation, together with voice-activity detection [CITATION PLACEHOLDER: Silero VAD or the VAD used by faster-whisper]. Errors in automatic transcripts are known to degrade spoken-content retrieval [CITATION PLACEHOLDER: spoken document retrieval and ASR error propagation].

**Semantic and dense retrieval.** Classical lexical retrieval ranks documents with term-weighting functions such as BM25 [CITATION PLACEHOLDER: BM25 / probabilistic relevance framework]. Dense retrieval represents queries and passages as vectors and ranks by similarity [CITATION PLACEHOLDER: dense passage retrieval]. Sentence-embedding models trained with siamese networks [CITATION PLACEHOLDER: Sentence-BERT] and compact distilled encoders [CITATION PLACEHOLDER: MiniLM] made sentence-level dense retrieval cheap. Efficient similarity search libraries index such vectors at scale [CITATION PLACEHOLDER: FAISS library paper].

**Hybrid retrieval and rank fusion.** Lexical and dense retrieval fail in different ways: dense models blur rare names and numbers, while lexical models miss paraphrases. Reciprocal rank fusion combines ranked lists without score calibration [CITATION PLACEHOLDER: reciprocal rank fusion].

**Retrieval-augmented generation.** RAG conditions a generator on retrieved documents [CITATION PLACEHOLDER: original RAG paper]; surveys of hallucination distinguish output that contradicts its sources from output that cannot be verified against them [CITATION PLACEHOLDER: survey of hallucination in natural language generation]. Systems that require attributable answers check that generated statements are supported by cited passages [CITATION PLACEHOLDER: attribution / citation evaluation in generative QA].

**Redundancy and novelty.** Maximal marginal relevance balances relevance against redundancy when selecting passages [CITATION PLACEHOLDER: maximal marginal relevance]. VaniStream uses only the novelty part of this idea, as an admission filter at ingestion time (Section 6.6).

**Audio and news information systems.** Broadcast news has long been a test bed for spoken document retrieval and topic segmentation [CITATION PLACEHOLDER: broadcast news retrieval benchmarks]. Zero-shot text classification by natural language inference [CITATION PLACEHOLDER: zero-shot classification via NLI] allows labelling news without task-specific training data.

The present work does not propose a new model. It differs from the work above in combining structured temporal and categorical filtering with hybrid retrieval and citation checking for a continuously published, repetitive broadcast source, and in reporting measurements on that source.

# System Architecture

Figure 1 shows the implemented architecture. Only components that exist in the repository are shown.

![Figure 1. VaniStream architecture as implemented. Ingestion runs as offline incremental jobs; question answering runs online in a FastAPI server. The article path (left) and the audio path (right) share the embedding, genre and storage stages; only bulletin sentences pass through the novelty gate.](figures/fig01_architecture.png){width=6.3in}

**Frontend.** A static web application served by the backend has three pages: a home page, a newsroom and an analytics dashboard. The newsroom lists articles with server-side pagination, provides a broadcast browser (all broadcasts, day-wise grouping, per-programme views, filters and full-text search over transcripts), a question-answering view and an ingestion panel. Every citation in an answer opens the cited article or the cited transcript sentence, and a play button streams the original recording from the cited second.

**Backend.** A FastAPI application exposes the question-answering endpoint (`/chat`), a timeline endpoint, paginated article and broadcast endpoints, transcript and audio endpoints, coverage and analytics endpoints, and ingestion control endpoints. Ingestion and live-stream control are accepted only from the server machine and, when configured, only with an administrator token.

**Ingestion pipeline.** The collector (Section 6.1) reads the NewsOnAIR WordPress REST API for articles and the server-rendered bulletin archive for audio. Jobs are incremental, run one at a time, record their progress in the database, and can be restricted to a date range and a set of programmes.

**Storage layer.** Passage vectors are stored in a FAISS `IndexFlatIP` index. All metadata — text, date, time, genre, programme, source URL, audio start and end, document identifier, activity flag — is stored in SQLite, which also holds an FTS5 full-text index for BM25 keyword search, the full text of every article, a per-bulletin table, the per-document ingestion state, the job history and a query log. FAISS ids are positional, so a superseded document version is marked inactive rather than deleted.

**Retrieval layer.** The retriever (Section 6.7) parses the question into structured filters, selects candidate passages with SQL (tier 1), runs dense and keyword search restricted to those candidates (tier 2), fuses the rankings, and applies relevance gates and cross-document de-duplication.

**Generation layer.** The context builder (Section 6.8) assembles whole articles and transcript sentence windows, and the generator (Section 6.9) prompts a Gemini model, validates citations, detects refusals and falls back to an extractive answer when no model is available.

**Monitoring and analytics.** Each request and job carries a trace identifier in the logs. The query log stores, per question, the filters, the retrieved document identifiers, chunk and source counts, retrieval, generation and total time, and the number of citations; it does not store the question text. The analytics dashboard computes dataset, coverage, ingestion and retrieval indicators from these tables on request.

# Dataset and Data Collection

## Source and provenance

All data come from NewsOnAIR (newsonair.gov.in), the news portal of All India Radio. Articles were read through the site's public WordPress REST API (`/wp-json/wp/v2/posts`, English language filter), which returns the full article HTML, its categories and its publication and modification times. Bulletins were read from the public, server-rendered archive (`/national-bulletins/page/N/`), which lists for each bulletin a stable identifier, title, programme category, language, broadcast date and time, an MP3 URL and a page URL. An older audio endpoint of the site returned HTTP 403 to non-browser clients; it was not used, and no access restriction was circumvented. The site's robots.txt allowed the paths used. Requests were sequential with a fixed delay of 1.5 s, and every MP3 was downloaded at most once. Content belongs to its publisher; every passage in the index keeps its source URL so that answers can be attributed. The authors' project is independent and not affiliated with the publisher.

## Collection period and coverage

Collection ran on 2–3 November 2026. The reporting window configured in the system is 2025-11-02 to 2026-11-02 (366 days). This window is a configuration value, not a statement of coverage. Actual coverage, computed from the dates of active indexed passages, is:

- **Broadcasts:** 2026-10-01 to 2026-11-02, with indexed bulletins on 33 distinct dates.
- **Articles:** 2025-11-02 to 2026-11-02, but with articles on only 42 distinct dates. Of the 2,075 articles, 1,936 were published in October 2026 and 128 on 1–2 November 2026; the remaining 11 are older posts collected by the first, category-sampled runs.

Figure 2 shows the difference between the window and the actual coverage. Section 8.6 computes the coverage percentage and explains why dates without records are not evidence that nothing was published.

![Figure 2. Actual indexed coverage compared with the configured reporting window. Each tick is a date with at least one active indexed article (blue) or bulletin (orange). Source: `segments` table.](figures/fig02_coverage.png){width=6.3in}


## Dataset size

Table 2 summarises the indexed dataset. All values are computed from the database (Section 9).

Table: Table 2. Indexed dataset (computed on 3 November 2026).

| Quantity | Value |
|---|---|
| Articles (full text, English) | 2,075 |
| Articles flagged as short (< 60 words, stored as published) | 75 (3.6%) |
| Audio bulletins collected, transcribed and indexed | 73 |
| Audio files on disk and readable | 73 |
| Total audio duration | 16.78 h |
| Bulletin duration, median (range) | 14.7 min (4.7–15.0 min) |
| Sentences produced by the transcriber | 6,770 |
| Bulletin sentences stored after the novelty gate | 5,389 |
| Bulletin sentences with audio start and end times | 5,388 |
| Article chunks | 3,465 |
| Active indexed chunks (articles + bulletin sentences) | 8,854 |
| Vectors in FAISS (including 4 superseded versions) | 8,858 |
| Languages | English (8,854 chunks) |

Articles account for 3,465 / 8,854 = 39.1% of chunks and bulletin sentences for 60.9%. All indexed content is English; Hindi and regional-language bulletins are listed by the source but were not collected (Section 11).

## Daily volume and broadcast types

Figure 3 shows the number of indexed articles and bulletins per day. October 2026 was collected completely for articles; for bulletins, two main bulletins per day were selected (Section 6.1).

![Figure 3. Indexed articles and broadcast bulletins per day, 1 October – 2 November 2026 (IST), by publication or broadcast date. Source: `segments` table.](figures/fig03_daily_documents.png){width=6.3in}


Table 3 and Figure 4(a) give the bulletins by programme. The programme is taken from the category that NewsOnAIR publishes for each bulletin. News at Nine is not broadcast every day; on days without it, the Evening News filled the second slot.

Table: Table 3. Indexed bulletins by programme.

| Programme | Bulletins | Audio (h) | Stored sentences |
|---|---|---|---|
| Midday News | 33 | 8.15 | 2,689 |
| Evening News | 30 | 7.34 | 2,319 |
| Hourly News | 7 | 0.55 | 126 |
| News at Nine | 2 | 0.50 | 189 |
| Morning News | 1 | 0.24 | 66 |
| **Total** | **73** | **16.78** | **5,389** |

![Figure 4. (a) Indexed bulletins by programme, with audio hours. (b) Share of transcribed sentences removed by the novelty gate at ingestion, by programme. Source: `broadcasts` table and ingestion logs.](figures/fig04_broadcasts.png){width=6.3in}

## Genre distribution

Every article and bulletin sentence carries one primary genre from an 18-genre taxonomy (Section 6.4). Figure 5 shows the distribution. For articles the five largest genres are International News (377, 18.2%), Politics & Governance (187, 9.0%), Sports (174, 8.4%), National News (170, 8.2%) and Business & Economy (165, 8.0%); only 6 articles (0.3%) are Other / Unclassified. Bulletin sentences are classified one at a time, and 2,422 of 5,389 (44.9%) remain Other / Unclassified, mostly greetings, sign-offs and sentences with no topical word.

![Figure 5. Primary genre distribution for articles (n = 2,075) and bulletin sentences (n = 5,389). Source: `articles` and `segments` tables.](figures/fig05_genres.png){width=6.3in}

## Cleaning and de-duplication

Article HTML is converted to paragraphs; invisible characters (zero-width spaces, word joiners, byte-order marks and soft hyphens) and non-breaking spaces are normalised. Short articles are stored exactly as published and flagged; no text is generated to lengthen them. Documents are identified by the publisher's identifiers (`noa-post-<id>`, `noa-bulletin-<id>`) and a SHA-256 hash of their content, so an unchanged document is skipped and a changed one replaces its older version. Cross-bulletin repetition is handled by the novelty gate (Section 6.6), which removed 1,381 of 6,770 transcribed sentences (20.4%).

## Data limitations

The dataset covers one month densely and the previous eleven months only sparsely. It contains English only. Bulletins were selected as two main bulletins per day, so hourly bulletins are under-represented. The transcripts are automatic and contain recognition errors, for example in personal names; their error rate was not measured.

# Methodology

## Data ingestion

**Source discovery.** For articles, a date-range job lists every English post published in the range through the REST API (100 posts per page, newest first). For bulletins, the archive is ordered newest first, so the page on which a date range begins is found by exponential search followed by bisection on the page number, and only the pages that cover the range are then read. Bulletins can be filtered by programme and by a maximum number per day; bulletins already in the index count towards that maximum.

**Metadata extraction.** For each article the collector stores the headline, paragraphs, excerpt, publisher category, publication date and time, modification time, URL, language and retrieval time. For each bulletin it stores the identifier, title, category, language, broadcast date and time, MP3 URL and page URL, and after processing the duration and the number of indexed sentences.

**Incremental ingestion and duplicate detection.** A per-document state table records the content hash, status, attempt count and last error. A document whose hash is unchanged is skipped; a changed article retires its old chunks before the new ones are added. Each job records discovered, downloaded, transcribed, chunked, embedded, indexed, skipped and failed counts, and its current stage, in a job table. A job that overlaps the dates and programmes of a running job is refused, and a job can be cancelled between documents without losing what was already indexed.

**Failure handling.** Network requests are retried with exponential back-off. An MP3 download interrupted mid-file is retried as a whole file. A document that still fails is recorded as failed with its error and listing, and can be retried later without an attempt limit. In the collection reported here, 9 downloads failed with broken connections in the first bulletin job and all 9 were indexed by a later job (Section 10.1).

## Audio transcription

Bulletins are transcribed with faster-whisper using the Whisper *small* model with int8 quantisation on the CPU, language fixed to English, beam size 5, the built-in voice-activity filter with a minimum silence of 500 ms, word-level timestamps, and conditioning on previous text disabled to avoid repetition loops on long audio. When the ffmpeg binary is available the audio is first filtered (high-pass at 60 Hz and dynamic loudness normalisation); on the machine used here ffmpeg was not installed, so audio was decoded with PyAV and these filters were not applied. The word timestamps are aligned to sentences, so every stored bulletin sentence carries the start and end second of its first and last word in the original recording. Transcripts and their timestamps are written to disk, so a bulletin is never transcribed twice.

## Text preprocessing

Transcript text is whitespace-normalised and split into sentences with the NLTK Punkt tokenizer, falling back to a punctuation-based split if the tokenizer data are unavailable. Article text keeps its paragraph structure. No stemming or stop-word removal is applied to the stored text; the keyword index applies Porter stemming internally, and stop-words are removed only when building keyword queries and when computing topic words for the relevance gate.

## Genre classification

Genres are assigned by a deterministic cascade with no language-model calls (Table 4). The taxonomy defines 18 genres with fixed identifiers and display names.

Table: Table 4. Genre assignment cascade for one item (first matching step decides).

| Step | Rule | Applies to |
|---|---|---|
| 1 Publisher category | a reliable NewsOnAIR category maps to a genre (e.g. Sports, Business, International, Health); keyword scores may refine a broad category (Business → Finance & Markets) | articles |
| 2 Keywords | weighted keyword score; headline matches count 3, matches in the first 120 words count 1; the best genre with score ≥ 3 wins, ties broken by a fixed priority | articles |
| 2′ Keywords | one unambiguous topic keyword in the sentence decides; ties leave the sentence unclassified | bulletin sentences |
| 3 Model label | the zero-shot NLI label (cross-encoder/nli-MiniLM2-L6-H768) mapped to the taxonomy, if its confidence is ≥ 0.5 | both |
| 4 Desk | the publisher's National or Regional desk | articles |
| 5 Fallback | Other / Unclassified | both |

Up to three secondary genres with keyword score ≥ 3 are stored as tags, and a genre filter matches the primary genre or a tag. For the 2,075 articles, the deciding step was the keyword rule for 1,201 (57.9%), the publisher category for 597 (28.8%), the desk for 254 (12.2%), the model label for 17 (0.8%), and the fallback for 6 (0.3%). The original publisher category and the original model label are kept, so the taxonomy can be changed and the metadata recomputed without re-embedding.

## Chunking

**Articles.** An article is divided into its sections, and within each section consecutive paragraphs are packed greedily into a chunk until adding the next paragraph would exceed $C = 150$ words. A paragraph is never split, so a paragraph longer than 150 words becomes a chunk of its own. List-like sections become one chunk each. There is no overlap between chunks ($O = 0$). Each chunk is stored with its article identifier, section name and position, and a coverage check rejects any chunking that loses a content word of the article. Words are whitespace-delimited tokens.

**Bulletins.** Each transcript sentence is one chunk, carrying its audio start and end times. Context across sentences is restored at answer time by adding neighbouring sentences (Section 6.8).

Figure 6 shows the resulting chunk lengths: article chunks have a median of 107 words (the 95th percentile is 147 words, close to the packing limit), bulletin sentences a median of 20 words.

![Figure 6. Chunk length distributions. (a) Words per article chunk, with the 150-word packing limit. (b) Words per bulletin sentence (values above 80 shown at 80). Source: `segments` table.](figures/fig06_chunk_lengths.png){width=6.3in}


## Embedding generation and novelty filtering

Each chunk is embedded with the sentence-transformers model all-MiniLM-L6-v2, which produces 384-dimensional vectors; vectors are L2-normalised before they are added to FAISS. For an article chunk the encoder input is the headline, the section name and the chunk text; for a bulletin, the sentence text. Embeddings are computed once per chunk and reused; changing the genre taxonomy does not require re-embedding.

Bulletin sentences pass a novelty gate before storage: a sentence is stored only if its cosine similarity to every previously stored bulletin sentence is below a threshold λ = 0.85 (Equation 5). For continuous live-stream ingestion the comparison set is restricted to the last 24 hours; for the archived bulletins reported here it is the whole stored set. Articles are not filtered, because two articles about the same event usually add different details.

## Retrieval

**Query understanding.** A rule-based parser extracts from the question an explicit or relative date range (e.g. "yesterday", "last week", "on 12 October 2026", resolved in Indian Standard Time), a time-of-day range ("evening" → 17:00–20:59), a hard genre filter when a genre is named as the subject of the question, a soft genre hint when only a topic word suggests a genre, the publisher's National or Regional desk, a request for the latest news, and the residual topic words. A genre chosen by the user in the interface is always a hard filter.

**Tier 1: structured prefilter.** An SQL query selects the active passages whose date, time, genre (primary or tag), desk, programme, content type and language satisfy the parsed filters. If a desk or genre filter derived from the question leaves no candidates, it is relaxed and a genre becomes a ranking hint.

**Tier 2: hybrid search.** The question is embedded with the same encoder, and FAISS returns the $k_c = 30$ candidates with the highest inner product (cosine similarity) among the tier-1 candidates, using an id selector so that only those passages are scored. In parallel, SQLite FTS5 returns the 30 best BM25 matches of the topic words among the same candidates. Passages found only by keyword search are scored for cosine similarity as well, so that every passage faces the same relevance gate. The two ranked lists are fused by reciprocal rank fusion (Equation 7).

**Relevance gates and de-duplication.** Five rules (Equations 8–10) remove weak matches: a minimum best cosine, a rule that rejects moderate matches whose topic words occur in no indexed passage, a per-passage threshold, removal of passages that nearly repeat a better passage from another document, and grouping of all chunks of an article into one source, of which at most $k = 6$ are kept.

## Context construction

For an article source, the matched chunk is shown as the quotation and the complete article text, with its headline, date, publisher category and URL, is used as context, so that the model can explain background and implications. For a bulletin source, the matched sentence is extended by one neighbouring sentence on each side from the same bulletin, and its audio start and end times, programme, date and time are included. Sources are numbered S1, S2, ... in rank order.

## Answer generation

The question is classified by regular expressions into one of six types — simple, standard, explain, overview, compare and gap — and each type adds a length and structure instruction to the prompt (for example 1–3 sentences for a simple factual question and about 300–600 words for an explanation). The system prompt requires that every factual sentence end with citations to the numbered sources; that only facts, names, numbers and dates present in the sources be used; that inferences be marked; that partially covered questions be answered only in part with the gap stated; and that a question the sources do not answer be met with a fixed refusal sentence.

Generation uses the Gemini API through the google-generativeai client, with a maximum of 8,192 output tokens. The configured model order is `gemini-flash-latest`, then `gemini-flash-lite-latest`, then `gemini-3.5-flash-lite`; a model that returns a quota or availability error is skipped for 300 s. In the runs reported here the answering model was `gemini-flash-lite-latest`, because the daily quota of the first model was exhausted.

After generation the server normalises grouped citations such as "[S1, S2]", removes and logs citations to sources that were not retrieved, and converts a model refusal into the system's "not covered" answer. If no model is available, an extractive answer is assembled from the most relevant sentences of the retrieved sources, with the same citations. When tier 1 or the gates leave no evidence, no model is called and the user is told which period and filters were searched.

# Mathematical Formulation

This section states the equations the implementation uses and the metrics it reports. Equations 1–10 describe the implemented system; Equations 11–16 define the measured quantities.

## A. Chunking

For a fixed-window chunker with window length $C$ and overlap $O$ applied to a document of $L$ tokens, the number of chunks is

$$N = \left\lceil \frac{L-O}{C-O} \right\rceil, \qquad L > O,\ C > O. \qquad (1)$$

VaniStream's article chunker uses no overlap ($O = 0$), so Equation 1 reduces to $\lceil L/C \rceil$, but it packs whole paragraphs rather than cutting at exactly $C$ words. For a section whose paragraphs have lengths $w_1, \dots, w_m$ with every $w_j \le C$, greedy packing closes a chunk as soon as the next paragraph would not fit, so any two consecutive chunks together exceed $C$ words. With $L = \sum_j w_j$, this gives the bounds

$$\left\lceil \frac{L}{C} \right\rceil \;\le\; N_{\text{greedy}} \;\le\; \max\!\left(1, \left\lceil \frac{2L}{C} \right\rceil\right). \qquad (2)$$

A paragraph with $w_j > C$ forms its own chunk, and each section is chunked separately, so the article's chunk count is the sum over its sections. For a bulletin, the chunk count is the number of stored sentences.

## B. Embedding representation

A chunk $x_i$ is encoded by the sentence encoder $f_\theta$ (all-MiniLM-L6-v2 with mean pooling, $d = 384$) and normalised:

$$\mathbf{e}_i = \frac{f_\theta(x_i)}{\lVert f_\theta(x_i) \rVert_2} \in \mathbb{R}^{384}. \qquad (3)$$

## C. Cosine similarity

FAISS `IndexFlatIP` scores a query vector $\mathbf{q}$ (normalised in the same way) against a passage vector $\mathbf{e}_i$ by inner product, which for unit vectors equals the cosine similarity:

$$s(\mathbf{q}, \mathbf{e}_i) = \mathbf{q}^{\top}\mathbf{e}_i = \frac{\mathbf{q}\cdot\mathbf{e}_i}{\lVert\mathbf{q}\rVert_2\,\lVert\mathbf{e}_i\rVert_2} \in [-1, 1]. \qquad (4)$$

## D. Novelty gate

Let $P_t$ be the set of bulletin sentences stored before sentence $t$ (for live ingestion, only those of the last 24 hours). Sentence $t$ is stored if and only if

$$\max_{j \in P_t} \; \mathbf{e}_t^{\top}\mathbf{e}_j < \lambda, \qquad \lambda = 0.85, \qquad (5)$$

with the maximum over an empty set taken as $-\infty$. This is the redundancy term of maximal marginal relevance used as a hard admission rule; there is no relevance term at ingestion time.

## E. Retrieval ranking

**Tier 1.** For parsed filters $\phi(q)$ (date range $[d_1, d_2]$, hour range $[h_1, h_2]$, genre $g$, programme set $P$, desk $c$), the candidate set is

$$V(q) = \{\, i \in \mathcal{D} : \text{active}_i \wedge d_i \in [d_1, d_2] \wedge h_i \in [h_1, h_2] \wedge g \in G_i \wedge p_i \in P \wedge c_i = c \,\}, \qquad (6)$$

where an unset filter is always satisfied, $G_i$ is the primary genre of passage $i$ together with its tags, and $p_i$ its programme.

**Tier 2.** Dense search ranks $V(q)$ by Equation 4 and keyword search ranks it by the FTS5 BM25 score; let $r_{\text{den}}(i)$ and $r_{\text{bm25}}(i)$ be the ranks of passage $i$ in the two top-30 lists. The fused score is the reciprocal rank fusion

$$\text{RRF}(i) = \sum_{m \in \{\text{den}, \text{bm25}\}} \frac{\mathbb{1}[i \in R_m]}{k_{\text{RRF}} + r_m(i)} + \frac{0.5}{k_{\text{RRF}}}\,\mathbb{1}[\text{hint}(q) = g_i], \qquad k_{\text{RRF}} = 60, \qquad (7)$$

where $R_m$ is the result list of method $m$ and the last term is a small boost for a passage whose genre matches a soft genre hint. FTS5 computes BM25 with its built-in parameters; the sign of its score is inverted so that larger is better.

**Relevance gates.** Let $s^{*} = \max_{i} s(\mathbf{q}, \mathbf{e}_i)$ over the fused candidates and $T(q)$ the topic words of at least four characters. The retriever returns no evidence if

$$s^{*} < \tau_{\text{top}} = 0.40 \quad\text{or}\quad \left( s^{*} < \tau_{\text{strong}} = 0.55 \;\wedge\; 2\,\bigl|\{w \in T(q) : w \notin \mathcal{D}\}\bigr| \ge |T(q)| \right), \qquad (8)$$

where $w \notin \mathcal{D}$ means that no active passage contains $w$ in the keyword index. Otherwise a passage $i$ is kept if

$$s_i \ge \max(\tau_{\min}, 0.6\,s^{*}) \quad\text{or}\quad \left(\text{BM25}_i > 0 \wedge s_i \ge 0.6\,\tau_{\min}\right), \qquad \tau_{\min} = 0.25. \qquad (9)$$

Kept passages are sorted by RRF, and a passage $i$ is dropped when a better-ranked passage $j$ from a different document has a word-set Jaccard similarity

$$J(i, j) = \frac{|W_i \cap W_j|}{|W_i \cup W_j|} \ge 0.8. \qquad (10)$$

Finally all chunks of one article are grouped into one source and the first $k = 6$ sources are kept.

## F. Generation

The answer is generated as

$$\hat{y} = \operatorname{LLM}_{\phi}\bigl(\pi_{\text{sys}},\ \gamma(\kappa(q)),\ q,\ \operatorname{Ctx}(\operatorname{Retrieve}(q, \mathcal{D}))\bigr), \qquad (11)$$

where $\operatorname{Retrieve}$ is the composition of Equations 6–10, $\operatorname{Ctx}$ the context builder of Section 6.8, $\kappa(q)$ the question type, $\gamma$ its length and structure guidance, $\pi_{\text{sys}}$ the system prompt and $\operatorname{LLM}_\phi$ the Gemini model. If $\operatorname{Retrieve}$ returns the empty set, no model is called and $\hat{y}$ is the "not covered" answer. The returned answer is post-processed so that its citation set satisfies $C(\hat{y}) \subseteq \{S_1, \dots, S_k\}$.

## G. Evaluation metrics

**Retrieval of a known passage.** For a question set $Q$ with one known relevant document per answerable question, the hit rate and mean reciprocal rank are

$$\text{Hit} = \frac{1}{|Q_a|}\sum_{q \in Q_a} \mathbb{1}[\operatorname{rank}_q \le k], \qquad \text{MRR} = \frac{1}{|Q_a|}\sum_{q \in Q_a} \frac{1}{\operatorname{rank}_q}, \qquad (12)$$

where $Q_a$ are the answerable questions, $\operatorname{rank}_q$ is the position of the first returned source whose document identifier matches the expected document (with $1/\operatorname{rank}_q = 0$ if it is absent) and $k$ is the number of returned sources.

**Date precision.** In the two-tier experiment each question names a date $d_q$. For the top-$K$ returned passages with dates $d_{q,1}, \dots, d_{q,K}$,

$$\text{DateP@}K = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{K}\sum_{r=1}^{K} \mathbb{1}[d_{q,r} = d_q], \qquad \text{DateCorrect@1} = \frac{1}{|Q|}\sum_{q \in Q}\mathbb{1}[d_{q,1} = d_q]. \qquad (13)$$

The source-sentence hit@5 is the share of questions for which the passage that the question was built from appears in the top five.

**Citations and groundedness.** Citation validity is the share of answers in which every cited identifier refers to a returned source. Citation correctness is the share of answered, answerable questions in which at least one cited source is the expected document. Groundedness is the share of cited answer sentences $u$ whose content words $W(u)$ are mostly found in the cited passages $\Pi(u)$:

$$\text{Grounded} = \frac{1}{|U|}\sum_{u \in U}\mathbb{1}\!\left[\frac{|W(u) \cap W(\Pi(u))|}{|W(u)|} \ge 0.6\right]. \qquad (14)$$

Groundedness is a lexical proxy: it detects unsupported wording, not unsupported meaning. Refusal accuracy is the share of unanswerable questions for which the system returned "not covered".

**Latency.** The server measures, for each request, the retrieval time $T_{\text{ret}}$ (query parsing, prefilter, hybrid search, gates and context assembly), the generation time $T_{\text{gen}}$ (the model call and citation checks) and the total time $T_{\text{total}}$:

$$T_{\text{total}} = T_{\text{ret}} + T_{\text{gen}} + T_{\text{other}}, \qquad (15)$$

where $T_{\text{other}}$ is measured as the remainder (question classification and answer post-processing). The stages run sequentially in the implementation, so the sum is exact.

## H. Coverage

For a reporting window $W$ of $|W|$ days and the set $D_{\text{idx}}$ of dates with at least one active indexed passage,

$$\text{Coverage} = \frac{|D_{\text{idx}} \cap W|}{|W|} \times 100\%. \qquad (16)$$

Coverage describes the index, not the source: a date outside $D_{\text{idx}}$ may simply not have been collected. The system records separately the dates on which the source listing was actually checked.

# Calculations and Numerical Analysis

All values in this section are computed from the project database and logs by the scripts listed in Section 9.

## Chunking of a sample article

The article *SEBI Steps Up Measures to Protect Investors from Online Financial Fraud* (`noa-post-1944513`) has one section of 14 paragraphs with lengths 24, 26, 28, 15, 22, 27, 21, 20, 26, 49, 27, 15, 27 and 36 words, so $L = 363$ and $C = 150$. Greedy packing proceeds as follows:

- Chunk 1: 24 + 26 + 28 + 15 + 22 + 27 = 142 words; adding the next paragraph (21) would give 163 > 150, so the chunk closes.
- Chunk 2: 21 + 20 + 26 + 49 + 27 = 143 words; adding 15 would give 158 > 150.
- Chunk 3: 15 + 27 + 36 = 78 words.

The result is $N = 3$ chunks of 142, 143 and 78 words. Equation 2 bounds the count by $\lceil 363/150 \rceil = 3 \le N \le \lceil 726/150 \rceil = 5$, so this article reaches the lower bound, which is also the value of Equation 1 with $O = 0$. Over all 2,075 articles the mean is 1.7 chunks per article (median 2, maximum 25); 1,034 articles (49.8%) fit in a single chunk.

## Embedding storage

Each vector has $d = 384$ single-precision components, i.e. $384 \times 4 = 1{,}536$ bytes. The FAISS index holds 8,858 vectors, so the raw vector data occupy $8{,}858 \times 1{,}536 = 13{,}605{,}888$ bytes (about 13.6 MB). A flat index scores every candidate exactly; at this size the cost of exact search is negligible (Section 10.5).

## Content proportions and genre percentages

Of the 8,854 active chunks, $3{,}465 / 8{,}854 = 39.1\%$ are article chunks and $5{,}389 / 8{,}854 = 60.9\%$ bulletin sentences. Counted as documents, articles are $2{,}075 / (2{,}075 + 73) = 96.6\%$ of the collection. The genre shares in Section 5.5 are computed as $n_g / 2{,}075$; for example International News is $377 / 2{,}075 = 18.2\%$. The share of unclassified bulletin sentences is $2{,}422 / 5{,}389 = 44.9\%$.

## Ingestion success and duplicate rates

**Articles.** The five article-collecting jobs discovered 2,169 posts in total (a post found by several jobs is counted each time). Of these, 2,078 were indexed — 2,075 new and 3 updated versions — and 91 were unchanged and skipped. No article failed. With the eligible set defined as discovered minus unchanged,

$$\text{Article success} = \frac{2{,}078}{2{,}169 - 91} \times 100 = 100.0\%, \qquad \text{Duplicate rate} = \frac{91}{2{,}169} \times 100 = 4.2\%.$$

**Bulletins.** The bulletin jobs made 83 processing attempts: 73 were indexed, 9 failed because the connection broke during the MP3 download, and 1 was a bulletin already indexed by an earlier job. All 9 failed bulletins were indexed by a later job. Per attempt and per unique bulletin,

$$\text{Attempt success} = \frac{73}{83 - 1} \times 100 = 89.0\%, \qquad \text{Final success} = \frac{73}{73} \times 100 = 100\%.$$

In the first October bulletin job alone, 31 of 40 selected bulletins (77.5%) were indexed. Transcription succeeded for every downloaded file: 73 of 73 (100%); all failures occurred before transcription.

**Processing rate.** The October top-up job processed 32 bulletins with 27,590.3 s of audio in 7,153.6 s of wall-clock time, including sequential downloads with 1.5 s request delays, transcription, embedding and indexing. This is $7{,}153.6 / 27{,}590.3 = 0.26$ s of processing per second of audio, or 223.6 s per bulletin. The article job for October processed 2,009 posts in 1,566.6 s (0.78 s per post, dominated by the request delay and embedding).

## Novelty filtering

At ingestion the gate stored 5,389 of 6,770 transcribed sentences, removing $1{,}381 / 6{,}770 = 20.4\%$. By programme (Figure 4b) the removed share was 43.0% for Hourly News (95 of 221), 32.0% for Morning News (31 of 97), 22.3% for Evening News (667 of 2,986), 17.6% for Midday News (573 of 3,262) and 7.4% for News at Nine (15 of 204). Hourly bulletins repeat the most because consecutive hours summarise the same headlines.

## Coverage

With $|W| = 366$ days and $|D_{\text{idx}} \cap W| = 42$ dates, Equation 16 gives $42/366 \times 100 = 11.5\%$. For broadcasts alone, 33 of the 366 days (9.0%) have indexed bulletins, but within October 2026 every date from 1 October to 1 November has at least two. The source listing was checked for 33 dates for articles and 7 dates for bulletins; on all of them the source had content. The remaining 324 dates without records were not checked and must not be read as days without news.

## Latency statistics

Warm retrieval latency over 162 questions (Section 9.3) had mean 37.5 ms, median 32.5 ms, 90th percentile 50.4 ms, 95th percentile 54.1 ms and maximum 177.3 ms. In the generative evaluation run, end-to-end answer time over 11 answered questions had median 2.3 s; the first question took 192.7 s, of which 155.0 s was the model call: the first model in the chain had exhausted its daily quota, and the request waited for that error before falling back to the next model. The warm median over the other 10 questions is 2.1 s (mean 2.4 s, maximum 4.3 s), of which generation is 2.0 s at the median. The application's query log holds 11 real requests; for the 7 warm generative ones the median total was 3,926 ms and the maximum 22,201 ms, the latter caused by a slow model response.

# Experimental Setup

## Environment

Table 5 lists the verified environment. All experiments ran on one laptop. A GPU is present but was not used: transcription ran on the CPU as configured.

Table: Table 5. Experimental environment.

| Item | Value |
|---|---|
| CPU | 12th Gen Intel Core i5-12500H (12 cores, 16 threads) |
| Memory | 15.7 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (not used) |
| Operating system | Windows 11 Home (build 10.0.26300) |
| Language | Python 3.13.7 |
| Web framework | FastAPI 0.136.1, Uvicorn 0.46.0 |
| Speech-to-text | faster-whisper 1.2.1, Whisper *small*, int8, CPU, beam 5, VAD, word timestamps |
| Audio decoding | PyAV 17.0.1 (ffmpeg binary not installed) |
| Embeddings | sentence-transformers 5.5.0, all-MiniLM-L6-v2, 384 dimensions |
| Zero-shot classifier | cross-encoder/nli-MiniLM2-L6-H768 (transformers 5.8.1, torch 2.11.0) |
| Vector index | faiss-cpu 1.13.2, IndexFlatIP |
| Metadata and keyword index | SQLite 3.50.4 with FTS5 |
| Sentence segmentation | NLTK 3.9.4 (Punkt) |
| Generation | Gemini API via google-generativeai 0.8.6; answering model gemini-flash-lite-latest |

## Configuration

Table 6 lists the retrieval and indexing parameters as configured in the repository.

Table: Table 6. Retrieval and indexing parameters.

| Parameter | Value |
|---|---|
| Article chunk limit $C$ / overlap $O$ | 150 words / 0 |
| Novelty threshold λ (bulletins) | 0.85 |
| Dense and keyword candidates $k_c$ | 30 each |
| RRF constant $k_{\text{RRF}}$ | 60 |
| Best-match floor $\tau_{\text{top}}$ / strong-match level $\tau_{\text{strong}}$ | 0.40 / 0.55 |
| Per-passage floor $\tau_{\min}$ | 0.25 |
| Near-duplicate Jaccard threshold | 0.8 |
| Sources per answer $k$ | 6 |
| Transcript context neighbours | 1 on each side |
| Maximum output tokens | 8,192 |
| Collector request delay | 1.5 s |

## Evaluation data and procedures

**Experiment 1 (novelty threshold).** All 73 transcripts were re-segmented into sentences, sentences with fewer than four words were dropped (6,524 sentences remained), and the sentences were processed in broadcast order. For each $\lambda \in \{0.50, 0.55, \dots, 0.95\}$ Equation 5 was applied over the full history, and we recorded the share of sentences stored and the share of the vocabulary (lower-cased words of at least three letters, excluding stop-words; 11,866 distinct words) that occurs in at least one stored sentence.

**Experiment 2 (two-tier retrieval).** We drew 150 indexed bulletin sentences longer than 70 characters at random (fixed seed) and turned each into a question of the form "What did the news on ⟨day month year⟩ say about ⟨first seven content words of the sentence⟩?". The question's date is the sentence's broadcast date; a question was kept only if the parser resolved that date. The two-tier condition used the normal retriever; the baseline used the same hybrid search, gates and de-duplication over the whole index, without the date filter. Because the questions reuse the sentence's own words, this experiment measures temporal precision, not paraphrase robustness.

**Answer evaluation.** The NewsOnAIR evaluation set contains 12 hand-written questions about indexed content: 10 answerable questions (facts, an explanation, overviews of national and international news, a question that requires a bulletin with audio timestamps) with an expected document for each, and 2 unanswerable questions (the price of Bitcoin and a FIFA World Cup final). Each answerable question also has detail checks (for example a phrase that must appear, a word limit, a genre or category filter, links to the source site). The set was run on 2 November 2026 with 81 articles and 8 bulletins indexed, and again on 3 November 2026 on the full dataset, both with Gemini and, on 3 November, also with the extractive fallback.

**Latency.** Retrieval latency was measured in-process on the 12 evaluation questions and the 150 Experiment 2 questions after one warm-up query (162 measurements). End-to-end times come from the generative evaluation run and from the application's query log.

**Reproducibility.** The statistics, experiments and figures are produced by `eval/phase3_experiments.py`, `research_paper/scripts/latency_bench.py`, `research_paper/scripts/compute_stats.py` and `research_paper/scripts/make_figures.py`; the computed values are in `research_paper/data/paper_stats.json` and each figure's plotted values are saved next to it as CSV. The repository's automated test suite (116 tests) passed on the final code.

# Results and Discussion

## Ingestion

Figure 7 summarises ingestion. All 2,078 eligible article discoveries were indexed and the 91 unchanged ones were skipped without re-embedding, which shows that the content hash makes repeated runs incremental. For bulletins, the only failures were interrupted downloads; after the download retry was added, a second job indexed all 9 together with the rest of the month. For a three-day range (1–3 October 2026), the archive page search found the start of the range near page 130 and read 27 archive pages in total (28 requests including one article count); a sequential scan from the first page would have read about 157 pages.

![Figure 7. Ingestion outcomes. (a) Bulletin processing attempts across all jobs. (b) Article discoveries across all jobs; a post discovered by several jobs is counted each time. Source: `ingestion_jobs` table.](figures/fig07_ingestion.png){width=6.3in}


## Experiment 1: novelty threshold

Figure 8 and Table 7 show the sweep. Storage falls steadily as λ decreases, but vocabulary falls slowly at first and quickly below λ ≈ 0.75. At the configured λ = 0.85 the gate stores 80.8% of the sentences and keeps 97.4% of the vocabulary; at λ = 0.75, 28.6% of the sentences are removed but 6.0% of the vocabulary is lost; at λ = 0.50 two thirds of the sentences are removed and a third of the vocabulary. The configured value therefore removes about a fifth of the sentences, almost all of them repetitions, while losing few distinct words such as names and figures. Vocabulary retention is a proxy for information retention; the sweep does not measure whether removed sentences carried facts absent elsewhere.

![Figure 8. Novelty-threshold sweep over 6,524 sentences of 73 bulletins: share of sentences stored and share of vocabulary retained as a function of λ. Source: `eval/phase3_results.json`.](figures/fig08_lambda_sweep.png){width=4.6in}

Table: Table 7. Novelty-threshold sweep (6,524 sentences, vocabulary 11,866 words).

| λ | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| Sentences stored (%) | 32.1 | 50.3 | 65.5 | 71.4 | 76.5 | 80.8 | 84.5 | 88.0 |
| Vocabulary retained (%) | 66.0 | 81.9 | 91.0 | 94.0 | 95.9 | 97.4 | 98.4 | 99.3 |

The same embeddings also measure how much consecutive bulletins repeat each other. Among same-day consecutive pairs, a mean of 36.9% of the sentences of an Hourly News bulletin had a counterpart with cosine ≥ 0.85 in the previous hourly bulletin (5 pairs); for the other 35 consecutive pairs the mean was 10.6%. Repetition is therefore substantial for hourly summaries but much lower than is sometimes assumed for broadcast news in general.

## Experiment 2: two-tier versus whole-index retrieval

Figure 9 and Table 8 compare the two conditions on 150 questions. With the date prefilter, 96.7% of the top-five passages came from the date named in the question, against 44.7% without it; the top passage had the right date in 96.7% of questions against 82.0%. The search space shrank from 8,858 vectors to 272 candidates on average, a factor of about 32.

![Figure 9. Two-tier retrieval (SQL date prefilter followed by hybrid search) versus the same hybrid search over the whole index, on 150 date-anchored questions. Source: `eval/phase3_results.json`.](figures/fig09_two_tier.png){width=4.6in}

Table: Table 8. Two-tier versus whole-index retrieval (150 questions, %).

| Metric | Two-tier | Whole index |
|---|---|---|
| Date precision@5 | 96.7 | 44.7 |
| Date correct@1 | 96.7 | 82.0 |
| Source sentence in top 5 | 92.7 | 94.0 |
| Mean candidates searched | 272 | 8,858 |

The whole-index search found the source sentence about as often as the two-tier search (94.0% against 92.7%), so the date filter does not mainly improve finding the passage. Its effect is on what surrounds it: without the filter, more than half of the top-five context consists of similar sentences from other days. These are exactly the passages that lead a generator to state a correct-looking fact under the wrong date. The small loss in source-sentence hit rate comes from questions whose date parsing or gating removed the target sentence. Because the questions reuse the source sentence's words, these numbers are an upper bound for real paraphrased questions.

## Answer quality and citations

Figure 10 and Table 9 report the evaluation set. Hit rate was 1.0 in all three runs: an expected document was always among the returned sources. Citation validity was 1.0: no answer cited a source that was not retrieved. Citation correctness was 1.0: every answered question cited its expected document. Groundedness ranged from 0.836 to 0.911; the extractive answers score highest because they quote the sources directly.

![Figure 10. Answer-level evaluation on the 12-question NewsOnAIR set: Gemini on 2 November (81 articles indexed), Gemini on 3 November (2,075 articles) and the extractive fallback on 3 November. Source: `eval/newsonair_report*.json`.](figures/fig10_rag_eval.png){width=6.3in}

Table: Table 9. Answer-level evaluation (10 answerable and 2 unanswerable questions).

| Metric | Gemini, 2 Nov | Gemini, 3 Nov | Extractive, 3 Nov |
|---|---|---|---|
| Indexed articles / bulletins | 81 / 8 | 2,075 / 73 | 2,075 / 73 |
| Hit rate | 1.000 | 1.000 | 1.000 |
| MRR | 0.900 | 0.808 | 0.808 |
| Citation validity | 1.000 | 1.000 | 1.000 |
| Citation correctness | 1.000 | 1.000 | 1.000 |
| Groundedness | 0.836 | 0.840 | 0.911 |
| Refusal accuracy (2 unanswerable) | 1.0 | 0.5 | 0.5 |
| False refusals (10 answerable) | 0 | 0 | 0 |
| Detail checks passed | 10/10 | 9/10 | 9/10 |
| Mean words, explanatory answers | 564 | 483 | 507 |

Growing the corpus 25-fold lowered MRR from 0.900 to 0.808, because more stories now compete for the top ranks, and it produced the first refusal error. The question about a FIFA World Cup final, meant to be unanswerable, was answered on 3 November: the corpus now contains an article about an India–Uruguay FIFA friendly and a passage about an under-15 FIFA tournament, so the relevance gate (Equation 8) let the question through. Gemini's answer stated that the sources contain no information about a World Cup final and cited the related tournament sentence; the extractive fallback, which cannot make that judgement, listed loosely related sports sentences. The Bitcoin question was refused in all runs. The one failed detail check is an artefact of the evaluation script: it requires every source for the "international news" question to carry the older model label "International", while the genre scope now also admits documents whose secondary tag is International News (3 of the 8 returned sources).

In the application's query log, 7 of the 8 logged generative answers contained citations; one answer to a politics-scoped question was returned without citations. The server removes invalid citations but does not yet reject an answer that has none.

## Latency

Figure 11(a) shows warm retrieval latency: median 32.5 ms and 95th percentile 54.1 ms over 162 questions, so retrieval is not the bottleneck. Figure 11(b) shows end-to-end time per evaluation question on a logarithmic scale; apart from the first request (192.7 s, dominated by the quota fallback described in Section 8.7), every answer took between 1.3 s and 4.3 s, and at the median 2.0 s of the 2.1 s went to generation. Retrieval time inside the server, including context assembly, was 8–409 ms for warm requests in the query log (median 95.5 ms). One earlier logged request took 51.2 s in retrieval because the embedding model was loaded on first use; the server now loads it at start-up.

![Figure 11. Latency. (a) Warm retrieval latency on 162 questions. (b) End-to-end answer time per question in the generative evaluation run, split into generation and the remaining time; logarithmic scale. Source: `research_paper/data/retrieval_latency.json` and the evaluation log.](figures/fig11_latency.png){width=6.3in}

## Genre-aware retrieval

A genre scope chosen in the interface is a hard filter on the primary genre or a tag (Equation 6). In the evaluation, the international-news overview returned 8 sources: 5 with primary genre International News and 3 that carry it as a secondary tag. Because 44.9% of all 5,389 bulletin sentences are Other / Unclassified (Section 5.5), a strict genre filter over bulletins can miss relevant sentences that the classifier did not label; the retriever relaxes a genre filter derived from the question when it finds nothing, but not one chosen by the user.

## Strengths and failure cases

The results show three strengths. Date filtering before similarity search removes most wrong-date context at negligible cost. Citations are traceable: every bulletin citation links to the recording and the exact second, and 5,388 of 5,389 stored sentences have timestamps. Ingestion is incremental and recovers from partial failures without duplicates.

The failure cases are equally clear. The relevance gate can admit a question whose topic is only partly covered (the FIFA case). The extractive fallback cannot recognise that its passages do not answer the question. One generative answer lacked citations. Transcription mis-recognises some names; for example the transcript of a 2 November bulletin renders "Amit Shah" as "Mitch Shaw". The keyword index then cannot match the correct spelling, and only the dense search and the audio link can help a user.

# Limitations

**Coverage.** The index covers October 2026 densely and the previous eleven months sparsely, so questions about earlier periods will mostly be refused.

**Source availability.** Collection depends on the publisher's site structure and on the public archive; the older audio endpoint was not usable, and the live audio stream found on the site was offline when checked on 3 November 2026, so live ingestion was verified only on recorded audio.

**Transcription quality.** Word error rate was not measured, because no reference transcripts exist; measuring it requires manually transcribing a sample of bulletins. Errors in names and rare words affect keyword retrieval.

**Language.** Only English content was collected and the pipeline is configured for English.

**Retrieval relevance.** The relevance gate uses fixed thresholds chosen during development; they were not tuned on a held-out set, and the FIFA case shows they can admit partly covered questions.

**Evaluation scale.** The answer-level evaluation has 12 questions and was written by the authors with knowledge of the corpus; it is not blind. Experiment 2 uses questions generated from indexed sentences. Groundedness is lexical. No human judgement of answer quality was made beyond the detail checks.

**Context window and API availability.** Whole articles are placed in the prompt; very long articles enlarge the context. Generation depends on an external API whose quota was exhausted during the study, which forced the use of a smaller model, and whose latency dominated the answer time.

**Latency sample.** End-to-end latency is based on 11 evaluation answers and 11 logged requests; percentiles from such small samples are indicative only.

# Conclusion and Future Work

We presented VaniStream, an implemented system that turns NewsOnAIR articles and All India Radio audio bulletins into a searchable, citable knowledge base and answers questions about them. The pipeline collects content politely and incrementally, transcribes bulletins with sentence-level timestamps, removes cross-bulletin repetition with a cosine novelty gate, labels content with a rule-based genre taxonomy, and answers questions with a two-tier retriever and a citation-checked generator.

On a month of real data — 2,075 articles and 73 bulletins totalling 16.78 hours — the measurements support three findings. A novelty threshold of 0.85 removes about a fifth of bulletin sentences while keeping 97.4% of their vocabulary. Filtering by date before vector search raises the share of correctly dated passages in the top five from 44.7% to 96.7%, which addresses the main risk of answering time-sensitive news questions from a repetitive source. Answers cited their expected sources in every answered evaluation question, at a warm median of about two seconds, almost all of it generation time. The evaluation also exposed a partly covered question that was not refused and an uncited answer, which define the next steps.

Future work follows from these limitations: measuring word error rate on a manually transcribed sample and testing a larger speech model on the GPU; extending collection to Hindi and regional-language bulletins with a multilingual encoder; collecting the eleven sparsely covered months with the existing date-range jobs; tuning the relevance gate and rejecting uncited answers on a larger, independently written question set with human judgements; and evaluating retrieval on paraphrased rather than generated questions.

# References {.unnumbered}
