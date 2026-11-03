# VaniStream research paper — provenance report

This report states where each value in the paper comes from, what was not measured, and what the authors still need to fill in.

## Deliverables (this folder)

| File | What it is |
|---|---|
| `VaniStream_Paper.docx` | The paper, editable in Word. Equations are native Word equations; figures are 300-dpi images. |
| `VaniStream_Paper.pdf` | PDF exported by Microsoft Word from the DOCX (25 pages). |
| `paper.md` | Paper source: Markdown with LaTeX equations. |
| `figures/fig01…fig11.png` | Figures 1–11, 300 dpi. Each chart has a `.csv` file next to it with the plotted values. |
| `data/paper_stats.json` | Every computed value used in the paper. |
| `data/retrieval_latency.json` | 162 retrieval latency measurements. |
| `data/*.log` | Logs of the ingestion jobs and the generative evaluation run used for per-bulletin and latency values. |
| `scripts/compute_stats.py` | Computes `paper_stats.json` from the database, transcripts, logs and evaluation reports. |
| `scripts/latency_bench.py` | Retrieval latency benchmark. |
| `scripts/make_figures.py` | Draws all figures from `paper_stats.json`. |
| `scripts/build_paper.py` | `paper.md` → DOCX (pandoc, via `pypandoc_binary`) → PDF (Word). |
| `scripts/verify_paper.py` | Checks 45 quoted numbers against `paper_stats.json`, figure/table/equation references and numbering, the author list, and that References is empty. |
| `../eval/phase3_experiments.py` | Experiments 1 and 2. Results are in `../eval/phase3_results.json`. |

Regenerate everything:

```
python eval/phase3_experiments.py
python research_paper/scripts/latency_bench.py
python research_paper/scripts/compute_stats.py
python research_paper/scripts/make_figures.py
python research_paper/scripts/build_paper.py
python research_paper/scripts/verify_paper.py
```

## Values taken from real data

All values were computed on 3 November 2026 from the project's own data: no value was estimated or typed in by hand.

| Paper content | Source |
|---|---|
| **Dataset counts:** 2,075 articles, 73 bulletins, 16.78 h, 8,854 active chunks, 8,858 vectors, 5,388 timed sentences, 75 short articles | `news_store.db` (`articles`, `broadcasts`, `segments`), `faiss_index.bin` |
| **Coverage:** broadcasts 2026-10-01 → 2026-11-02 on 33 dates; articles 2025-11-02 → 2026-11-02 on 42 dates; 11.5% of the 366-day window | active `segments` rows with valid dates; window from `config.py` |
| Daily volume (Fig. 3) and articles by month | `segments`, `articles` |
| **Programmes:** bulletins and hours by programme (Table 3, Fig. 4a) | `broadcasts.broadcast_type`, from the NewsOnAIR category |
| 6,770 transcribed sentences; 1,381 removed (20.4%); removal share by programme (Fig. 4b) | ingestion log lines "N sentences -> M new segments (R repeats removed)", joined with `broadcasts` |
| **Genres:** distribution (Fig. 5) and which classification rule decided | `articles.genre_id`, `segments.genre_id`; rule recomputed with `taxonomy.classify` |
| **Chunking:** sample article, chunk counts and lengths (Fig. 6) | `articles.data_json` re-chunked with `news_articles.chunk_article`; `segments.text` |
| **Ingestion outcomes:** 2,169 / 2,078 / 91 / 3 for articles; 83 / 73 / 9 / 1 for bulletins; job durations and audio seconds (Fig. 7) | `ingestion_jobs.counts_json`, `ingestion_state`, `broadcasts.duration_s` |
| **Experiment 1:** λ sweep (Fig. 8, Table 7) and consecutive-bulletin repetition | `eval/phase3_experiments.py` over the 73 transcripts on disk |
| **Experiment 2:** two-tier vs whole-index retrieval (Fig. 9, Table 8) | the same script; 150 questions, fixed seed 7 |
| **Answer evaluation** (Fig. 10, Table 9) | `eval/newsonair_report.json` (Gemini, 2 Nov), `eval/newsonair_report_p3.json` (Gemini, 3 Nov), `eval/newsonair_report_extractive_p3.json` (extractive, 3 Nov) |
| **Retrieval latency:** median 32.5 ms, p95 54.1 ms, n = 162 (Fig. 11a) | `scripts/latency_bench.py` |
| **End-to-end latency:** warm median 2.1 s, n = 10; cold 192.7 s (Fig. 11b) | generative evaluation log (`data/eval_generative_run.log`) |
| Query-log latencies; citations present in 7 of 8 logged generative answers | `query_log` table: 11 real requests to the running application |
| Environment (Table 5) | installed package versions, `platform`; CPU, RAM and GPU from Windows WMI |
| Parameters (Table 6) and method details | `config.py`, `news_articles.py`, `nlp_pipeline.py`, `transcriber.py`, `taxonomy.py`, `vector_store.py`, `rag_engine.py` |

## Not measured (stated as such in the paper)

- **Word error rate and character error rate** of the transcripts. No reference transcripts exist. Measuring them requires manually transcribing a sample of bulletins.
- **Blind or human evaluation of answer quality.** The 12-question set was written by the authors with knowledge of the corpus. Groundedness is a lexical proxy.
- **Retrieval on paraphrased questions.** Experiment 2 builds its questions from the indexed sentences, so its results are an upper bound.
- **Recall@k and MAP.** The evaluation set has one expected document per question, so only hit rate and MRR are reported.
- **Statistical significance and confidence intervals.** These were not computed; the samples are small.
- **Live-stream ingestion on a real broadcast.** The public AIR stream was offline on 3 Nov 2026. Live ingestion was verified only on recorded audio and is mentioned only as a limitation.
- **Effect of the ffmpeg pre-filters on ASR.** The machine had no ffmpeg binary, so the filters were not applied.

## Placeholders the authors must complete

- **Affiliation and corresponding author:** two placeholders on the title page. The repository contains no verified affiliation. Your Phase III presentation names Sardar Patel Institute of Technology, but that was not used without confirmation.
- **Author names:** written without titles (Varad More, Manav Jadhav, Soham Jadhav, Anant Nimkar), because the request used both "Dr." and "Prof." for Anant Nimkar. Confirm the author order.
- **Related work citations:** 18 distinct `[CITATION PLACEHOLDER: …]` markers, mostly in Related Work and two in the Introduction. They describe what to cite and contain no invented titles, authors, years or DOIs.
- **References section:** present and left empty, as requested.

## Notes for interpreting the results

- **Answering model:** `gemini-flash-lite-latest`. The configured first model, `gemini-flash-latest`, had exhausted its daily quota.
- **Evaluation runs on different corpora:** the 2 Nov run used 81 articles and 8 bulletins; the 3 Nov runs used the full dataset. The MRR drop and the refusal error on the FIFA question are reported as observed.
- **Programme labels:** bulletins are typed by the NewsOnAIR category. Some 21:00 bulletins are titled "News At Nine" but categorised "Evening News", and they count as Evening News.
- **Query log:** the 11 logged requests were made while testing and demonstrating the application. They are real requests, but not a user study.
