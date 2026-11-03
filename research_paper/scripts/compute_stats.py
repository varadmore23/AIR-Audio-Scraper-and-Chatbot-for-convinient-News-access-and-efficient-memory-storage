"""
compute_stats.py - every number used in the paper, computed from the project's real data.

Reads: news_store.db (SQLite), faiss_index.bin, output/text transcripts, eval/*.json,
research_paper/data/*.log. Writes research_paper/data/paper_stats.json.

    python research_paper/scripts/compute_stats.py
"""
from __future__ import annotations

import datetime as dt
import glob
import importlib.metadata as md
import json
import os
import platform
import re
import sqlite3
import statistics as st
import sys

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("VANI_QUERY_LOG", "0")

import config  # noqa: E402
import nlp_pipeline  # noqa: E402
import taxonomy  # noqa: E402

DB = os.path.join(ROOT, "news_store.db")


def pct(a, b, nd=1):
    return round(100.0 * a / b, nd) if b else None


def quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return None
    def q(p):
        k = (len(xs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": round(q(0.5), 1), "p90": round(q(0.9), 1),
            "p95": round(q(0.95), 1), "min": round(xs[0], 1), "max": round(xs[-1], 1)}


def main():
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = lambda s, *a: db.execute(s, a).fetchall()  # noqa: E731
    S: dict = {"computed_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "timezone": config.TIMEZONE}

    # --- environment -------------------------------------------------------------------------
    vers = {}
    for p in ["faiss-cpu", "sentence-transformers", "faster-whisper", "transformers", "torch", "fastapi", "uvicorn",
              "google-generativeai", "nltk", "numpy", "av", "requests", "beautifulsoup4"]:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    import sqlite3 as _s
    S["environment"] = {"python": platform.python_version(), "os": platform.platform(), "sqlite": _s.sqlite_version,
                        "packages": vers,
                        "config": {"asr_backend": config.ASR_BACKEND, "asr_model": config.ASR_MODEL,
                                   "asr_device": config.ASR_DEVICE, "asr_compute_type": config.ASR_COMPUTE_TYPE,
                                   "asr_language": config.ASR_LANGUAGE, "asr_beam_size": config.ASR_BEAM_SIZE,
                                   "embedding_model": config.EMBEDDING_MODEL, "embedding_dim": config.EMBEDDING_DIM,
                                   "mmr_threshold": config.MMR_THRESHOLD, "rag_top_k": config.RAG_TOP_K,
                                   "rag_candidates": config.RAG_CANDIDATES, "rag_min_dense": config.RAG_MIN_DENSE_SCORE,
                                   "rag_min_top": config.RAG_MIN_TOP_SCORE, "rag_strong_match": config.RAG_STRONG_MATCH,
                                   "rag_context_neighbors": config.RAG_CONTEXT_NEIGHBORS,
                                   "gemini_model": config.GEMINI_MODEL, "gemini_fallbacks": config.GEMINI_FALLBACK_MODELS,
                                   "gemini_max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
                                   "gemini_cooldown_s": config.GEMINI_COOLDOWN_SECONDS,
                                   "newsonair_delay_s": config.NEWSONAIR_DELAY_SECONDS,
                                   "report_window": [config.REPORT_WINDOW_START, config.REPORT_WINDOW_END]}}

    # --- dataset -----------------------------------------------------------------------------
    A = "active = 1"
    S["dataset"] = {
        "articles": q("SELECT COUNT(*) FROM articles")[0][0],
        "broadcasts": q("SELECT COUNT(*) FROM broadcasts")[0][0],
        "audio_files_on_disk": q("SELECT COUNT(*) FROM broadcasts WHERE duration_s IS NOT NULL")[0][0],
        "audio_hours": round(q("SELECT SUM(duration_s) FROM broadcasts")[0][0] / 3600, 2),
        "chunks_active": q(f"SELECT COUNT(*) FROM segments WHERE {A}")[0][0],
        "chunks_article": q(f"SELECT COUNT(*) FROM segments WHERE {A} AND content_type='article'")[0][0],
        "chunks_transcript": q(f"SELECT COUNT(*) FROM segments WHERE {A} AND content_type='audio_transcript'")[0][0],
        "chunks_inactive": q("SELECT COUNT(*) FROM segments WHERE active = 0")[0][0],
        "timed_sentences": q(f"SELECT COUNT(*) FROM segments WHERE {A} AND audio_start IS NOT NULL")[0][0],
        "faiss_vectors": None,
        "languages": q(f"SELECT language, COUNT(*) FROM segments WHERE {A} GROUP BY 1"),
        "state": q("SELECT kind, status, COUNT(*) FROM ingestion_state GROUP BY 1, 2"),
    }
    try:
        import faiss
        S["dataset"]["faiss_vectors"] = faiss.read_index(config.FAISS_INDEX_PATH).ntotal
    except Exception as exc:  # pragma: no cover
        S["dataset"]["faiss_vectors_error"] = str(exc)

    # coverage (actual indexed dates, separately per content type)
    cov = {}
    for kind, cond in (("articles", "content_type='article'"), ("broadcasts", "content_type='audio_transcript'"),
                       ("all", "1=1")):
        r = q(f"SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM segments WHERE {A} AND {cond} "
              "AND date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'")[0]
        cov[kind] = {"earliest": r[0], "latest": r[1], "distinct_dates": r[2]}
    w0, w1 = (dt.date.fromisoformat(x) for x in (config.REPORT_WINDOW_START, config.REPORT_WINDOW_END))
    wdays = (w1 - w0).days + 1
    in_win = q(f"SELECT COUNT(DISTINCT date) FROM segments WHERE {A} AND date BETWEEN ? AND ?",
               config.REPORT_WINDOW_START, config.REPORT_WINDOW_END)[0][0]
    avail = q("SELECT kind, COUNT(*), SUM(available = 0), SUM(available) FROM source_availability GROUP BY kind")
    cov["window_days"] = wdays
    cov["dates_with_records_in_window"] = in_win
    cov["coverage_pct"] = pct(in_win, wdays)
    cov["source_availability_checked"] = avail
    S["coverage"] = cov
    S["article_dates"] = [r[0] for r in q(f"SELECT DISTINCT date FROM segments WHERE {A} AND content_type='article' ORDER BY 1")]
    S["daily"] = q(f"SELECT date, COUNT(DISTINCT CASE WHEN content_type='article' THEN doc_id END), "
                   f"COUNT(DISTINCT CASE WHEN content_type='audio_transcript' THEN doc_id END) FROM segments "
                   f"WHERE {A} AND date >= '2026-10-01' GROUP BY date ORDER BY date")
    S["articles_by_month"] = q("SELECT substr(date,1,7), COUNT(*) FROM articles GROUP BY 1 ORDER BY 1")

    # --- broadcasts --------------------------------------------------------------------------
    S["broadcast_types"] = q("SELECT broadcast_type, COUNT(*), ROUND(SUM(duration_s)/3600.0, 2), "
                             "SUM(segments) FROM broadcasts GROUP BY 1 ORDER BY 2 DESC")
    durs = [r[0] / 60 for r in q("SELECT duration_s FROM broadcasts WHERE duration_s IS NOT NULL")]
    S["broadcast_duration_min"] = quantiles(durs)
    raw_sent = []
    for (sf,) in q("SELECT source_file FROM broadcasts"):
        p = os.path.join(config.OUTPUT_DIR, sf)
        if os.path.isfile(p):
            raw_sent.append(len(nlp_pipeline.segment_sentences(open(p, encoding="utf-8").read())))
    S["raw_sentences_per_bulletin"] = quantiles(raw_sent)
    S["raw_sentences_total"] = sum(raw_sent)
    # MMR repeats per bulletin from the ingestion logs
    pat = re.compile(r"Indexed broadcast (noa-bulletin-\d+) \| (.+?) \(English\) .*?\| (\d+) sentences -> (\d+) new "
                     r"segments \((\d+) repeats removed by MMR\), (\d+) with audio timestamps")
    per = {}
    for f in glob.glob(os.path.join(ROOT, "logs", "vanistream.log*")) + glob.glob(os.path.join(PAPER, "data", "*.log")):
        for line in open(f, encoding="utf-8", errors="replace"):
            m = pat.search(line)
            if m:
                per[m[1]] = {"title": m[2], "sentences": int(m[3]), "stored": int(m[4]), "repeats": int(m[5]),
                             "timed": int(m[6])}
    btype = {r[0]: r[1] for r in q("SELECT doc_id, broadcast_type FROM broadcasts")}
    per = {k: v for k, v in per.items() if k in btype}
    agg = {}
    for k, v in per.items():
        # grouped by the stored broadcast type (from the NewsOnAIR category), not the listing title: some 21:00
        # bulletins are titled "News At Nine" but categorised "Evening News"
        a = agg.setdefault(btype[k], {"bulletins": 0, "sentences": 0, "repeats": 0})
        a["bulletins"] += 1
        a["sentences"] += v["sentences"]
        a["repeats"] += v["repeats"]
    for a in agg.values():
        a["repeat_pct"] = pct(a["repeats"], a["sentences"])
    S["mmr_by_programme"] = agg
    S["mmr_bulletins_logged"] = len(per)
    S["mmr_total"] = {"sentences": sum(v["sentences"] for v in per.values()),
                      "repeats": sum(v["repeats"] for v in per.values()),
                      "stored": sum(v["stored"] for v in per.values())}
    S["mmr_total"]["repeat_pct"] = pct(S["mmr_total"]["repeats"], S["mmr_total"]["sentences"])

    # --- articles and chunking ------------------------------------------------------------------
    import news_articles
    wc, nch, limited = [], [], 0
    sample = None
    rows = q("SELECT article_id, data_json FROM articles WHERE article_id LIKE 'noa-post-%'")
    for aid, dj in rows:
        a = json.loads(dj)
        words = len(news_articles.full_text(a).split())
        ch = news_articles.chunk_article(a)
        wc.append(words)
        nch.append(len(ch))
        limited += bool(a.get("content_limited"))
        if sample is None and len(ch) == 3 and 300 <= words <= 420:
            paras = [len(p.split()) for p in a["body"]]
            sample = {"id": aid, "headline": a["headline"], "words_full_text": words, "body_words": sum(paras),
                      "paragraph_words": paras, "chunks": [len(c["text"].split()) for c in ch],
                      "sections": [c["section"] for c in ch]}
    hist = {}
    for n in nch:
        hist[n] = hist.get(n, 0) + 1
    S["articles_stats"] = {"n": len(rows), "words": quantiles(wc), "chunks_per_article": quantiles(nch),
                           "chunks_hist": sorted(hist.items()), "content_limited": limited,
                           "content_limited_pct": pct(limited, len(rows)), "chunk_words_max": news_articles.CHUNK_WORDS}
    S["sample_article"] = sample
    S["article_chunk_words"] = quantiles([len(r[0].split()) for r in q(
        f"SELECT text FROM segments WHERE {A} AND content_type='article'")])
    S["transcript_chunk_words"] = quantiles([len(r[0].split()) for r in q(
        f"SELECT text FROM segments WHERE {A} AND content_type='audio_transcript'")])

    # --- genres ------------------------------------------------------------------------------
    S["genres_articles"] = [(g, taxonomy.genre_name(g), n) for g, n in q(
        "SELECT IFNULL(genre_id,'other'), COUNT(*) FROM articles GROUP BY 1 ORDER BY 2 DESC")]
    S["genres_transcript_chunks"] = [(g, taxonomy.genre_name(g), n) for g, n in q(
        f"SELECT IFNULL(genre_id,'other'), COUNT(*) FROM segments WHERE {A} AND content_type='audio_transcript' "
        "GROUP BY 1 ORDER BY 2 DESC")]
    S["publisher_categories"] = q("SELECT source_category, COUNT(*) FROM articles GROUP BY 1 ORDER BY 2 DESC")
    rules = {}
    for h, ft, cat, g, aid in q("SELECT headline, full_text, source_category, genre, article_id FROM articles"):
        conf = q("SELECT MAX(genre_confidence) FROM segments WHERE doc_id = ?", aid)[0][0]
        r = taxonomy.classify(h, ft, cat, g, conf, "article")[2]
        rules[r] = rules.get(r, 0) + 1
    S["genre_rules_articles"] = rules
    S["taxonomy_size"] = len(taxonomy.GENRES)

    # --- ingestion jobs ----------------------------------------------------------------------
    jobs = []
    for jid, status, started, finished, cj, pj in q("SELECT job_id, status, started_at, finished_at, counts_json, "
                                                    "params_json FROM ingestion_jobs ORDER BY started_at"):
        jobs.append({"job_id": jid, "status": status, "started": started, "finished": finished,
                     "counts": json.loads(cj or "{}"), "params": json.loads(pj or "{}")})
    S["jobs"] = jobs
    # audio processed by each bulletin job (seconds of indexed audio) -> processing speed
    S["job_audio_seconds"] = {r[0]: round(r[1] or 0, 1) for r in q(
        "SELECT s.job_id, SUM(b.duration_s) FROM ingestion_state s JOIN broadcasts b ON b.doc_id = s.doc_id "
        "WHERE s.kind = 'broadcast' GROUP BY s.job_id")}
    bjobs = [j for j in jobs if j["counts"].get("broadcasts_discovered")]
    S["bulletin_attempts"] = {"attempted": sum(j["counts"].get("broadcasts_discovered", 0) - j["counts"].get(
        "already_indexed", 0) for j in bjobs), "indexed": sum(j["counts"].get("broadcasts_indexed", 0) for j in bjobs),
        "failed": sum(j["counts"].get("failed", 0) for j in bjobs),
        "transcribed": sum(j["counts"].get("transcriptions_ok", 0) for j in bjobs),
        "audio_processed": sum(j["counts"].get("audio_processed", 0) for j in bjobs)}
    ajobs = [j for j in jobs if j["counts"].get("articles_discovered")]
    S["article_attempts"] = {"discovered": sum(j["counts"]["articles_discovered"] for j in ajobs),
                             "new": sum(j["counts"].get("articles_indexed", 0) - j["counts"].get("updated", 0)
                                        for j in ajobs),
                             "updated": sum(j["counts"].get("updated", 0) for j in ajobs),
                             # duplicates_skipped also counts a skipped bulletin in mixed jobs: use discovered - indexed
                             "duplicates": sum(j["counts"]["articles_discovered"] - j["counts"].get("articles_indexed", 0)
                                               for j in ajobs),
                             "failed": sum(j["counts"].get("failed", 0) for j in ajobs if not j["counts"].get(
                                 "broadcasts_discovered"))}

    # --- latency -----------------------------------------------------------------------------
    S["query_log"] = q("SELECT ts, genre_scope, question_type, mode, model, n_chunks, retrieval_ms, generation_ms, "
                       "total_ms, cited_count, insufficient FROM query_log ORDER BY ts")
    e2e = []
    lp = os.path.join(PAPER, "data", "eval_generative_run.log")
    if os.path.isfile(lp):
        for line in open(lp, encoding="utf-8", errors="replace"):
            m = re.search(r"Answer generated: mode=(\w+), model=([\w.\-]+), type=(\w+), (\d+) words, \d+ chars, ([\d.]+)s",
                          line)
            if m:
                e2e.append({"mode": m[1], "model": m[2], "type": m[3], "words": int(m[4]), "generation_s": float(m[5])})
        tot = [float(m) for m in re.findall(r"\| total ([\d.]+)s", open(lp, encoding="utf-8", errors="replace").read())]
        for i, t in enumerate(tot):
            if i < len(e2e):
                e2e[i]["total_s"] = t
    S["eval_generative_timings"] = e2e
    tot = [t["total_s"] for t in e2e if "total_s" in t]
    gen_rows = [r for r in S["query_log"] if r[3] == "generative"]
    S["latency_summary"] = {
        "eval_total_s_all": quantiles(tot), "eval_total_s_warm": quantiles(tot[1:]),
        "eval_generation_s_warm": quantiles([t["generation_s"] for t in e2e[1:]]),
        "querylog_generative_total_ms": quantiles([r[8] for r in gen_rows]),
        "querylog_generative_total_ms_warm": quantiles([r[8] for r in gen_rows if r[6] < 10000]),
        "querylog_retrieval_ms_warm": quantiles([r[6] for r in S["query_log"] if r[6] < 10000]),
        "querylog_rows": len(S["query_log"]), "querylog_generative": len(gen_rows),
        "querylog_generative_with_citations": sum(1 for r in gen_rows if r[9] > 0),
        "querylog_insufficient": sum(1 for r in S["query_log"] if r[10]),
    }
    lb = os.path.join(PAPER, "data", "retrieval_latency.json")
    if os.path.isfile(lb):
        S["retrieval_latency"] = json.load(open(lb, encoding="utf-8"))["summary"]

    # --- experiments + evaluation --------------------------------------------------------------
    S["phase3_experiments"] = json.load(open(os.path.join(ROOT, "eval", "phase3_results.json"), encoding="utf-8"))
    ev = {}
    for name, f in (("gen_2nov", "newsonair_report.json"), ("ext_2nov", "newsonair_report_extractive.json"),
                    ("gen_3nov", "newsonair_report_p3.json"), ("ext_3nov", "newsonair_report_extractive_p3.json")):
        p = os.path.join(ROOT, "eval", f)
        if os.path.isfile(p):
            d = json.load(open(p, encoding="utf-8"))
            ev[name] = {"summary": d["summary"],
                        "rows": [{k: r.get(k) for k in ("question", "answerable", "insufficient", "mode", "rank",
                                                         "citations_valid", "citation_correct", "groundedness",
                                                         "question_type", "words")} for r in d["rows"]]}
    S["rag_eval"] = ev
    S["perf"] = {k: json.load(open(os.path.join(ROOT, "eval", f"perf_{k}.json"), encoding="utf-8"))
                 for k in ("before", "after")}
    out = os.path.join(PAPER, "data", "paper_stats.json")
    json.dump(S, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
