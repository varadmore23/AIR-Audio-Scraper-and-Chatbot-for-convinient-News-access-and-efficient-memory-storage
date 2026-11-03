"""
eval_rag.py - Lightweight, offline evaluation of VaniStream retrieval and answers.

For each hand-written case in eval/rag_eval_set.json it runs the real
answer_query() against the current corpus (news_store.db + faiss_index.bin)
and measures:

  hit@k                 expected source appears among the returned sources
  MRR                   1 / rank of the first expected source
  citation validity     every [S#] in the answer refers to a returned source
  citation correctness  at least one cited source is an expected source
  groundedness          share of cited answer sentences whose content words are
                        mostly (>= 60%) found in the cited source passages
  refusal accuracy      unanswerable questions are reported as insufficient,
                        answerable ones are not

Optional per-case checks for detailed answers (eval/rag_reasoning_set.json):
  min_words / max_words  answer length fits the question (citations not counted)
  require_all_expected   every expected source is retrieved AND cited
                         (comparison and overview questions)
  must_mention           phrases a correct grounded answer must contain
The answer text is saved in the report so it can be reviewed by hand.

Checks for the real NewsOnAIR set (eval/newsonair_eval_set.json):
  expected_docs        doc ids (prefix match) instead of transcript files
  genre                passed as the UI genre scope; every source must have it
  expected_category    every source comes from that NewsOnAIR category
  require_audio        a cited transcript source carries an audio time range
  check_source_urls    every source links to newsonair.gov.in

No paid evaluation API is used. Gemini is used only if GEMINI_API_KEY is set;
otherwise the extractive answer path is evaluated.

Usage:
  python eval_rag.py                       # uses eval/rag_eval_set.json
  python eval_rag.py --set my_cases.json --out eval/report.json
  python eval_rag.py --set eval/rag_reasoning_set.json --out eval/rag_reasoning_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import config
from tracing import trace

STOP = set("the a an of in on and to for is was were at by with has have had it its this that from as be are "
           "inference".split())


def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/")


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def groundedness(answer: str, sources: list[dict]) -> tuple[float | None, int]:
    """Fraction of cited sentences whose content words are >=60% covered by their cited passages."""
    by_id = {s["id"]: s for s in sources}
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if re.search(r"\[S\d+\]", s)]
    if not sentences:
        return None, 0
    ok = 0
    for sent in sentences:
        ids = re.findall(r"\[(S\d+)\]", sent)
        passage = " ".join(by_id[i]["context"] for i in ids if i in by_id)
        words = _content_words(re.sub(r"\[S\d+\]|\(\d{4}-\d{2}-\d{2}[^)]*\)", " ", sent))
        if not words:
            ok += 1
            continue
        if len(words & _content_words(passage)) / len(words) >= 0.6:
            ok += 1
    return ok / len(sentences), len(sentences)


def _in_genre(source: dict, genre: str) -> bool:
    """Source matches a genre scope: taxonomy primary genre or secondary tag (or the legacy label)."""
    import taxonomy
    import vector_store
    gid = taxonomy.genre_id_of(genre)
    if not gid:
        return source.get("genre") == genre
    if source.get("genre_id") == gid:
        return True
    row = vector_store.get_segments([source["faiss_id"]]).get(source["faiss_id"], {})
    return gid in taxonomy.decode_tags(row.get("genre_tags"))


def evaluate(cases: list[dict], k: int | None = None) -> dict:
    import rag_engine

    rows = []
    for case in cases:
        t0 = time.perf_counter()
        api_error = None
        with trace("EVAL"):
            try:
                res = rag_engine.answer_query(case["question"], genre=case.get("genre"))
            except Exception as exc:  # an API or pipeline failure is a result, not a crash
                api_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                res = {"answer": "", "sources": [], "insufficient_context": False, "mode": "error"}
        elapsed = round(time.perf_counter() - t0, 3)
        returned = [_norm_path(s["source_file"]) for s in res["sources"]][: k or None]
        expected = [_norm_path(e) for e in case.get("expected_sources", [])]
        rank = next((i + 1 for i, f in enumerate(returned) if any(f.endswith(e) for e in expected)), None)
        exp_docs = case.get("expected_docs") or []
        if exp_docs:
            docs = [s.get("doc_id") or "" for s in res["sources"]][: k or None]
            rank = next((i + 1 for i, d in enumerate(docs) if any(d.startswith(e) for e in exp_docs)), None)
        cited = re.findall(r"\[(S\d+)\]", res["answer"])
        valid_ids = {s["id"] for s in res["sources"]}
        cited_files = [_norm_path(s["source_file"]) for s in res["sources"] if s["id"] in cited]
        g, n_g = groundedness(res["answer"], res["sources"])
        words = len(re.sub(r"\[S\d+\]", "", res["answer"]).split())
        cited_expected = {e for e in expected if any(f.endswith(e) for f in cited_files)}
        cited_srcs = [s for s in res["sources"] if s["id"] in cited]
        checks = {}
        if "min_words" in case:
            checks["min_words"] = words >= case["min_words"]
        if "max_words" in case:
            checks["max_words"] = words <= case["max_words"]
        if case.get("require_all_expected"):
            checks["all_expected_cited"] = len(cited_expected) == len(expected)
        if case.get("must_mention"):
            checks["must_mention"] = all(m.lower() in res["answer"].lower() for m in case["must_mention"])
        if exp_docs and case.get("answerable", True):
            checks["expected_doc_cited"] = any(any((s.get("doc_id") or "").startswith(e) for e in exp_docs)
                                               for s in cited_srcs)
        if case.get("genre"):
            checks["genre_filter"] = bool(res["sources"]) and all(_in_genre(s, case["genre"]) for s in res["sources"])
        if case.get("expected_date"):
            checks["date_filter"] = bool(res["sources"]) and all(s.get("date") == case["expected_date"]
                                                                 for s in res["sources"])
        if case.get("expected_category"):
            checks["category_filter"] = bool(res["sources"]) and all(
                s.get("source_category") == case["expected_category"] for s in res["sources"])
        if case.get("require_audio"):
            checks["audio_timestamps"] = any(s.get("kind") == "broadcast" and s.get("audio_start") is not None
                                             and s.get("audio_end") is not None for s in cited_srcs)
        if case.get("check_source_urls") and res["sources"]:
            checks["source_urls"] = all((s.get("source_url") or "").startswith("https://newsonair.gov.in/")
                                        for s in res["sources"])
        rows.append({
            "question": case["question"],
            "answerable": case.get("answerable", True),
            "insufficient": res["insufficient_context"],
            "mode": res["mode"],
            "rank": rank,
            "n_sources": len(returned),
            "citations_valid": all(c in valid_ids for c in cited),
            "citation_correct": (any(any(f.endswith(e) for e in expected) for f in cited_files) if expected else
                                 (checks.get("expected_doc_cited") if exp_docs else None)),
            "groundedness": g,
            "grounded_sentences": n_g,
            "question_type": res.get("question_type"),
            "words": words,
            "checks": checks,
            "answer": res["answer"],
            "total_s": elapsed,
            "model": res.get("model"),
            "api_error": api_error,
            "timings": res.get("timings"),
        })

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    grounded = [r["groundedness"] for r in rows if r["groundedness"] is not None]
    cc = [r["citation_correct"] for r in ans if r["citation_correct"] is not None and not r["insufficient"]]
    summary = {
        "cases": len(rows),
        "hit_rate": round(sum(r["rank"] is not None for r in ans) / len(ans), 3) if ans else None,
        "mrr": round(sum(1 / r["rank"] for r in ans if r["rank"]) / len(ans), 3) if ans else None,
        "citation_validity": round(sum(r["citations_valid"] for r in rows) / len(rows), 3) if rows else None,
        "citation_correctness": round(sum(cc) / len(cc), 3) if cc else None,
        "groundedness": round(sum(grounded) / len(grounded), 3) if grounded else None,
        "refusal_accuracy_unanswerable": round(sum(r["insufficient"] for r in una) / len(una), 3) if una else None,
        "false_refusals_answerable": sum(r["insufficient"] for r in ans),
        "modes": sorted({r["mode"] for r in rows}),
    }
    times = sorted(r["total_s"] for r in rows if r["mode"] != "error")
    if times:
        summary["latency_median_s"] = times[len(times) // 2] if len(times) % 2 else round(
            (times[len(times) // 2 - 1] + times[len(times) // 2]) / 2, 3)
        summary["latency_max_s"] = times[-1]
    summary["api_failures"] = sum(1 for r in rows if r["api_error"])
    summary["generative_answers"] = sum(1 for r in rows if r["mode"] == "generative")
    summary["uncited_answers"] = sum(1 for r in rows if r["mode"] in ("generative", "extractive")
                                     and not re.search(r"\[S\d+\]", r["answer"]))
    checked = [r for r in rows if r["checks"]]
    if checked:
        summary["detail_checks_passed"] = f"{sum(all(r['checks'].values()) for r in checked)}/{len(checked)}"
        long_rows = [r["words"] for r in rows if r["question_type"] in ("explain", "compare", "overview", "gap")
                     and not r["insufficient"]]
        if long_rows:
            summary["avg_words_explanatory"] = round(sum(long_rows) / len(long_rows))
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VaniStream retrieval + grounded answers.")
    parser.add_argument("--set", default=os.path.join("eval", "rag_eval_set.json"))
    parser.add_argument("--out", default=os.path.join("eval", "rag_eval_report.json"))
    parser.add_argument("--k", type=int, default=None, help="Only count the first k sources")
    args = parser.parse_args()
    with open(args.set, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("date_anchor"):
        config.DATE_ANCHOR = spec["date_anchor"]  # sample corpus is historical

    report = evaluate(spec["cases"], args.k)
    print(f"\n{'question':<62} {'rank':>4} {'cite':>5} {'grnd':>5} {'words':>5}  result")
    for r in report["rows"]:
        if r["answerable"]:
            verdict = "REFUSED (wrong)" if r["insufficient"] else ("ok" if r["rank"] else "MISS")
        else:
            verdict = "refused (correct)" if r["insufficient"] else "ANSWERED (should refuse)"
        g = f"{r['groundedness']:.2f}" if r["groundedness"] is not None else "-"
        c = {True: "yes", False: "no", None: "-"}[r["citation_correct"]]
        failed = [k for k, ok in r["checks"].items() if not ok]
        if failed:
            verdict += " | failed: " + ", ".join(failed)
        print(f"{r['question'][:62]:<62} {r['rank'] or '-':>4} {c:>5} {g:>5} {r['words']:>5}  {verdict}")
    print("\nSummary:")
    for key, val in report["summary"].items():
        print(f"  {key:<32} {val}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
