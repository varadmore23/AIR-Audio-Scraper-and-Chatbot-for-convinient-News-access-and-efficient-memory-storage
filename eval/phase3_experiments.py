"""
phase3_experiments.py - the two Phase III experiments, run on the real indexed NewsOnAIR data.

E1  MMR novelty threshold sweep (lambda in 0.50..0.95) over the raw sentences of every transcribed
    bulletin, in broadcast order: storage retained vs vocabulary retained. Also measures how much
    consecutive bulletins actually repeat each other.
E2  Two-tier (SQL date prefilter + hybrid search) vs naive (hybrid search over the whole index)
    on date-anchored questions built from indexed bulletin sentences.

    python eval/phase3_experiments.py            -> eval/phase3_results.json
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("VANI_QUERY_LOG", "0")  # experiment queries must not enter the analytics log

import numpy as np  # noqa: E402

import config  # noqa: E402
import nlp_pipeline  # noqa: E402
import rag_engine  # noqa: E402
import vector_store  # noqa: E402

STOP = set(vector_store.STOPWORDS) | set("""also said says will would could should has have had been being from that this with
for are was were its it's their they them there here about into over after before under more most other such than then
these those which while what when where who whom whose why how all any each few some very just only also into""".split())
LAMBDAS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-']+", text.lower()) if len(w) >= 3 and w not in STOP}


def load_bulletins() -> list[dict]:
    db = vector_store._get_db()
    try:
        rows = [dict(r) for r in db.execute("SELECT source_file, title, broadcast_type, date, time FROM broadcasts "
                                            "WHERE ingest_mode = 'archive' ORDER BY date, time")]
    finally:
        db.close()
    out = []
    for r in rows:
        path = os.path.join(config.OUTPUT_DIR, r["source_file"])
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            sents = [s for s in nlp_pipeline.segment_sentences(fh.read()) if len(s.split()) >= 4]
        out.append(dict(r, sentences=sents))
    return out


def e1_sweep(bulletins: list[dict]) -> dict:
    sents, owner = [], []
    for bi, b in enumerate(bulletins):
        for s in b["sentences"]:
            sents.append(s)
            owner.append(bi)
    t0 = time.time()
    emb = np.asarray(nlp_pipeline.embed_sentences(sents), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    t_embed = time.time() - t0
    vocab_all = set().union(*(words(s) for s in sents))
    # max similarity of each sentence to any *earlier* sentence (in broadcast order) - computed once
    n = len(sents)
    sweep = []
    for lam in LAMBDAS:
        admitted = np.zeros(n, dtype=bool)
        kept_idx: list[int] = []
        for i in range(n):
            if kept_idx:
                sim = float(np.max(emb[kept_idx] @ emb[i]))
                if sim >= lam:
                    continue
            admitted[i] = True
            kept_idx.append(i)
        vocab_kept = set().union(*(words(sents[i]) for i in kept_idx)) if kept_idx else set()
        sweep.append({"lambda": lam, "stored": int(admitted.sum()),
                      "storage_retained_pct": round(100 * admitted.sum() / n, 1),
                      "compression_pct": round(100 * (1 - admitted.sum() / n), 1),
                      "vocab_retained_pct": round(100 * len(vocab_kept) / len(vocab_all), 1)})
    # how much does each bulletin repeat the one before it (same day, consecutive airing)?
    pairs = []
    by_bulletin = {}
    for i, bi in enumerate(owner):
        by_bulletin.setdefault(bi, []).append(i)
    for bi in range(1, len(bulletins)):
        a, b = bulletins[bi - 1], bulletins[bi]
        if a["date"] != b["date"] or bi not in by_bulletin or bi - 1 not in by_bulletin:
            continue
        ea, eb = emb[by_bulletin[bi - 1]], emb[by_bulletin[bi]]
        rep = float(np.mean((eb @ ea.T).max(axis=1) >= 0.85))
        pairs.append({"date": b["date"], "from": f"{a['title']} {a['time']}", "to": f"{b['title']} {b['time']}",
                      "repeated_pct": round(100 * rep, 1),
                      "kind": "hourly->hourly" if a["broadcast_type"] == b["broadcast_type"] == "hourly" else "other"})
    hourly = [p["repeated_pct"] for p in pairs if p["kind"] == "hourly->hourly"]
    other = [p["repeated_pct"] for p in pairs if p["kind"] == "other"]
    return {"bulletins": len(bulletins), "sentences": n, "vocabulary": len(vocab_all), "embed_seconds": round(t_embed, 1),
            "sweep": sweep, "consecutive_pairs": pairs,
            "hourly_repeat_mean_pct": round(float(np.mean(hourly)), 1) if hourly else None,
            "hourly_pairs": len(hourly),
            "other_consecutive_repeat_mean_pct": round(float(np.mean(other)), 1) if other else None,
            "other_pairs": len(other)}


def e2_two_tier(n_samples: int = 150, seed: int = 7) -> dict:
    db = vector_store._get_db()
    try:
        rows = [dict(r) for r in db.execute(
            f"SELECT faiss_id, text, date FROM segments WHERE {vector_store.ACTIVE} AND content_type = 'audio_transcript' "
            "AND LENGTH(text) > 70")]
    finally:
        db.close()
    random.Random(seed).shuffle(rows)
    rows = rows[:n_samples]
    res = {"two_tier": {"date_p5": [], "date_top1": [], "hit5": [], "candidates": []},
           "naive": {"date_p5": [], "date_top1": [], "hit5": [], "candidates": []}}
    examples = []
    for r in rows:
        d = dt.date.fromisoformat(r["date"])
        topic = " ".join([w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", r["text"]) if w.lower() not in STOP][:7])
        q = f"What did the news on {d.day} {d.strftime('%B %Y')} say about {topic}?"
        f = rag_engine.parse_query(q)
        if f.date_from != r["date"]:
            continue  # the date parser must resolve the question's date, or the comparison is meaningless
        for mode in ("two_tier", "naive"):
            ff = rag_engine.parse_query(q)
            if mode == "naive":
                ff.date_from = ff.date_to = None
                ids = None
            else:
                ids = vector_store.metadata_prefilter(ff.date_from, ff.date_to, ff.genre)
            hits = rag_engine._limit_groups(rag_engine._drop_near_duplicates(rag_engine._hybrid_search(q, ids, ff)), 5)[:5]
            dates = [h.get("date") for h in hits]
            res[mode]["date_p5"].append(sum(x == r["date"] for x in dates) / max(1, len(dates)))
            res[mode]["date_top1"].append(1.0 if dates and dates[0] == r["date"] else 0.0)
            res[mode]["hit5"].append(1.0 if any(h["faiss_id"] == r["faiss_id"] for h in hits) else 0.0)
            res[mode]["candidates"].append(len(ids) if ids is not None else vector_store._load_index().ntotal)
        if len(examples) < 3:
            examples.append(q)
    summary = {m: {"date_precision_at5": round(float(np.mean(v["date_p5"])) * 100, 1),
                   "date_correct_top1": round(float(np.mean(v["date_top1"])) * 100, 1),
                   "exact_sentence_hit_at5": round(float(np.mean(v["hit5"])) * 100, 1),
                   "mean_candidates": round(float(np.mean(v["candidates"])), 0)} for m, v in res.items()}
    return {"questions": len(res["naive"]["hit5"]), "summary": summary, "example_questions": examples}


def main() -> int:
    t0 = time.time()
    bulletins = load_bulletins()
    out = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "e1_mmr_sweep": e1_sweep(bulletins), "e2_two_tier": e2_two_tier()}
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(ROOT, "eval", "phase3_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    e1, e2 = out["e1_mmr_sweep"], out["e2_two_tier"]
    print(f"E1: {e1['bulletins']} bulletins, {e1['sentences']} sentences, vocab {e1['vocabulary']}")
    for s in e1["sweep"]:
        print(f"   lambda={s['lambda']:.2f} stored={s['storage_retained_pct']}% vocab={s['vocab_retained_pct']}%")
    print(f"   consecutive hourly repeat: {e1['hourly_repeat_mean_pct']}% over {e1['hourly_pairs']} pairs; "
          f"other consecutive: {e1['other_consecutive_repeat_mean_pct']}% over {e1['other_pairs']} pairs")
    print(f"E2: {e2['questions']} questions -> {json.dumps(e2['summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
