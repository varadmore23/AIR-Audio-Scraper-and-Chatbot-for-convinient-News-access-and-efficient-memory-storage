"""
latency_bench.py - retrieval latency on the real index (no generation, no query log).

Questions: the 12 questions of eval/newsonair_eval_set.json and 150 date-anchored questions built
the same way as Experiment 2 (eval/phase3_experiments.py). The first query warms the models and is
excluded. Writes research_paper/data/retrieval_latency.json.
"""
import json
import os
import statistics as st
import sys
import time

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
os.chdir(ROOT)
os.environ["VANI_QUERY_LOG"] = "0"

import random  # noqa: E402
import re  # noqa: E402
import datetime as dt  # noqa: E402

import rag_engine  # noqa: E402
import vector_store  # noqa: E402
from phase3_experiments import STOP  # noqa: E402


def main():
    data = json.load(open(os.path.join(ROOT, "eval", "newsonair_eval_set.json"), encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    qs = [c["question"] for c in cases]
    db = vector_store._get_db()
    rows = [dict(r) for r in db.execute(
        f"SELECT text, date FROM segments WHERE {vector_store.ACTIVE} AND content_type='audio_transcript' AND LENGTH(text) > 70")]
    db.close()
    random.Random(7).shuffle(rows)
    for r in rows[:150]:
        d = dt.date.fromisoformat(r["date"])
        topic = " ".join([w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", r["text"]) if w.lower() not in STOP][:7])
        qs.append(f"What did the news on {d.day} {d.strftime('%B %Y')} say about {topic}?")
    rag_engine.retrieve("warm up the embedding model")
    times = []
    for qq in qs:
        t0 = time.perf_counter()
        rag_engine.retrieve(qq)
        times.append((time.perf_counter() - t0) * 1000)
    xs = sorted(times)
    pc = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]  # noqa: E731
    summary = {"n": len(xs), "mean_ms": round(st.mean(xs), 1), "median_ms": round(st.median(xs), 1),
               "p90_ms": round(pc(0.9), 1), "p95_ms": round(pc(0.95), 1), "min_ms": round(xs[0], 1),
               "max_ms": round(xs[-1], 1)}
    json.dump({"summary": summary, "samples_ms": [round(t, 2) for t in times]},
              open(os.path.join(PAPER, "data", "retrieval_latency.json"), "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
