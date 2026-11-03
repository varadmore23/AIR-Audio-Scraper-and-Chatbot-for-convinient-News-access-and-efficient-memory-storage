"""
perf_bench.py - measure response time and payload size of the main read endpoints and RAG retrieval.

Runs against a live server (HTTP) plus in-process retrieval on the real index, and writes
eval/perf_<label>.json so before/after runs can be compared.

  python eval/perf_bench.py --label before
  python eval/perf_bench.py --label after --compare before
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (label, path) pairs; a path that 404s on an older build is recorded as unavailable.
ENDPOINTS = [
    ("articles (first page)", "/articles"),
    ("articles genre=Sports", "/articles?genre=Sports"),
    ("articles search 'india'", "/articles?q=india"),
    ("broadcast list", "/sources"),
    ("broadcasts day-wise", "/broadcasts?mode=day"),
    ("broadcasts Midday", "/broadcasts?type=midday"),
    ("corpus stats", "/stats"),
    ("ingestion status", "/ingest/status"),
    ("coverage", "/coverage"),
    ("analytics summary", "/analytics/summary"),
    ("analytics timeseries", "/analytics/timeseries?interval=day"),
]
QUERIES = ["What did the Prime Minister say about renewable energy?",
           "Summarize the latest sports news",
           "What happened in the economy in October 2026?",
           "cyclone warning coastal districts",
           "India foreign policy developments"]


def _get(base: str, path: str) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(base + path, timeout=120) as r:
            body = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        body, code = b"", e.code
    return time.perf_counter() - t0, len(body), code


def bench_http(base: str, repeat: int) -> dict:
    out = {}
    for label, path in ENDPOINTS:
        times, size, code = [], 0, 0
        for _ in range(repeat):
            t, size, code = _get(base, path)
            times.append(t)
        out[label] = ({"path": path, "median_ms": round(statistics.median(times) * 1000, 1), "bytes": size}
                      if code == 200 else {"path": path, "unavailable": code})
    return out


def bench_retrieval() -> dict:
    import rag_engine
    rag_engine.retrieve(QUERIES[0])  # warm up models and the index
    out = {}
    for q in QUERIES:
        t0 = time.perf_counter()
        r = rag_engine.retrieve(q)
        out[q] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "hits": len(r["hits"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--compare", help="label of an earlier run to compare against")
    ap.add_argument("--no-retrieval", action="store_true")
    a = ap.parse_args()
    res = {"label": a.label, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "http": bench_http(a.base, a.repeat)}
    if not a.no_retrieval:
        res["retrieval"] = bench_retrieval()
    path = os.path.join(ROOT, "eval", f"perf_{a.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    old = None
    if a.compare:
        with open(os.path.join(ROOT, "eval", f"perf_{a.compare}.json"), encoding="utf-8") as f:
            old = json.load(f)
    print(f"{'endpoint':28} {'median ms':>10} {'bytes':>10}" + (f" {'was ms':>9} {'was bytes':>10}" if old else ""))
    for k, v in res["http"].items():
        if "unavailable" in v:
            line = f"{k:28} {'HTTP ' + str(v['unavailable']):>10} {'':>10}"
        else:
            line = f"{k:28} {v['median_ms']:>10} {v['bytes']:>10}"
        if old:
            o = old["http"].get(k, {})
            line += (f" {o['median_ms']:>9} {o['bytes']:>10}" if "median_ms" in o else f" {'n/a':>9} {'':>10}")
        print(line)
    for q, v in res.get("retrieval", {}).items():
        o = (old or {}).get("retrieval", {}).get(q)
        print(f"retrieve {q[:40]:40} {v['ms']:>8} ms  {v['hits']} hits" + (f"  (was {o['ms']} ms)" if o else ""))
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
