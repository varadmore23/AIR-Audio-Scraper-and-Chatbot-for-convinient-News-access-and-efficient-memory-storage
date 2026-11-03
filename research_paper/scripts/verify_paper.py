"""
verify_paper.py - checks that the numbers quoted in paper.md match research_paper/data/paper_stats.json,
that every figure/table/equation referenced exists, and that the References section is empty.

    python research_paper/scripts/verify_paper.py
"""
import json
import os
import re
import sys

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(PAPER, "data", "paper_stats.json"), encoding="utf-8"))
md = open(os.path.join(PAPER, "paper.md"), encoding="utf-8").read()
fails = []


def fmt(v, nd=None):
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.{nd}f}" if nd is not None else str(v)


def need(label, value, text=None):
    t = text or value
    if t not in md and t.replace("%", r"\%") not in md:  # numbers inside LaTeX write \%
        fails.append(f"{label}: '{t}' not found in paper.md")


D, C = S["dataset"], S["coverage"]
e1, e2 = S["phase3_experiments"]["e1_mmr_sweep"], S["phase3_experiments"]["e2_two_tier"]["summary"]
k85 = next(x for x in e1["sweep"] if x["lambda"] == 0.85)
checks = [
    ("articles", fmt(D["articles"])), ("broadcasts", f"{D['broadcasts']} bulletins"), ("audio hours", f"{D['audio_hours']} h"),
    ("active chunks", fmt(D["chunks_active"])), ("article chunks", fmt(D["chunks_article"])),
    ("transcript chunks", fmt(D["chunks_transcript"])), ("faiss vectors", fmt(D["faiss_vectors"])),
    ("timed", fmt(D["timed_sentences"])), ("raw sentences", fmt(S["raw_sentences_total"])),
    ("mmr removed", fmt(S["mmr_total"]["repeats"])), ("mmr pct", f"{S['mmr_total']['repeat_pct']}%"),
    ("broadcast earliest", C["broadcasts"]["earliest"]), ("broadcast latest", C["broadcasts"]["latest"]),
    ("broadcast dates", f"{C['broadcasts']['distinct_dates']} distinct dates"),
    ("article dates", f"{C['articles']['distinct_dates']} distinct dates"), ("coverage pct", f"{C['coverage_pct']}%"),
    ("e1 sentences", fmt(e1["sentences"])), ("e1 vocab", fmt(e1["vocabulary"])),
    ("e1 stored@0.85", f"{k85['storage_retained_pct']}%"), ("e1 vocab@0.85", f"{k85['vocab_retained_pct']}%"),
    ("e2 two-tier p5", f"{e2['two_tier']['date_precision_at5']}%"), ("e2 naive p5", f"{e2['naive']['date_precision_at5']}%"),
    ("e2 naive top1", f"{e2['naive']['date_correct_top1']}%"), ("e2 candidates", str(int(e2['two_tier']['mean_candidates']))),
    ("hourly repeat", f"{S['phase3_experiments']['e1_mmr_sweep']['hourly_repeat_mean_pct']}%"),
    ("lat median", f"{S['retrieval_latency']['median_ms']} ms"), ("lat p95", f"{S['retrieval_latency']['p95_ms']} ms"),
    ("lat n", f"n = {S['retrieval_latency']['n']}"),
    ("warm e2e median", f"{S['latency_summary']['eval_total_s_warm']['median']} s"),
    ("article discovered", fmt(S["article_attempts"]["discovered"])), ("article dup", fmt(S["article_attempts"]["duplicates"])),
    ("bulletin attempts", f"{S['bulletin_attempts']['attempted']} processing attempts"),
    ("mrr 3nov", f"{S['rag_eval']['gen_3nov']['summary']['mrr']:.3f}"),
    ("grounded ext", f"{S['rag_eval']['ext_3nov']['summary']['groundedness']:.3f}"),
    ("sample chunks", "142, 143 and 78 words"),
]
for gid, name, n in S["genres_articles"][:5]:
    checks.append((f"genre {gid}", f"{name} ({n}, {100 * n / D['articles']:.1f}%)"))
for t, v in S["mmr_by_programme"].items():
    checks.append((f"mmr {t}", f"{v['repeat_pct']}%"))
for label, t in checks:
    need(label, t)

# figure / table / equation references resolve
figs = set(int(x) for x in re.findall(r"!\[Figure (\d+)\.", md))
for n in set(int(x) for x in re.findall(r"Figure (\d+)", md)):
    if n not in figs:
        fails.append(f"Figure {n} referenced but not present")
for f in re.findall(r"\]\((figures/[^)]+)\)", md):
    if not os.path.isfile(os.path.join(PAPER, f)):
        fails.append(f"missing image {f}")
tabs = set(int(x) for x in re.findall(r"^Table: Table (\d+)\.", md, re.M))
for n in set(int(x) for x in re.findall(r"Table (\d+)", md)):
    if n not in tabs:
        fails.append(f"Table {n} referenced but not defined")
eqs = set(int(x) for x in re.findall(r"\\qquad \((\d+)\)\$\$", md))
for n in set(int(x) for x in re.findall(r"Equations? (\d+)", md)):
    if n not in eqs:
        fails.append(f"Equation {n} referenced but not defined")
if sorted(figs) != list(range(1, len(figs) + 1)) or sorted(tabs) != list(range(1, len(tabs) + 1)) or \
        sorted(eqs) != list(range(1, len(eqs) + 1)):
    fails.append(f"numbering gaps: figures {sorted(figs)}, tables {sorted(tabs)}, equations {sorted(eqs)}")
# references empty
tail = md.split("# References {.unnumbered}", 1)
if len(tail) != 2 or tail[1].strip():
    fails.append("References section missing or not empty")
for name in ("Varad More", "Manav Jadhav", "Soham Jadhav", "Anant Nimkar"):
    need("author", name)

print(f"figures {len(figs)}, tables {len(tabs)}, equations {len(eqs)}, numeric checks {len(checks)}")
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
