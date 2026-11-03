"""
make_figures.py - all paper figures, drawn from research_paper/data/paper_stats.json (real project data).

    python research_paper/scripts/make_figures.py   -> research_paper/figures/fig*.png (300 dpi)
                                                       research_paper/figures/fig*.csv (plotted values)
"""
import csv
import datetime as dt
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PAPER, "figures")
os.makedirs(FIG, exist_ok=True)
S = json.load(open(os.path.join(PAPER, "data", "paper_stats.json"), encoding="utf-8"))

NAVY, ORANGE, TEAL, GREY, RED, LIGHT = "#1F3A93", "#C8641E", "#2A9D8F", "#7A7A8C", "#B3261E", "#DCE3F2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6EE",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
                     "figure.dpi": 150})


def save(fig, name, rows=None, header=None):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    if rows is not None:
        with open(os.path.join(FIG, name + ".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    print("saved", name)


# ---------------------------------------------------------------- Fig 1 architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def node(x, y, w, h, title, sub, fc=LIGHT, tc="#111", bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec="#8C96B8", lw=0.8))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8.2, weight="bold" if bold else "normal", color=tc)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.6, color=tc if tc != "#111" else "#333")

    def arr(x1, y1, x2, y2, txt=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=NAVY))
        if txt:
            ax.text((x1 + x2) / 2 + 1, (y1 + y2) / 2, txt, fontsize=6.2, color=GREY, va="center")

    ax.text(1, 98, "Ingestion (offline, incremental jobs)", fontsize=8, weight="bold", color=NAVY)
    A, B, C, W = 1, 35, 69, 30
    node(A, 87, W, 7.5, "WordPress REST API", "posts: HTML, category, dates")
    node(B, 87, W, 7.5, "NewsOnAIR", "newsonair.gov.in", fc=NAVY, tc="white")
    node(C, 87, W, 7.5, "Bulletin archive pages", "id, programme, time, MP3 URL")
    arr(B, 90.7, A + W, 90.7)
    arr(B + W, 90.7, C, 90.7)
    node(A, 77, W, 7.5, "Article extraction", "HTML → paragraphs + metadata")
    node(C, 77, W, 7.5, "Audio download (cached)", "sequential, 1.5 s delay, retry")
    arr(A + W / 2, 87, A + W / 2, 84.5)
    arr(C + W / 2, 87, C + W / 2, 84.5)
    node(A, 67, W, 7.5, "Article chunking", "paragraph packing, ≤ 150 words")
    node(C, 67, W, 7.5, "Speech-to-text", "faster-whisper small, int8, VAD,\nword timestamps")
    arr(A + W / 2, 77, A + W / 2, 74.5)
    arr(C + W / 2, 77, C + W / 2, 74.5)
    node(C, 57, W, 7.5, "Sentence segmentation", "NLTK punkt; 1 sentence = 1 chunk")
    arr(C + W / 2, 67, C + W / 2, 64.5)
    node(B, 57, W, 7.5, "Embedding", "all-MiniLM-L6-v2, 384-d, L2-norm")
    arr(A + W / 2, 67, B + 4, 64.5)
    arr(C, 60.7, B + W, 60.7)
    node(C, 47, W, 7.5, "MMR novelty gate", "bulletins: drop if cos ≥ λ = 0.85\nto a stored sentence")
    node(B, 47, W, 7.5, "Genre taxonomy (18)", "publisher category → keywords\n→ NLI label → desk")
    arr(B + W, 59, C + 4, 54.5)
    arr(C, 50.7, B + W, 50.7)
    arr(B + 6, 57, B + 6, 54.5)
    ax.text(1, 43, "Storage", fontsize=8, weight="bold", color=NAVY)
    node(A, 32, W, 9, "FAISS IndexFlatIP", "8,858 vectors (8,854 active)")
    node(B, 32, W, 9, "SQLite + FTS5 (BM25)", "chunks, metadata, timestamps,\ningestion state, query log")
    node(C, 32, W, 9, "Audio + transcripts", "MP3 per bulletin,\nsentence start/end (s)")
    arr(B + W / 2, 47, B + W / 2, 41)
    arr(B + 3, 47, A + W / 2, 41)
    arr(C + W / 2, 47, C + W / 2, 41)
    ax.text(1, 27.5, "Question answering (online, FastAPI)", fontsize=8, weight="bold", color=NAVY)
    node(1, 16, 18, 9, "Query parsing", "dates, time of day,\ngenre, desk, topic")
    node(22.5, 16, 18, 9, "Tier 1: SQL", "date / hour / genre /\ntype prefilter")
    node(44, 16, 18, 9, "Tier 2: hybrid", "dense top-30 + BM25\ntop-30 → RRF (k = 60)")
    node(65.5, 16, 15.5, 9, "Gates + dedup", "cosine thresholds,\nabsent terms")
    node(84.5, 16, 14.5, 9, "Context", "whole articles,\nsentence ± 1")
    for x1, x2 in ((19, 22.5), (40.5, 44), (62, 65.5), (81, 84.5)):
        arr(x1, 20.5, x2, 20.5)
    arr(53, 32, 53, 25)
    node(56, 1, 26, 10, "Gemini generation", "question-typed prompt,\n[S#] citations, fallback chain")
    node(1, 1, 50, 10, "Answer with validated citations", "S# → article, or transcript at the cited second + audio;\n"
         "“not covered” when no evidence clears the gates", fc=NAVY, tc="white")
    node(85, 1, 14, 10, "Extractive\nfallback", "no LLM", fc="#F3E3D6")
    arr(91.75, 16, 69, 11)
    arr(56, 6, 51, 6)
    arr(85, 6, 82, 6)
    save(fig, "fig01_architecture")


# ---------------------------------------------------------------- Fig 2 daily
def fig_daily():
    rows = S["daily"]
    days = [dt.date.fromisoformat(r[0]) for r in rows]
    a = [r[1] for r in rows]
    b = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(days, a, color=NAVY, width=0.8, label="Articles")
    ax.bar(days, b, bottom=a, color=ORANGE, width=0.8, label="Broadcast bulletins")
    ax.set_title("Indexed documents per day (1 Oct – 2 Nov 2026, IST)")
    ax.set_xlabel("Publication / broadcast date")
    ax.set_ylabel("Documents (count)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig03_daily_documents", rows, ["date", "articles", "bulletins"])


# ---------------------------------------------------------------- Fig 3 coverage timeline
def fig_coverage():
    c = S["coverage"]
    w0 = dt.date.fromisoformat(S["environment"]["config"]["report_window"][0])
    w1 = dt.date.fromisoformat(S["environment"]["config"]["report_window"][1])
    art = [dt.date.fromisoformat(x) for x in S["article_dates"]]
    bc = [dt.date.fromisoformat(r[0]) for r in S["daily"] if r[2]]
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.barh(2, (w1 - w0).days + 1, left=w0, height=0.5, color="#C9CFE3")
    ax.vlines(art, 0.75, 1.25, color=NAVY, lw=1.4)
    ax.vlines(bc, -0.25, 0.25, color=ORANGE, lw=1.4)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([f"Reporting window\n({c['window_days']} days, configured)",
                        f"Articles\n({c['articles']['distinct_dates']} dates)",
                        f"Broadcasts\n({c['broadcasts']['distinct_dates']} dates)"])
    ax.set_xlim(w0 - dt.timedelta(days=5), w1 + dt.timedelta(days=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Date (IST)")
    ax.set_title("Actual indexed coverage vs the configured reporting window")
    ax.grid(axis="y", visible=False)
    rows = [["window", str(w0), str(w1), c["window_days"]], ["articles", c["articles"]["earliest"], c["articles"]["latest"],
            c["articles"]["distinct_dates"]], ["broadcasts", c["broadcasts"]["earliest"], c["broadcasts"]["latest"],
            c["broadcasts"]["distinct_dates"]]]
    save(fig, "fig02_coverage", rows, ["series", "earliest", "latest", "distinct_dates"])


# ---------------------------------------------------------------- Fig 4 broadcasts by type + MMR
def fig_broadcasts():
    names = {"midday": "Midday News", "evening": "Evening News", "hourly": "Hourly News", "news-at-nine": "News at Nine",
             "morning": "Morning News"}
    bt = S["broadcast_types"]
    mm = S["mmr_by_programme"]
    title_of = {t: t for t in names}  # mmr_by_programme is keyed by broadcast type
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    labs = [names[r[0]] for r in bt]
    a1.barh(labs, [r[1] for r in bt], color=ORANGE)
    for i, r in enumerate(bt):
        a1.text(r[1] + 0.6, i, f"{r[1]} ({r[2]:.2f} h)", va="center", fontsize=7.5)
    a1.invert_yaxis()
    a1.set_xlabel("Indexed bulletins (count)")
    a1.set_title("(a) Bulletins by programme")
    a1.set_xlim(0, max(r[1] for r in bt) * 1.45)
    order = [r[0] for r in bt]
    pcts = [mm[title_of[t]]["repeat_pct"] for t in order]
    a2.barh(labs, pcts, color=NAVY)
    for i, t in enumerate(order):
        a2.text(pcts[i] + 0.6, i, f"{pcts[i]:.1f}% of {mm[title_of[t]]['sentences']}", va="center", fontsize=7.5)
    a2.invert_yaxis()
    a2.set_xlim(0, 55)
    a2.set_yticklabels([])
    a2.set_xlabel("Sentences removed by the MMR gate (%)")
    a2.set_title("(b) Repetition removed at ingestion")
    rows = [[names[r[0]], r[1], r[2], mm[title_of[r[0]]]["sentences"], mm[title_of[r[0]]]["repeats"],
             mm[title_of[r[0]]]["repeat_pct"]] for r in bt]
    save(fig, "fig04_broadcasts", rows, ["programme", "bulletins", "hours", "sentences", "repeats_removed", "repeat_pct"])


# ---------------------------------------------------------------- Fig 5 genres
def fig_genres():
    ga = S["genres_articles"]
    gt = {g: n for g, _, n in S["genres_transcript_chunks"]}
    ta = sum(n for *_, n in ga)
    tt = sum(gt.values())
    names = [n for _, n, _ in ga]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = range(len(ga))
    pa = [100 * n / ta for *_, n in ga]
    pt = [100 * gt.get(g, 0) / tt for g, _, _ in ga]
    ax.barh([i - 0.2 for i in y], pa, height=0.4, color=NAVY, label=f"Articles (n = {ta:,})")
    ax.barh([i + 0.2 for i in y], pt, height=0.4, color=ORANGE, label=f"Bulletin sentences (n = {tt:,})")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Share of items (%)")
    ax.set_title("Primary genre distribution (18-genre taxonomy)")
    ax.legend(loc="center right")
    rows = [[g, n, c, round(100 * c / ta, 2), gt.get(g, 0), round(100 * gt.get(g, 0) / tt, 2)] for g, n, c in ga]
    save(fig, "fig05_genres", rows, ["genre_id", "genre", "articles", "articles_pct", "bulletin_sentences",
                                     "bulletin_sentences_pct"])


# ---------------------------------------------------------------- Fig 6 lambda sweep
def fig_lambda():
    e1 = S["phase3_experiments"]["e1_mmr_sweep"]
    lam = [x["lambda"] for x in e1["sweep"]]
    st_ = [x["storage_retained_pct"] for x in e1["sweep"]]
    vo = [x["vocab_retained_pct"] for x in e1["sweep"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.plot(lam, st_, "-o", color=ORANGE, ms=4, label="Sentences stored")
    ax.plot(lam, vo, "-o", color=NAVY, ms=4, label="Vocabulary retained")
    ax.axvline(0.85, color=GREY, ls="--", lw=0.8)
    k = lam.index(0.85)
    ax.annotate(f"λ = 0.85\n{st_[k]}% stored\n{vo[k]}% vocabulary", (0.85, st_[k]), xytext=(0.66, 38), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))
    ax.set_xlabel("Novelty threshold λ (cosine)")
    ax.set_ylabel("Retained (%)")
    ax.set_ylim(0, 102)
    ax.set_title(f"MMR threshold sweep ({e1['sentences']:,} sentences, {e1['bulletins']} bulletins)")
    ax.legend(loc="lower right")
    save(fig, "fig08_lambda_sweep", [[a, b, c] for a, b, c in zip(lam, st_, vo)], ["lambda", "stored_pct", "vocab_pct"])


# ---------------------------------------------------------------- Fig 7 two-tier vs naive
def fig_twotier():
    e2 = S["phase3_experiments"]["e2_two_tier"]
    s = e2["summary"]
    keys = [("date_precision_at5", "Top-5 on the\nasked date"), ("date_correct_top1", "Top-1 on the\nasked date"),
            ("exact_sentence_hit_at5", "Source sentence\nin top 5")]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    x = range(len(keys))
    tv = [s["two_tier"][k] for k, _ in keys]
    nv = [s["naive"][k] for k, _ in keys]
    b1 = ax.bar([i - 0.2 for i in x], tv, 0.4, color=NAVY, label="Two-tier (SQL prefilter + hybrid)")
    b2 = ax.bar([i + 0.2 for i in x], nv, 0.4, color=ORANGE, label="Hybrid over whole index")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([l for _, l in keys])
    ax.set_ylabel("Questions (%)")
    ax.set_ylim(0, 112)
    ax.set_title(f"Date-anchored retrieval ({e2['questions']} questions)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.5)
    rows = [[k, s["two_tier"][k], s["naive"][k]] for k, _ in keys] + [["mean_candidates", s["two_tier"]["mean_candidates"],
                                                                        s["naive"]["mean_candidates"]]]
    save(fig, "fig09_two_tier", rows, ["metric", "two_tier", "naive"])


# ---------------------------------------------------------------- Fig 8 latency
def fig_latency():
    lat = json.load(open(os.path.join(PAPER, "data", "retrieval_latency.json"), encoding="utf-8"))
    xs = lat["samples_ms"]
    e2e = [t for t in S["eval_generative_timings"] if "total_s" in t]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a1.hist(xs, bins=30, color=NAVY)
    sm = lat["summary"]
    a1.axvline(sm["median_ms"], color=ORANGE, lw=1, ls="--")
    a1.text(sm["median_ms"] + 3, a1.get_ylim()[1] * 0.85, f"median {sm['median_ms']} ms\np95 {sm['p95_ms']} ms", fontsize=7.5)
    a1.set_xlabel("Retrieval latency (ms)")
    a1.set_ylabel("Questions (count)")
    a1.set_title(f"(a) Retrieval, warm ({sm['n']} questions)")
    lab = [f"Q{i + 1}" for i in range(len(e2e))]
    a2.bar(lab, [t["total_s"] for t in e2e], color=ORANGE, label="End-to-end")
    a2.bar(lab, [t["total_s"] - t["generation_s"] for t in e2e], color=NAVY, label="Non-generation part")
    a2.set_yscale("log")
    a2.set_ylabel("Time (s, log scale)")
    a2.set_xlabel("Evaluation question (generative run)")
    a2.set_title("(b) /chat answer time per question")
    a2.legend(loc="upper right", fontsize=7)
    a2.tick_params(axis="x", labelsize=7)
    rows = [["retrieval_ms", v] for v in xs] + [[f"Q{i + 1}_total_s", t["total_s"]] for i, t in enumerate(e2e)] + \
           [[f"Q{i + 1}_generation_s", t["generation_s"]] for i, t in enumerate(e2e)]
    save(fig, "fig11_latency", rows, ["measure", "value"])


# ---------------------------------------------------------------- Fig 9 RAG evaluation
def fig_eval():
    ev = S["rag_eval"]
    keys = [("hit_rate", "Hit rate"), ("mrr", "MRR"), ("citation_validity", "Citation\nvalidity"),
            ("citation_correctness", "Citation\ncorrectness"), ("groundedness", "Grounded-\nness"),
            ("refusal_accuracy_unanswerable", "Refusal\naccuracy")]
    runs = [("gen_2nov", "Gemini, 2 Nov (81 articles)", "#9AA7CF"), ("gen_3nov", "Gemini, 3 Nov (2,075 articles)", NAVY),
            ("ext_3nov", "Extractive, 3 Nov (2,075 articles)", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(keys))
    for j, (r, lab, col) in enumerate(runs):
        vals = [ev[r]["summary"][k] for k, _ in keys]
        ax.bar([i + (j - 1) * 0.27 for i in x], vals, 0.27, color=col, label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels([l for _, l in keys])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Answer-level evaluation on the 12-question NewsOnAIR set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)
    rows = [[k] + [ev[r]["summary"][k] for r, _, _ in runs] for k, _ in keys]
    save(fig, "fig10_rag_eval", rows, ["metric", "gemini_2nov", "gemini_3nov", "extractive_3nov"])


# ---------------------------------------------------------------- Fig 10 ingestion outcomes
def fig_ingestion():
    ba, aa = S["bulletin_attempts"], S["article_attempts"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    lab1 = ["Attempts", "Indexed", "Failed\n(retried later)", "Duplicate"]
    v1 = [ba["attempted"], ba["indexed"], ba["failed"], ba["attempted"] - ba["indexed"] - ba["failed"]]
    a1.bar(lab1, v1, color=[GREY, NAVY, RED, ORANGE])
    for i, v in enumerate(v1):
        a1.text(i, v + 1, str(v), ha="center", fontsize=7.5)
    a1.set_ylabel("Bulletins (count)")
    a1.set_title("(a) Bulletin ingestion attempts")
    lab2 = ["Discovered", "New", "Unchanged\n(skipped)", "Updated"]
    v2 = [aa["discovered"], aa["new"], aa["duplicates"], aa["updated"]]
    a2.bar(lab2, v2, color=[GREY, NAVY, ORANGE, TEAL])
    for i, v in enumerate(v2):
        a2.text(i, v + 25, f"{v:,}", ha="center", fontsize=7.5)
    a2.set_ylabel("Article discoveries (count)")
    a2.set_title("(b) Article ingestion outcomes")
    rows = [["bulletins", a, b] for a, b in zip(["attempts", "indexed", "failed", "duplicate"], v1)] + \
           [["articles", a, b] for a, b in zip(["discovered", "new", "unchanged", "updated"], v2)]
    save(fig, "fig07_ingestion", rows, ["kind", "outcome", "count"])


# ---------------------------------------------------------------- Fig 11 chunk lengths
def fig_chunks():
    import sqlite3
    ROOT = os.path.dirname(PAPER)
    db = sqlite3.connect(f"file:{os.path.join(ROOT, 'news_store.db')}?mode=ro", uri=True)
    aw = [len(r[0].split()) for r in db.execute("SELECT text FROM segments WHERE active=1 AND content_type='article'")]
    tw = [len(r[0].split()) for r in db.execute("SELECT text FROM segments WHERE active=1 AND content_type='audio_transcript'")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a1.hist([min(w, 260) for w in aw], bins=26, color=NAVY)
    a1.axvline(150, color=ORANGE, ls="--", lw=1)
    a1.text(153, a1.get_ylim()[1] * 0.8, "packing limit\n150 words", fontsize=7.5)
    a1.set_xlabel("Words per article chunk")
    a1.set_ylabel("Chunks (count)")
    a1.set_title(f"(a) Article chunks (n = {len(aw):,})")
    a2.hist([min(w, 80) for w in tw], bins=40, color=ORANGE)
    a2.set_xlabel("Words per bulletin sentence (capped at 80)")
    a2.set_ylabel("Sentences (count)")
    a2.set_title(f"(b) Transcript chunks (n = {len(tw):,})")
    save(fig, "fig06_chunk_lengths", [["article", w] for w in aw] + [["transcript", w] for w in tw], ["kind", "words"])


if __name__ == "__main__":
    for f in (fig_architecture, fig_daily, fig_coverage, fig_broadcasts, fig_genres, fig_lambda, fig_twotier, fig_latency,
              fig_eval, fig_ingestion, fig_chunks):
        f()
