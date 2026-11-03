"""
build_chatbot_eval.py - deterministic chatbot evaluation set built from real indexed NewsOnAIR documents.

Every case stores the question, the expected source document(s), the evidence it was built from and the
expected answer characteristics. No expected answer text is written: answers are judged against the
real source by eval_rag.py (citations, groundedness, phrases taken verbatim from the source).

    python eval/build_chatbot_eval.py          -> eval/chatbot_eval_set.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import taxonomy  # noqa: E402
import vector_store  # noqa: E402

SEED = 20261103
STOP = set(vector_store.STOPWORDS)
# Topics checked to be absent from the index at build time (see insufficient cases).
ABSENT_TOPICS = [("What did NewsOnAIR report about the price of Bitcoin?", ["bitcoin"]),
                 ("What was the result of the UEFA Champions League final?", ["uefa"]),
                 ("What did the bulletins say about Tesla's quarterly earnings?", ["tesla"])]


def words(t):
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", t) if w.lower() not in STOP]


def number_phrase(text):
    """A distinctive figure from the source (e.g. '8-0', '27.4 per cent') to check the answer quotes it."""
    m = re.search(r"\b\d[\d,.]*(?:-\d+)?(?:\s?(?:per cent|percent|crore|lakh|million|billion|kilometres|km))?\b", text)
    return m.group(0).strip() if m and len(m.group(0)) >= 2 else None


def main():
    rnd = random.Random(SEED)
    db = sqlite3.connect(f"file:{os.path.join(ROOT, 'news_store.db')}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    arts = [dict(r) for r in db.execute(
        "SELECT article_id, headline, date, genre_id, source_category, word_count, data_json FROM articles "
        "WHERE article_id LIKE 'noa-post-%' AND date BETWEEN '2026-10-05' AND '2026-11-02' AND word_count >= 150 "
        "ORDER BY article_id")]
    rnd.shuffle(arts)
    cases, used = [], set()

    def add(case):
        key = hashlib.sha1(case["question"].lower().encode()).hexdigest()
        if key in used:
            return
        used.add(key)
        case["id"] = f"c{len(cases) + 1:02d}"
        cases.append(case)

    # 1) direct factual questions about specific articles (headline as the topic)
    for a in [a for a in arts if a["genre_id"] not in ("other",)][:5]:
        body = " ".join(json.loads(a["data_json"])["body"])
        ph = number_phrase(body)
        add({"category": "factual", "question": f"What did NewsOnAIR report about: {a['headline']}?",
             "expected_docs": [a["article_id"]], "evidence": body[:300], "expected_type": "simple/standard",
             "must_mention": [ph] if ph else [], "max_words": 400, "answerable": True, "check_source_urls": True})
    # 2) analytical questions on long articles
    for a in [a for a in arts if a["word_count"] >= 380][:3]:
        body = " ".join(json.loads(a["data_json"])["body"])
        add({"category": "analytical",
             "question": f"Explain the background, key developments and implications of: {a['headline']}",
             "expected_docs": [a["article_id"]], "evidence": body[:300], "expected_type": "explain",
             "min_words": 200, "answerable": True, "check_source_urls": True})
    # 3) genre-scoped questions (taxonomy genre as UI scope)
    for gid in ("sports", "finance", "environment"):
        name = taxonomy.genre_name(gid)
        add({"category": "genre", "question": f"What are the main {name} stories in the latest news?",
             "genre": gid, "expected_docs": ["noa-"], "evidence": f"any indexed document with genre {gid}",
             "expected_type": "overview", "min_words": 120, "answerable": True})
    # 4) cross-genre comparison
    add({"category": "cross_genre", "question": "Compare the major sports stories with the major business stories "
         "reported on 1 November 2026", "expected_docs": ["noa-"], "evidence": "articles of 2026-11-01",
         "expected_type": "compare", "min_words": 150, "answerable": True})
    # 5) date-specific summaries
    for d in ("2026-10-15", "2026-10-28"):
        day = dt.date.fromisoformat(d)
        add({"category": "summary", "question": f"Summarize the news of {day.day} {day.strftime('%B %Y')}",
             "expected_docs": ["noa-"], "expected_date": d, "evidence": f"documents dated {d}",
             "expected_type": "overview", "min_words": 150, "answerable": True})
    # 6) broadcast transcript questions (date + programme + topic), audio timestamps required
    rows = [dict(r) for r in db.execute(
        "SELECT s.doc_id, s.text, s.date, b.title FROM segments s JOIN broadcasts b ON b.doc_id = s.doc_id "
        "WHERE s.active = 1 AND s.content_type = 'audio_transcript' AND LENGTH(s.text) > 90 ORDER BY s.faiss_id")]
    rnd.shuffle(rows)
    for r in rows[:3]:
        day = dt.date.fromisoformat(r["date"])
        topic = " ".join(words(r["text"])[:6])
        add({"category": "broadcast", "question": f"What did the {r['title']} on {day.day} {day.strftime('%B %Y')} "
             f"say about {topic}?", "expected_docs": [r["doc_id"]], "expected_date": r["date"], "evidence": r["text"],
             "expected_type": "standard", "require_audio": True, "answerable": True})
    # 7) insufficient context (topic words verified absent from the index)
    for q, terms in ABSENT_TOPICS:
        absent = all(not vector_store.term_in_corpus(t) for t in terms)
        if absent:
            add({"category": "insufficient", "question": q, "expected_docs": [], "evidence": f"absent terms: {terms}",
                 "expected_type": "refusal", "answerable": False})
    spec = {"name": "chatbot_eval_set", "version": 1, "seed": SEED,
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "corpus": {"articles": db.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
                       "broadcasts": db.execute("SELECT COUNT(*) FROM broadcasts").fetchone()[0]},
            "note": "Built from indexed documents; expected answers are not written - see eval_rag.py metrics.",
            "cases": cases}
    out = os.path.join(ROOT, "eval", "chatbot_eval_set.json")
    json.dump(spec, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    by = {}
    for c in cases:
        by[c["category"]] = by.get(c["category"], 0) + 1
    print(f"wrote {out}: {len(cases)} cases {by}")


if __name__ == "__main__":
    main()
