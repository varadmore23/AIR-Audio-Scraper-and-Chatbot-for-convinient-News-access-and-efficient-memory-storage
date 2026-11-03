"""The detailed-reasoning question set runs end-to-end on the demo articles (offline stand-in models)."""

import json
import os

import config
import eval_rag
import news_articles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reasoning_set_on_demo_articles(env, monkeypatch):
    news_articles.ingest()
    with open(os.path.join(ROOT, "eval", "rag_reasoning_set.json"), encoding="utf-8") as f:
        spec = json.load(f)
    monkeypatch.setattr(config, "DATE_ANCHOR", "corpus")
    report = eval_rag.evaluate(spec["cases"])
    for r in report["rows"]:
        print(r["question"][:60], r["question_type"], r["rank"], r["words"], r["checks"], r["insufficient"])
    s = report["summary"]
    print(s)
    assert s["citation_validity"] == 1.0 and s["hit_rate"] == 1.0 and s["groundedness"] >= 0.9
    assert s["refusal_accuracy_unanswerable"] == 1.0 and s["false_refusals_answerable"] == 0
    failed = {r["question"]: r["checks"] for r in report["rows"] if not all(r["checks"].values())}
    assert not failed, failed
