"""The evaluation harness runs end-to-end on the repo's sample transcripts (offline stand-in models)."""

import json
import os
import shutil

import config
import eval_rag
import ingest_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_eval_on_sample_corpus(env, monkeypatch):
    shutil.copytree(os.path.join(ROOT, "output", "text"), env / "output" / "text")
    shutil.copy(os.path.join(ROOT, "output", "metadata.csv"), env / "output" / "metadata.csv")
    ingest_pipeline.run(metadata_csv=str(env / "output" / "metadata.csv"), fast=True, articles=False)
    with open(os.path.join(ROOT, "eval", "rag_eval_set.json"), encoding="utf-8") as f:
        spec = json.load(f)
    monkeypatch.setattr(config, "DATE_ANCHOR", "corpus")
    s = eval_rag.evaluate(spec["cases"])["summary"]
    assert s["citation_validity"] == 1.0
    assert s["refusal_accuracy_unanswerable"] == 1.0 and s["false_refusals_answerable"] == 0
    assert s["hit_rate"] >= 0.9 and s["groundedness"] >= 0.9
