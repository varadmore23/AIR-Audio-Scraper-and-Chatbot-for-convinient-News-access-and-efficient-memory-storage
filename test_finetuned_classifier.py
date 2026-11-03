"""
test_finetuned_classifier.py - Lightweight tests for the fine-tuned genre classifier.

Tests:
  a) classify_genres_batch_finetuned() returns correct shape and valid labels
     on domain sentences even when no fine-tuned model is present (via fallback).
  b) Missing-model path logs a WARNING and does NOT raise an exception.
  c) process_transcript() with classifier='finetuned' doesn't throw when model absent.

Style: plain assert + sys.exit, matching the project's existing test files.
"""

import logging
import os
import sys
import warnings

# Force CWD to repo root (relevant when run from another directory)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Test 1: classify_genres_batch_finetuned() shape + valid labels
# ---------------------------------------------------------------------------

print("Test 1: classify_genres_batch_finetuned() — shape and valid labels")

from nlp_pipeline import classify_genres_batch_finetuned, GENRE_LABELS, FINETUNED_MODEL_PATH

domain_sentences = [
    "ISRO successfully launched a new earth observation satellite into low earth orbit.",
    "The Rajya Sabha passed the amendment bill after a marathon debate.",
    "RBI kept the repo rate unchanged at 6.5 percent in its monetary policy meeting.",
    "India and Australia played a thrilling T20 cricket match at Wankhede Stadium.",
]

# Whether the model is present or not, the function must NOT raise.
try:
    results = classify_genres_batch_finetuned(domain_sentences)
except Exception as exc:
    _assert(False, f"classify_genres_batch_finetuned() raised an exception: {exc}")

# Shape: same number of outputs as inputs
_assert(len(results) == len(domain_sentences), (
    f"Expected {len(domain_sentences)} results, got {len(results)}"
))

for i, (label, conf) in enumerate(results):
    # Label must be one of the 10 GENRE_LABELS
    _assert(label in GENRE_LABELS, (
        f"Result[{i}] label '{label}' is not in GENRE_LABELS"
    ))
    # Confidence must be in [0.0, 1.0]
    _assert(0.0 <= conf <= 1.0, (
        f"Result[{i}] confidence {conf:.4f} is out of [0, 1] range"
    ))

print(f"  [OK] {len(results)} results, all labels valid, all confidences in [0, 1]")
for sent, (label, conf) in zip(domain_sentences, results):
    print(f"       '{sent[:60]}...' -> {label!r} ({conf:.3f})")


# ---------------------------------------------------------------------------
# Test 2: Missing-model fallback logs WARNING and doesn't throw
# ---------------------------------------------------------------------------

print("\nTest 2: Missing-model fallback — WARNING logged, no exception")

import tempfile
import shutil
from nlp_pipeline import _get_finetuned_classifier
import nlp_pipeline

# Temporarily redirect _finetuned_classifier cache + point model path to a
# non-existent directory to force the missing-model code path.
original_path = nlp_pipeline.FINETUNED_MODEL_PATH
original_cache = nlp_pipeline._finetuned_classifier

# Patch to a guaranteed-missing path
nlp_pipeline.FINETUNED_MODEL_PATH = "__nonexistent_model_dir_test__"
nlp_pipeline._finetuned_classifier = None  # clear cache

# Capture warning logs
import io
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.WARNING)
logging.getLogger("nlp_pipeline").addHandler(handler)

try:
    result = _get_finetuned_classifier()
    _assert(result is None, "Expected None when model dir is missing")

    log_output = log_stream.getvalue()
    _assert("warning" in log_output.lower() or "Fine-tuned model not found" in log_output, (
        "Expected a WARNING log message about missing model, got: " + log_output
    ))
    print("  [OK] _get_finetuned_classifier() returned None with WARNING logged")

    # Also check that classify_genres_batch_finetuned() falls back gracefully
    fallback_results = classify_genres_batch_finetuned(domain_sentences[:1])
    _assert(len(fallback_results) == 1, "Fallback should return 1 result for 1 input sentence")
    _assert(fallback_results[0][0] in GENRE_LABELS, "Fallback result label must be valid")
    print("  [OK] classify_genres_batch_finetuned() fell back silently — no exception")

finally:
    # Restore original values
    nlp_pipeline.FINETUNED_MODEL_PATH = original_path
    nlp_pipeline._finetuned_classifier = original_cache
    logging.getLogger("nlp_pipeline").removeHandler(handler)


# ---------------------------------------------------------------------------
# Test 3: process_transcript() with classifier='finetuned' doesn't throw
# ---------------------------------------------------------------------------

print("\nTest 3: process_transcript(classifier='finetuned') — no exception even when model absent")

from nlp_pipeline import process_transcript

test_text = (
    "ISRO launched a communication satellite today. "
    "The Rajya Sabha passed the finance bill with amendments."
)

# Force missing model for this test too
nlp_pipeline.FINETUNED_MODEL_PATH = "__nonexistent_model_dir_test__"
nlp_pipeline._finetuned_classifier = None

try:
    segments = process_transcript(test_text, classifier="finetuned", fast=False)
    _assert(isinstance(segments, list), "Expected a list of segments")
    _assert(all("genre" in seg for seg in segments), "Each segment must have a 'genre' key")
    _assert(all(seg["genre"] in GENRE_LABELS for seg in segments), "All genres must be valid GENRE_LABELS")
    print(f"  [OK] {len(segments)} segment(s) produced, all with valid genres — fallback path confirmed")
finally:
    nlp_pipeline.FINETUNED_MODEL_PATH = original_path
    nlp_pipeline._finetuned_classifier = original_cache


# ---------------------------------------------------------------------------
# Test 4: --fast backward compat still maps to fast/zero-shot path
# ---------------------------------------------------------------------------

print("\nTest 4: process_transcript(fast=True) still uses zero-shot fast (backward compat)")

segments_fast = process_transcript(test_text, fast=True)
_assert(isinstance(segments_fast, list), "Expected a list")
_assert(all(seg["genre"] in GENRE_LABELS for seg in segments_fast), "All genres must be valid")
print(f"  [OK] fast=True produced {len(segments_fast)} segment(s) — backward compat OK")


# ---------------------------------------------------------------------------
# All tests passed
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("  ALL TESTS PASSED")
print("=" * 50)
sys.exit(0)
