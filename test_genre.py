"""test_genre.py — Throwaway assertion script for _extract_genre fixes.

Tests 3 known false positives and 5 true positives.
Exit code 0 = all pass.
"""

import sys
sys.path.insert(0, ".")

from rag_engine import _extract_genre

TESTS = [
    # False-positive fixes (previously broken)
    ("What is the main news today?", None),
    ("Tell us about the ISRO launch", "Science & Technology"),
    ("Summarise the whole day", None),
    # True positives
    ("cricket match results", "Sports"),
    ("parliament election", "Politics"),
    ("RBI inflation data", "Economy"),
    ("ISRO satellite launch", "Science & Technology"),
    ("army military operations", "Defence & Security"),
]

passed = 0
failed = 0
for query, expected in TESTS:
    result = _extract_genre(query)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        failed += 1
        print(f"  {status}: {query!r} -> {result!r} (expected {expected!r})")
    else:
        passed += 1
        print(f"  {status}: {query!r} -> {result!r}")

print(f"\n{passed}/{passed + failed} passed")
if failed:
    sys.exit(1)
