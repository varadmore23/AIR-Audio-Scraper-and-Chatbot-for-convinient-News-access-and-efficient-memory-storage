"""
build_genre_training_data.py - Bootstrap genre training data for fine-tuning.

Pulls segment texts from news_store.db and output/text/ transcripts, runs the
existing zero-shot classifier to produce silver labels, and exports a CSV ready
for human review + fine-tuning.

Usage:
    # Basic: label all DB segments + transcripts
    python build_genre_training_data.py

    # With synthetic augmentation (uses GENRE_KEYWORD_MAP templates)
    python build_genre_training_data.py --augment

    # Custom output and DB paths
    python build_genre_training_data.py --augment --out my_labels.csv --db news_store.db

IMPORTANT: Fill in the 'true_genre' column before running train_genre_classifier.py.
           The 'predicted_genre' column is a silver label from the zero-shot model
           and may be incorrect — human review is required for training quality.
"""

import argparse
import csv
import logging
import os
import sqlite3
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("build_genre_training_data")

SQLITE_DB_PATH = "news_store.db"
TEXT_DIR = os.path.join("output", "text")
DEFAULT_OUT = "genre_training_data.csv"
COLUMNS = ["text", "predicted_genre", "confidence", "true_genre"]


# ---------------------------------------------------------------------------
# Synthetic augmentation templates (AIR-style sentences per genre)
# ---------------------------------------------------------------------------

# Templates use {kw} as a placeholder for a GENRE_KEYWORD_MAP keyword.
GENRE_TEMPLATES: dict[str, list[str]] = {
    "Politics": [
        "The {kw} session discussed key legislative reforms today.",
        "Senior leaders from {kw} addressed the press conference this evening.",
        "The {kw} committee tabled a new proposal on electoral procedures.",
        "Protests erupted outside parliament over the {kw} bill.",
        "The {kw} alliance held a crucial strategy meeting ahead of elections.",
    ],
    "Economy": [
        "The {kw} index climbed two percent in early morning trading.",
        "Analysts at {kw} revised their quarterly growth forecast upward.",
        "The government announced a new {kw} stimulus package worth fifty thousand crore rupees.",
        "Rising {kw} pressures forced the central bank to review its policy stance.",
        "Foreign investment in the {kw} sector reached a record high this quarter.",
    ],
    "Science & Technology": [
        "{kw} successfully demonstrated its latest orbital manoeuvre today.",
        "Scientists published a breakthrough {kw} study in a leading international journal.",
        "The new {kw} mission will carry three payloads into low earth orbit.",
        "India's {kw} sector attracted significant venture capital funding this year.",
        "Researchers unveiled a new {kw} platform capable of processing data at record speeds.",
    ],
    "Sports": [
        "India won the {kw} tournament after a thrilling final against Australia.",
        "The national {kw} team began intensive camp ahead of the championship.",
        "A record crowd watched the {kw} match at the stadium last evening.",
        "The {kw} federation announced new selection criteria for upcoming events.",
        "Star player led the {kw} team to a decisive victory in the semi-final.",
    ],
    "International": [
        "The two nations signed a bilateral agreement on {kw} cooperation.",
        "Diplomatic talks between {kw} and India resumed after a brief recess.",
        "The {kw} summit concluded with a joint communique on trade and security.",
        "India's ambassador to {kw} presented credentials at a formal ceremony.",
        "World leaders gathered in {kw} to discuss climate and economic challenges.",
    ],
    "Defence & Security": [
        "The {kw} conducted a successful exercise near the northern border.",
        "India's {kw} modernisation programme received budgetary approval this week.",
        "Joint {kw} drills between allied forces concluded at a forward base.",
        "The ministry reviewed the operational readiness of the {kw} units.",
        "A new {kw} surveillance system was commissioned along the coastal perimeter.",
    ],
    "Health": [
        "The government launched a new {kw} awareness campaign across rural districts.",
        "Doctors at the central {kw} hospital reported a decline in seasonal cases.",
        "The national {kw} authority approved three new treatment protocols today.",
        "A team of scientists developed a rapid diagnostic test for {kw} detection.",
        "Vaccination drives targeting {kw} reached over ten million beneficiaries.",
    ],
    "Environment": [
        "Heavy {kw} lashed the coastal belt overnight, disrupting transport links.",
        "The {kw} ministry unveiled a new policy to protect mangrove ecosystems.",
        "Unseasonal {kw} affected crop yields across three eastern states.",
        "India pledged to expand its {kw} energy capacity by thirty gigawatts.",
        "The {kw} authority issued an orange alert for five districts in the region.",
    ],
    "Society & Culture": [
        "The annual {kw} festival attracted thousands of visitors from across the country.",
        "New {kw} policies aim to improve literacy rates in underserved communities.",
        "The {kw} commission released its report on gender equity in higher education.",
        "Traditional {kw} art forms were showcased at the international heritage fair.",
        "A national {kw} awareness week began with events in all state capitals.",
    ],
    "Other": [
        "Officials reviewed the quarterly progress on {kw} infrastructure projects.",
        "The department released a white paper on emerging {kw} challenges.",
        "A special task force was constituted to study the impact of {kw} policies.",
        "Regional authorities convened to discuss implementation of {kw} guidelines.",
        "The committee recommended sweeping changes to existing {kw} regulations.",
    ],
}


def _genre_keyword_map() -> dict[str, str]:
    """Import GENRE_KEYWORD_MAP from rag_engine without pulling all of rag_engine."""
    try:
        from rag_engine import GENRE_KEYWORD_MAP
        return GENRE_KEYWORD_MAP
    except Exception:
        logger.warning("Could not import GENRE_KEYWORD_MAP from rag_engine -- augmentation will use defaults")
        return {}


def _generate_synthetic(genre: str, keyword_map: dict[str, str]) -> list[str]:
    """Generate template-based synthetic sentences for a genre."""
    templates = GENRE_TEMPLATES.get(genre, [])
    if not templates:
        return []

    # Collect keywords that map to this genre
    keywords = [kw for kw, g in keyword_map.items() if g == genre]
    if not keywords:
        keywords = [genre.lower()]

    sentences = []
    for kw in keywords[:8]:  # cap at 8 keywords per genre to avoid explosion
        for tpl in templates:
            sentences.append(tpl.replace("{kw}", kw))
    return sentences


# ---------------------------------------------------------------------------
# Text sources
# ---------------------------------------------------------------------------

def _load_db_texts(db_path: str) -> list[str]:
    """Load all segment texts from SQLite news_store.db."""
    if not os.path.isfile(db_path):
        logger.warning("DB not found: %s -- skipping DB source", db_path)
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT text FROM segments ORDER BY id").fetchall()
    conn.close()
    texts = [r[0].strip() for r in rows if r[0] and r[0].strip()]
    logger.info("Loaded %d segment texts from %s", len(texts), db_path)
    return texts


def _load_transcript_texts(text_dir: str) -> list[str]:
    """Walk output/text/ and collect sentences from all .txt transcript files."""
    if not os.path.isdir(text_dir):
        logger.info("Transcript dir not found: %s -- skipping transcript source", text_dir)
        return []

    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        tokenize = nltk.sent_tokenize
    except ImportError:
        logger.warning("NLTK not available -- using simple period split for transcripts")
        tokenize = lambda t: [s.strip() for s in t.split(".") if s.strip()]

    all_sentences: list[str] = []
    for root, _, files in os.walk(text_dir):
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().lstrip("\ufeff").strip()
                if content:
                    all_sentences.extend([s.strip() for s in tokenize(content) if len(s.strip()) > 15])
            except Exception as e:
                logger.warning("Could not read %s: %s", fpath, e)

    logger.info("Loaded %d sentences from transcript files in %s", len(all_sentences), text_dir)
    return all_sentences


# ---------------------------------------------------------------------------
# Silver-label classification
# ---------------------------------------------------------------------------

def _classify_batch(texts: list[str], fast: bool = True) -> list[tuple[str, float]]:
    """Run zero-shot genre classification over a list of texts."""
    from nlp_pipeline import classify_genres_batch
    logger.info("Classifying %d texts with zero-shot (fast=%s)...", len(texts), fast)

    results: list[tuple[str, float]] = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        results.extend(classify_genres_batch(batch, fast=fast))
        logger.info("  Classified %d / %d", min(start + batch_size, len(texts)), len(texts))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(
    db_path: str = SQLITE_DB_PATH,
    text_dir: str = TEXT_DIR,
    out_path: str = DEFAULT_OUT,
    augment: bool = False,
    fast: bool = True,
) -> int:
    """Build and export the genre training data CSV.

    Returns the number of rows written.
    """
    # Collect raw texts
    texts: list[str] = []

    db_texts = _load_db_texts(db_path)
    texts.extend(db_texts)

    transcript_texts = _load_transcript_texts(text_dir)
    # Avoid exact duplicates with DB texts
    db_set = set(db_texts)
    for t in transcript_texts:
        if t not in db_set:
            texts.append(t)

    logger.info("Total unique texts before augmentation: %d", len(texts))

    # Synthetic augmentation
    synthetic_by_genre: dict[str, list[str]] = {}
    if augment:
        from nlp_pipeline import GENRE_LABELS
        kw_map = _genre_keyword_map()
        total_synth = 0
        for genre in GENRE_LABELS:
            synth = _generate_synthetic(genre, kw_map)
            if synth:
                synthetic_by_genre[genre] = synth
                total_synth += len(synth)
        logger.info("Generated %d synthetic sentences across %d genres", total_synth, len(synthetic_by_genre))

    # Silver-label real texts
    if not texts and not synthetic_by_genre:
        logger.error("No texts found and augmentation disabled -- nothing to export")
        return 0

    rows: list[dict] = []

    if texts:
        preds = _classify_batch(texts, fast=fast)
        for text, (genre, conf) in zip(texts, preds):
            rows.append({
                "text": text,
                "predicted_genre": genre,
                "confidence": round(conf, 4),
                "true_genre": "",
            })

    # Synthetic rows: genre is known by construction
    for genre, synth_texts in synthetic_by_genre.items():
        for t in synth_texts:
            rows.append({
                "text": t,
                "predicted_genre": genre,
                "confidence": 1.0,
                "true_genre": genre,  # synthetic labels pre-filled; review still recommended
            })
        logger.info("  Added %d synthetic rows for genre '%s'", len(synth_texts), genre)

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    real_count = len(texts)
    synth_count = sum(len(v) for v in synthetic_by_genre.values())
    logger.info("Wrote %d rows to %s (%d real, %d synthetic)", len(rows), out_path, real_count, synth_count)

    # Summary
    from nlp_pipeline import GENRE_LABELS
    print("=" * 60)
    print("  GENRE TRAINING DATA EXPORT")
    print("=" * 60)
    print(f"  Total rows:    {len(rows)}")
    print(f"  Real texts:    {real_count} (true_genre = EMPTY -- needs human review)")
    print(f"  Synthetic:     {synth_count} (true_genre = pre-filled by template)")
    print(f"  Output file:   {out_path}")
    print()
    print("  NEXT STEP:")
    print(f"  Open {out_path} and fill in 'true_genre' for all real-text rows.")
    print("  Valid labels:", ", ".join(GENRE_LABELS))
    print("  Then run: python train_genre_classifier.py")
    print("=" * 60)

    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap genre classifier training data from AIR corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=SQLITE_DB_PATH, help="Path to news_store.db")
    parser.add_argument("--text-dir", default=TEXT_DIR, help="Path to transcript text directory")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--augment", action="store_true",
                        help="Also generate synthetic sentences using GENRE_KEYWORD_MAP templates")
    parser.add_argument("--slow", action="store_true",
                        help="Use the heavy bart-large-mnli classifier instead of fast model")
    args = parser.parse_args()

    n = build(
        db_path=args.db,
        text_dir=args.text_dir,
        out_path=args.out,
        augment=args.augment,
        fast=not args.slow,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
