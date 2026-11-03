"""
taxonomy.py - the one place that defines genres and broadcast types.

Genres
  Every indexed document (article or bulletin chunk) gets a primary `genre_id` from GENRES
  plus up to three secondary tags. Classification is deterministic and cheap (no LLM):

    1. a reliable publisher category (NewsOnAIR "Sports", "Business", ...) wins, refined by
       keywords where the publisher category is broad (Business -> Finance & Markets);
    2. otherwise keyword rules over the headline (weight 3) and the opening text (weight 1);
    3. otherwise the existing zero-shot model label, when its confidence is high enough;
    4. otherwise the publisher's National / Regional desk;
    5. otherwise "other" (Other / Unclassified).

  The original publisher category stays in `source_category` and the original model label
  in `genre`, so nothing is lost and the taxonomy can change without re-embedding: run
  `python taxonomy.py --backfill` to recompute the metadata columns in place.

Broadcast types
  Derived from the bulletin title published by NewsOnAIR ("Midday News", "News At Nine", ...).
  A title that matches no known programme is "other" (Other / Unclassified), never guessed.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Optional

import config

logger = logging.getLogger("taxonomy")

# id -> display name. Order is the display order. Only genres that occur in the indexed data
# are offered as filters (see genre_counts()).
GENRES: dict[str, str] = {
    "national": "National News",
    "international": "International News",
    "politics": "Politics & Governance",
    "business": "Business & Economy",
    "finance": "Finance & Markets",
    "technology": "Technology & AI",
    "science": "Science & Research",
    "health": "Health & Medicine",
    "education": "Education",
    "environment": "Environment & Climate",
    "agriculture": "Agriculture & Rural Development",
    "infrastructure": "Infrastructure & Transport",
    "defence": "Defence & Security",
    "sports": "Sports",
    "culture": "Entertainment & Culture",
    "regional": "Regional News",
    "social": "Social Issues",
    "other": "Other / Unclassified",
}
OTHER = "other"

# Publisher (NewsOnAIR) categories that reliably name a genre.
SOURCE_CATEGORY_GENRE = {
    "sports": "sports", "business": "business", "international": "international", "health": "health",
    "science and technology": "science", "science & technology": "science", "entertainment": "culture",
    "elections": "politics", "weather": "environment",
}
# Broad publisher categories that keywords may refine to a more specific genre.
REFINE = {"business": {"finance"}, "science": {"technology"}, "international": {"defence"}}
# Publisher desks used only when nothing more specific is found.
DESK_GENRE = {"national": "national", "regional": "regional"}

# Zero-shot model labels (nlp_pipeline.GENRE_LABELS) -> taxonomy.
MODEL_LABEL_GENRE = {
    "Politics": "politics", "Economy": "business", "Sports": "sports", "Science & Technology": "science",
    "International": "international", "Defence & Security": "defence", "Health": "health",
    "Environment": "environment", "Society & Culture": "culture",
}
MODEL_MIN_CONFIDENCE = 0.5

# Keyword rules. Terms are matched as whole words, case-insensitively; a trailing * matches a word prefix.
KEYWORDS: dict[str, list[str]] = {
    "politics": ["parliament", "lok sabha", "rajya sabha", "cabinet", "election*", "poll", "polls", "polling", "bjp", "congress party",
                 "chief minister", "governor", "assembly", "legislat*", "ordinance", "constitution*", "vote*",
                 "voter*", "electoral", "bill passed", "opposition", "party president", "manifesto", "governance",
                 "bypoll*", "president murmu", "vice president"],
    "business": ["economy", "economic", "gdp", "trade", "export*", "import*", "industry", "industrial", "msme*",
                 "investment*", "investor*", "manufactur*", "gst", "tax", "taxes", "taxation", "budget", "fiscal", "inflation",
                 "startup*", "company", "companies", "revenue", "free trade", "commerce", "economic partnership",
                 "fdi", "tariff*"],
    "finance": ["sensex", "nifty", "stock market*", "share market", "rbi", "reserve bank", "repo rate",
                "interest rate*", "rupee", "forex", "bank*", "loan*", "credit", "mutual fund*", "sebi", "ipo*",
                "insurance", "pension*", "bond*", "financial inclusion", "jan dhan", "upi", "savings"],
    "technology": ["artificial intelligence", "ai", "digital", "semiconductor*", "software", "cyber*", "internet",
                   "5g", "6g", "data centre*", "data center*", "it sector", "startup india", "electronics",
                   "technolog*", "quantum", "machine learning"],
    "science": ["isro", "space", "satellite*", "launch vehicle", "rocket", "research*", "scientist*", "drdo test",
                "gaganyaan", "chandrayaan", "astronom*", "laborator*", "csir", "nasa"],
    "health": ["health", "hospital*", "disease*", "vaccin*", "medical", "medicine*", "doctor*", "patient*",
               "ayushman", "cancer", "tuberculosis", "tb", "covid*", "dengue", "malaria", "drug*", "pharma*",
               "nutrition", "aiims", "ayush", "cdsco", "mental health", "world health organization"],
    "education": ["education", "school*", "student*", "university", "universities", "college*", "exam", "exams", "examination*",
                  "teacher*", "cbse", "ugc", "iit*", "neet", "jee", "scholarship*", "literacy", "nep",
                  "board exam*", "admission*", "curriculum", "skill*"],
    "environment": ["climate", "environment*", "pollution", "air quality", "aqi", "forest*", "wildlife",
                    "tiger*", "cyclone*", "monsoon", "rainfall", "imd", "heatwave", "flood*", "earthquake",
                    "renewable", "solar", "emission*", "biodiversity", "weather", "landslide*", "rain*",
                    "cop30", "net zero", "green energy", "conservation"],
    "agriculture": ["farmer*", "agricultur*", "crop*", "kharif", "rabi", "msp", "sugarcane", "paddy", "wheat",
                    "irrigation", "fertili*", "rural", "kisan", "dairy", "fisher*", "horticulture", "procurement",
                    "pm-kisan", "village*", "panchayat*", "gram"],
    "infrastructure": ["railway*", "train*", "metro", "highway*", "road", "roads", "airport*", "port", "ports", "bridge*",
                       "infrastructure", "vande bharat", "namo bharat", "expressway", "aviation", "flight*",
                       "shipping", "connectivity", "transport*", "rail", "corridor", "tunnel"],
    "defence": ["army", "navy", "air force", "iaf", "defence", "defense", "military", "missile*", "terror*",
                "security forces", "bsf", "crpf", "naxal*", "maoist*", "soldier*", "border", "drdo", "warship",
                "operation sindoor", "militant*", "encounter", "jawans", "military exercise", "joint exercise", "coast guard"],
    "sports": ["cricket", "football", "hockey", "olympic*", "medal*", "tournament", "championship*", "world cup",
               "match", "athlete*", "badminton", "tennis", "chess", "wrestl*", "boxing", "shooting", "bcci",
               "ipl", "t20", "odi", "test match", "khelo india", "grandmaster", "kabaddi", "squad", "wicket*"],
    "culture": ["festival*", "film*", "cinema", "music", "artist*", "arts", "culture", "cultural", "heritage", "dance",
                "durga puja", "diwali", "chhath", "navratri", "dussehra", "book fair", "museum", "award*",
                "yoga", "tourism", "temple*", "literature", "actor*", "theatre", "birth anniversary", "jayanti"],
    "social": ["women", "child*", "welfare", "poverty", "caste", "tribal", "disabilit*", "divyang*", "gender",
               "dowry", "human trafficking", "senior citizen*", "migrant*", "accident*", "stampede", "crime*",
               "police", "court", "housing", "slum*", "minority", "self-help group*", "census", "domestic violence"],
    "international": ["united nations", "un", "bilateral", "foreign minister", "external affairs", "diplomat*",
                      "summit", "g20", "brics", "quad", "sco", "asean", "embassy", "visit to", "prime minister of",
                      "president of", "united states", "us president", "china", "pakistan", "russia", "ukraine", "israel", "gaza", "iran",
                      "nepal", "bangladesh", "sri lanka", "japan", "canada", "uk", "europe*", "african"],
}
# When two genres tie, the earlier one in this list wins (more specific before broader).
PRIORITY = ["sports", "finance", "technology", "science", "health", "education", "agriculture", "infrastructure",
            "defence", "environment", "culture", "social", "politics", "business", "international"]
MIN_SCORE = 3          # one headline hit, or three hits in the opening text
MIN_TAG_SCORE = 3
MAX_TAGS = 3


def _compile(terms: list[str]) -> re.Pattern:
    parts = []
    for t in terms:
        t = t.lower()
        if t.endswith("*"):
            parts.append(r"\b" + re.escape(t[:-1]) + r"\w*")
        else:
            parts.append(r"\b" + re.escape(t) + r"\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_PATTERNS = {g: _compile(ts) for g, ts in KEYWORDS.items()}


def genre_name(genre_id: Optional[str]) -> str:
    return GENRES.get(genre_id or OTHER, GENRES[OTHER])


def genre_id_of(value: Optional[str]) -> Optional[str]:
    """Taxonomy id for an id or a display name ('sports', 'Sports', 'Business & Economy'); None if unknown."""
    if not value:
        return None
    v = value.strip().lower()
    if v in GENRES:
        return v
    for gid, name in GENRES.items():
        if name.lower() == v:
            return gid
    return None


def keyword_scores(headline: str, text: str, body_words: int = 120) -> dict[str, int]:
    opening = " ".join((text or "").split()[:body_words])
    scores: dict[str, int] = {}
    for g, pat in _PATTERNS.items():
        s = 3 * len(pat.findall(headline or "")) + len(pat.findall(opening))
        if s:
            scores[g] = s
    return scores


def _leader(scores: dict[str, int], minimum: int, strict: bool = False) -> Optional[str]:
    """Best-scoring genre at or above `minimum`; ties go to PRIORITY order, or to nobody when strict."""
    good = {g: s for g, s in scores.items() if s >= minimum}
    if not good:
        return None
    best = max(good.values())
    tied = [g for g, s in good.items() if s == best]
    if strict and len(tied) > 1:
        return None
    return min(tied, key=lambda g: PRIORITY.index(g) if g in PRIORITY else len(PRIORITY))


def classify(headline: str = "", text: str = "", source_category: Optional[str] = None,
             model_label: Optional[str] = None, model_confidence: Optional[float] = None,
             content_type: str = "article") -> tuple[str, list[str], str]:
    """Return (genre_id, secondary tag ids, rule) for one document or chunk.

    `rule` says which step decided ("publisher", "keywords", "model", "desk", "unclassified").
    For bulletin chunks the publisher category is the programme title, which is not a genre.
    """
    scores = keyword_scores(headline, text)
    cat = (source_category or "").strip().lower() if content_type == "article" else ""
    genre, rule = None, "unclassified"
    if cat in SOURCE_CATEGORY_GENRE:
        genre, rule = SOURCE_CATEGORY_GENRE[cat], "publisher"
        for finer in REFINE.get(genre, ()):
            if scores.get(finer, 0) >= MIN_SCORE and scores.get(finer, 0) > scores.get(genre, 0):
                genre = finer
    if genre is None:
        # A bulletin chunk is one or two sentences: one unambiguous topic word decides it.
        lead = (_leader(scores, MIN_SCORE) if content_type == "article"
                else _leader(scores, 1, strict=True))
        if lead:
            genre, rule = lead, "keywords"
    model_genre = MODEL_LABEL_GENRE.get(model_label or "")
    confident = model_genre and (model_confidence is None or model_confidence >= MODEL_MIN_CONFIDENCE)
    if genre is None and confident:
        genre, rule = model_genre, "model"
    if genre is None and cat in DESK_GENRE:
        genre, rule = DESK_GENRE[cat], "desk"
    genre = genre or OTHER
    tags = [g for g, s in sorted(scores.items(), key=lambda x: -x[1]) if s >= MIN_TAG_SCORE and g != genre]
    if confident and model_genre != genre and model_genre not in tags:
        tags.append(model_genre)
    if cat in DESK_GENRE and DESK_GENRE[cat] != genre and DESK_GENRE[cat] not in tags:
        tags.append(DESK_GENRE[cat])
    return genre, tags[:MAX_TAGS], rule


def encode_tags(tags: list[str]) -> Optional[str]:
    """Stored as ',a,b,' so a LIKE '%,a,%' filter cannot match a substring of another id."""
    return ("," + ",".join(tags) + ",") if tags else None


def decode_tags(value: Optional[str]) -> list[str]:
    return [t for t in (value or "").split(",") if t]


# ---------------------------------------------------------------------------
# Broadcast types
# ---------------------------------------------------------------------------

# id -> (display name, NewsOnAIR titles that belong to it)
BROADCAST_TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "midday": ("Midday News", ("midday news", "mid-day news", "midday")),
    "news-at-nine": ("News at Nine", ("news at nine", "news at 9")),
    "hourly": ("Hourly News", ("hourly news", "hourly bulletin")),
    "morning": ("Morning News", ("morning news",)),
    "evening": ("Evening News", ("evening news",)),
    "other": ("Other / Unclassified", ()),
}


def broadcast_type_of(title: Optional[str]) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    for tid, (_, titles) in BROADCAST_TYPES.items():
        if t in titles:
            return tid
    return "other"


def broadcast_type_name(type_id: Optional[str]) -> str:
    return BROADCAST_TYPES.get(type_id or "other", BROADCAST_TYPES["other"])[0]


def broadcast_type_ids(values: Optional[list[str]]) -> list[str]:
    """Validate a list of type ids / names; raises ValueError naming any unknown value."""
    out, bad = [], []
    for v in values or []:
        v = (v or "").strip()
        if not v:
            continue
        tid = v.lower() if v.lower() in BROADCAST_TYPES else next(
            (k for k, (name, titles) in BROADCAST_TYPES.items() if v.lower() == name.lower() or v.lower() in titles),
            None)
        (out if tid else bad).append(tid or v)
    if bad:
        raise ValueError(f"Unknown broadcast type(s): {', '.join(bad)}. Known: {', '.join(BROADCAST_TYPES)}")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Backfill: recompute metadata in place (no re-embedding)
# ---------------------------------------------------------------------------

def backfill(db_path: Optional[str] = None) -> dict:
    """Assign genre_id / genre_tags / broadcast_type to every segment and article row.

    Articles are classified once per document (headline + full text) and every chunk inherits
    the result; bulletin chunks are classified one by one. Only metadata columns change.
    """
    import vector_store
    db = vector_store._get_db(db_path)
    stats = {"articles": 0, "transcript_chunks": 0, "rules": {}}
    try:
        arts = db.execute("SELECT article_id, headline, full_text, genre, data_json FROM articles").fetchall()
        seg_meta = {r[0]: (r[1], r[2]) for r in db.execute(
            "SELECT doc_id, MAX(source_category), MAX(genre_confidence) FROM segments WHERE content_type = 'article' "
            "GROUP BY doc_id")}
        art_rows, seg_rows = [], []
        for a in arts:
            cat, conf = seg_meta.get(a["article_id"], (None, None))
            gid, tags, rule = classify(a["headline"], a["full_text"], cat, a["genre"], conf, "article")
            stats["rules"][rule] = stats["rules"].get(rule, 0) + 1
            art_rows.append((gid, encode_tags(tags), cat, a["article_id"]))
            seg_rows.append((gid, encode_tags(tags), a["article_id"]))
        db.executemany("UPDATE articles SET genre_id = ?, genre_tags = ?, source_category = ? WHERE article_id = ?",
                       art_rows)
        db.executemany("UPDATE segments SET genre_id = ?, genre_tags = ? WHERE doc_id = ? AND content_type = 'article'",
                       seg_rows)
        stats["articles"] = len(art_rows)
        rows = db.execute("SELECT faiss_id, text, genre, genre_confidence, headline FROM segments "
                          "WHERE IFNULL(content_type, '') != 'article'").fetchall()
        upd = []
        for r in rows:
            gid, tags, rule = classify("", r["text"], None, r["genre"], r["genre_confidence"], "audio_transcript")
            upd.append((gid, encode_tags(tags), r["faiss_id"]))
        db.executemany("UPDATE segments SET genre_id = ?, genre_tags = ? WHERE faiss_id = ?", upd)
        stats["transcript_chunks"] = len(upd)
        db.execute("UPDATE segments SET broadcast_type = NULL WHERE content_type = 'article'")
        for cat, in db.execute("SELECT DISTINCT source_category FROM segments "
                               "WHERE IFNULL(content_type, '') != 'article'").fetchall():
            db.execute("UPDATE segments SET broadcast_type = ? WHERE IFNULL(content_type, '') != 'article' AND "
                       "IFNULL(source_category, '') = IFNULL(?, '')", (broadcast_type_of(cat), cat))
        db.commit()
    finally:
        db.close()
    stats["broadcasts"] = vector_store.sync_broadcasts(db_path=db_path)
    stats["genres"] = genre_counts(db_path)
    import analytics
    analytics.invalidate()
    logger.info("Taxonomy backfill: %d articles, %d transcript chunks, %d broadcasts | rules %s",
                stats["articles"], stats["transcript_chunks"], stats["broadcasts"], stats["rules"])
    return stats


def genre_counts(db_path: Optional[str] = None) -> dict[str, dict]:
    """genre_id -> {name, articles, broadcasts, chunks} over active indexed rows (genres with data only)."""
    import vector_store
    db = vector_store._get_db(db_path)
    try:
        rows = db.execute(
            f"SELECT IFNULL(genre_id, 'other') g, COUNT(DISTINCT CASE WHEN content_type = 'article' THEN doc_id END), "
            f"COUNT(DISTINCT CASE WHEN content_type != 'article' OR content_type IS NULL THEN source_file END), COUNT(*) "
            f"FROM segments WHERE {vector_store.ACTIVE} GROUP BY g").fetchall()
    finally:
        db.close()
    found = {r[0]: {"name": genre_name(r[0]), "articles": r[1], "broadcasts": r[2], "chunks": r[3]} for r in rows}
    return {g: found[g] for g in GENRES if g in found}


def main() -> int:
    ap = argparse.ArgumentParser(description="Genre taxonomy and broadcast types")
    ap.add_argument("--backfill", action="store_true", help="Recompute genre/broadcast-type metadata in place")
    ap.add_argument("--counts", action="store_true", help="Print genre counts")
    a = ap.parse_args()
    import json
    if a.backfill:
        print(json.dumps(backfill(), indent=2))
    elif a.counts:
        print(json.dumps(genre_counts(), indent=2))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    _ = config
    raise SystemExit(main())
