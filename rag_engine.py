"""
rag_engine.py - Grounded, traceable RAG over the VaniStream corpus.

Pipeline for one question (each step is logged under the request's trace ID):

  1. Query understanding  - date range ("yesterday", "on 20 September",
     "last week"), time of day ("around 7 PM", "between 6 and 8 PM"),
     explicit genre ("sports news"), "latest", and the remaining topic words.
     Relative dates are anchored on the real current date in IST
     (VANI_DATE_ANCHOR=corpus anchors on the newest corpus date instead).
  2. Metadata prefilter   - SQLite narrows candidates by date/hour/genre.
  3. Retrieval
       topical question  -> hybrid: dense FAISS cosine + BM25 keyword (FTS5),
                            fused with Reciprocal Rank Fusion, then a
                            relevance gate so unrelated chunks are dropped
       "what happened yesterday?" (no topic) -> chronological sample of the
                            filtered period instead of a meaningless vector search
  4. Context assembly     - each broadcast hit is expanded with its neighbouring
     sentences from the same broadcast (sentence-window chunking); article
     hits are merged per article and carry the COMPLETE article (all sections)
     as context. Sources are numbered [S1], [S2] ... with source name,
     headline, date, time and audio range.
  5. Generation           - the question is typed (simple / standard / explain /
     overview / compare / gap), which sets the target length (a few sentences
     up to ~300-600 words) and the section structure. Gemini answers ONLY from
     the context and must cite [S#]; citations are validated afterwards
     (unknown ids are removed). Without Gemini, a structured extractive answer
     is assembled from cited article sections and broadcast quotes.
  6. Insufficient context - if nothing relevant is found the answer says so
     and reports what the corpus does cover. Nothing is invented.
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import config
from tracing import current_trace_id, fmt_ts

logger = logging.getLogger("rag")

# Genre keyword mapping for Phase 1 (word-boundary matched)
GENRE_KEYWORD_MAP = {
    "cricket": "Sports", "football": "Sports", "hockey": "Sports",
    "tennis": "Sports", "olympics": "Sports", "sports": "Sports",
    "sport": "Sports",
    "match": "Sports", "tournament": "Sports", "athlete": "Sports",
    "parliament": "Politics", "election": "Politics", "minister": "Politics",
    "modi": "Politics", "bjp": "Politics", "congress": "Politics",
    "political": "Politics", "politics": "Politics", "vote": "Politics",
    "lok sabha": "Politics", "rajya sabha": "Politics", "bill": "Politics",
    "economy": "Economy", "gdp": "Economy", "inflation": "Economy",
    "rbi": "Economy", "rupee": "Economy", "stock": "Economy",
    "market": "Economy", "budget": "Economy", "trade": "Economy",
    "fiscal": "Economy", "tax": "Economy", "economic": "Economy",
    "science": "Science & Technology", "technology": "Science & Technology",
    "isro": "Science & Technology", "space": "Science & Technology",
    "artificial intelligence": "Science & Technology", "digital": "Science & Technology",
    "research": "Science & Technology", "satellite": "Science & Technology",
    "china": "International", "pakistan": "International",
    "america": "International", "russia": "International",
    "united nations": "International", "united states": "International",
    "international": "International", "global": "International",
    "foreign": "International", "diplomat": "International",
    "army": "Defence & Security", "navy": "Defence & Security",
    "military": "Defence & Security", "defence": "Defence & Security",
    "defense": "Defence & Security", "security": "Defence & Security",
    "terror": "Defence & Security", "border": "Defence & Security",
    "health": "Health", "covid": "Health", "hospital": "Health",
    "disease": "Health", "vaccine": "Health", "medical": "Health",
    "doctor": "Health", "pandemic": "Health",
    "environment": "Environment", "climate": "Environment",
    "pollution": "Environment", "forest": "Environment",
    "wildlife": "Environment", "green": "Environment", "flood": "Environment",
    "drought": "Environment", "cyclone": "Environment", "weather": "Environment",
    "culture": "Society & Culture", "festival": "Society & Culture",
    "education": "Society & Culture", "society": "Society & Culture",
    "women": "Society & Culture", "caste": "Society & Culture",
    "religion": "Society & Culture",
}

# Words that name a category explicitly -> used as a hard SQLite filter.
# Topic words such as "minister" or "cricket" are NOT hard filters (a story can be
# classified under a neighbouring genre); they only boost ranking.
EXPLICIT_GENRE_TERMS = {
    "sports": "Sports", "sport": "Sports", "sporting": "Sports",
    "politics": "Politics", "political": "Politics",
    "economy": "Economy", "economic": "Economy", "business": "Economy", "financial": "Economy",
    "science": "Science & Technology", "technology": "Science & Technology", "tech": "Science & Technology",
    "international": "International", "world": "International", "global": "International",
    "defence": "Defence & Security", "defense": "Defence & Security", "security": "Defence & Security",
    "health": "Health", "environment": "Environment", "environmental": "Environment",
    "weather": "Environment", "climate": "Environment",
    "culture": "Society & Culture", "cultural": "Society & Culture", "society": "Society & Culture",
}

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

FILLER = set("""what whats happened happening happen news tell me give show about any the a an of on in at for
to from between and or is was were are did do does there story stories stories' bulletin bulletins
headlines headline updates update report reports major main key top big important please summarise
summarize summary overview day days week weeks month today yesterday tonight morning afternoon evening
night around latest recent recently last past this that which who how all some anything everything
can you i us we say said around hrs hours hour am pm ist by with into
explain describe elaborate compare comparison contrast summarise summarize detail detailed details developments
development background context implications implication consequences why missing information know
two three both newsonair akashvani air reported report reports say says""".split())


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------

@dataclass
class QueryFilters:
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    hour_from: Optional[int] = None   # HHMM
    hour_to: Optional[int] = None     # HHMM
    genre: Optional[str] = None       # hard filter (explicit category)
    genre_hint: Optional[str] = None  # soft boost (topic keyword)
    latest: bool = False
    topic: str = ""                   # query with dates/times/filler removed
    source_category: Optional[str] = None  # publisher category, e.g. NewsOnAIR "National" / "Regional"

    def describe(self) -> str:
        parts = []
        if self.date_from or self.date_to:
            parts.append(f"date={self.date_from}..{self.date_to}")
        if self.hour_from is not None:
            parts.append(f"time={self.hour_from:04d}-{self.hour_to:04d}")
        if self.genre:
            parts.append(f"genre={self.genre}")
        if self.source_category:
            parts.append(f"source_category={self.source_category}")
        if self.genre_hint and not self.genre:
            parts.append(f"genre_hint={self.genre_hint}")
        if self.latest:
            parts.append("latest=yes")
        parts.append(f"topic='{self.topic}'" if self.topic else "topic=<none>")
        return ", ".join(parts)


def _anchor_today() -> datetime.date:
    if config.DATE_ANCHOR == "corpus":
        import vector_store
        _, max_d = vector_store.get_date_range()
        if max_d:
            try:
                return datetime.date.fromisoformat(max_d)
            except ValueError:
                pass
    return config.now_local().date()


# Kept for scripts that imported the old helper name.
def _get_corpus_anchor() -> datetime.date:
    return _anchor_today()


def _mk_date(year: int, month: int, day: int, today: datetime.date) -> Optional[datetime.date]:
    try:
        d = datetime.date(year, month, day)
    except ValueError:
        return None
    return d


def _parse_date_range(query: str, today: Optional[datetime.date] = None) -> tuple[Optional[str], Optional[str]]:
    """Extract an inclusive (date_from, date_to) from natural language, or (None, None)."""
    today = today or _anchor_today()
    q = query.lower()

    m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", q)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
    if m:
        return m.group(1), m.group(1)

    def explicit_year(y):
        return int(y) if y else None

    # "20 September 2026", "20th Sept", "on 5 May"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RE + r"\b,?\s*(\d{4})?", q)
    if not m:
        # "September 20, 2026", "Sept 20th"
        m2 = re.search(r"\b" + _MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b,?\s*(\d{4})?", q)
        if m2:
            month, day, year = m2.group(1), m2.group(2), m2.group(3)
        else:
            month = day = year = None
    else:
        day, month, year = m.group(1), m.group(2), m.group(3)
    if month and day:
        mi = MONTHS.get(month[:4] if month.startswith("sept") else month[:3]) or MONTHS.get(month)
        y = explicit_year(year) or today.year
        d = _mk_date(y, mi, int(day), today)
        if d and not year and d > today:
            d = _mk_date(y - 1, mi, int(day), today)  # "20 December" asked in January -> last year
        if d:
            return d.isoformat(), d.isoformat()

    if "day before yesterday" in q:
        d = today - datetime.timedelta(days=2)
        return d.isoformat(), d.isoformat()
    m = re.search(r"(?:last|past)\s+(\d+)\s+days?", q)
    if m:
        return (today - datetime.timedelta(days=int(m.group(1)))).isoformat(), today.isoformat()
    if re.search(r"\b(last|past)\s+week\b", q):
        return (today - datetime.timedelta(days=7)).isoformat(), today.isoformat()
    if re.search(r"\bthis\s+week\b", q):
        return (today - datetime.timedelta(days=today.weekday())).isoformat(), today.isoformat()
    if re.search(r"\b(last|past)\s+month\b", q):
        return (today - datetime.timedelta(days=30)).isoformat(), today.isoformat()
    if "yesterday" in q:
        d = today - datetime.timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if re.search(r"\b(today|tonight|this morning|this evening|this afternoon)\b", q):
        return today.isoformat(), today.isoformat()
    return None, None


def _to_24h(h: int, mer: Optional[str]) -> int:
    if mer:
        mer = mer.replace(".", "")
        if mer == "pm" and h < 12:
            h += 12
        if mer == "am" and h == 12:
            h = 0
    return h


def _parse_time_range(query: str) -> tuple[Optional[int], Optional[int]]:
    """Extract an HHMM range: 'around 7 PM', 'at 10:30 am', 'between 6 PM and 8 PM', 'evening'."""
    q = query.lower()
    t = r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?"
    m = re.search(r"(?:between|from)\s+" + t + r"\s+(?:and|to|-)\s+" + t, q)
    if m and (m.group(3) or m.group(6) or m.group(2) or m.group(5)):
        mer2 = m.group(6)
        mer1 = m.group(3) or mer2
        h1, h2 = _to_24h(int(m.group(1)), mer1), _to_24h(int(m.group(4)), mer2)
        if h1 <= 23 and h2 <= 23:
            return h1 * 100 + int(m.group(2) or 0), h2 * 100 + int(m.group(5) or 59)
    m = re.search(r"\b([01]\d|2[0-3])([0-5]\d)\s*(?:hrs|hours)\b", q)  # "1900 hrs"
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        return h * 100 + mi, h * 100 + 59
    for m in re.finditer(r"\b(around|about|at|near|by)?\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?|hrs|hours)\b"
                         r"|\b(around|about|at|near)\s+(\d{1,2}):(\d{2})\b", q):
        if m.group(2):
            prep, h, mi, mer = m.group(1), int(m.group(2)), int(m.group(3) or 0), m.group(4)
            mer = mer if mer and mer.startswith(("a", "p")) else None
        else:
            prep, h, mi, mer = m.group(5), int(m.group(6)), int(m.group(7)), None
        h = _to_24h(h, mer)
        if h > 23 or mi > 59:
            continue
        center = h * 60 + mi
        if prep in ("around", "about", "near"):
            lo, hi = max(0, center - 30), min(23 * 60 + 59, center + 59)
        else:
            lo, hi = center, min(23 * 60 + 59, center + 59)
        return (lo // 60) * 100 + lo % 60, (hi // 60) * 100 + hi % 60
    # A part-of-day word is a time filter only when used as a time ("this evening", "in the
    # morning", "evening news"), not as a topic word ("evening power demand", "morning walk").
    for word, rng in (("morning", (500, 1159)), ("afternoon", (1200, 1659)),
                      ("evening", (1700, 2059)), ("night", (2100, 2359)), ("tonight", (1800, 2359))):
        as_time = (r"\b(?:this|yesterday|today|tomorrow|last|in the|during the|on the|that|every|each)\s+" + word
                   + r"\b|\b" + word + r"\s+" + NEWS_NOUNS + r"\b|\b" + word + r"\s*\??\s*$")
        if word == "tonight" and re.search(r"\btonight\b", q) or re.search(as_time, q):
            return rng
    return None, None


def _extract_genre(query: str) -> Optional[str]:
    """Extract genre from query using word-boundary keyword matching.

    Returns None if no keywords match or if there is a tie between genres.
    """
    q = query.lower().strip()

    # Direct match for canonical genre names
    from nlp_pipeline import GENRE_LABELS
    for g in GENRE_LABELS:
        # Check exact string or word boundary match for genre name
        if q == g.lower():
            return g
        pattern = r'\b' + re.escape(g.lower()) + r'\b'
        if re.search(pattern, q):
            return g

    genre_scores: dict[str, int] = {}
    for keyword, genre in GENRE_KEYWORD_MAP.items():
        # Use word boundaries to avoid substring false positives
        pattern = r'\b' + re.escape(keyword) + r'\b'
        hits = len(re.findall(pattern, q))
        if hits > 0:
            genre_scores[genre] = genre_scores.get(genre, 0) + hits
    if not genre_scores:
        return None
    # Return None on a tie instead of guessing
    top_score = max(genre_scores.values())
    top_genres = [g for g, s in genre_scores.items() if s == top_score]
    if len(top_genres) > 1:
        return None
    return top_genres[0]


NEWS_NOUNS = r"(news|stories|story|headlines|headline|updates|update|developments|coverage|bulletins?|items|reports?)"


def _explicit_genre(query: str) -> Optional[str]:
    """Genre named as the subject of the question -> hard SQLite filter.

    'sports news', 'economic stories', 'what happened in politics yesterday?'
    qualify. A category word inside a topic ('T20 World Cup', 'defence
    forces') does not: it stays a soft ranking hint, so stories are not lost
    to a hard filter.
    """
    q = query.lower()
    rest_empty = lambda term: not _topic_terms(re.sub(r"\b" + re.escape(term) + r"\b", " ", q), None)  # noqa: E731
    from nlp_pipeline import GENRE_LABELS
    for g in GENRE_LABELS:
        gl = g.lower()
        if g != "Other" and re.search(r"\b" + re.escape(gl) + r"\b", q):
            if re.search(r"\b" + re.escape(gl) + r"\b\s+" + NEWS_NOUNS, q) or rest_empty(gl):
                return g
    hits = set()
    for term, genre in EXPLICIT_GENRE_TERMS.items():
        if re.search(r"\b" + term + r"\b", q):
            if re.search(r"\b" + term + r"\s+" + NEWS_NOUNS + r"\b", q) or rest_empty(term):
                hits.add(genre)
    return hits.pop() if len(hits) == 1 else None


def _topic_terms(query: str, genre: Optional[str]) -> str:
    q = query.lower().replace("\u2019", "'")
    q = re.sub(r"'s\b", "", q)
    q = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", q)
    q = re.sub(r"\b\d+(st|nd|rd|th)\b", " ", q)
    q = re.sub(r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?|hrs|hours)?\b", " ", q)
    q = re.sub(r"\b" + _MONTH_RE + r"\b", " ", q)
    words = re.findall(r"[a-zऀ-෿][a-zऀ-෿'-]*", q)
    drop = set(FILLER) | set(EXPLICIT_GENRE_TERMS if genre else ())
    if genre:
        drop |= set(genre.lower().replace("&", " ").split())
    return " ".join(w for w in words if w not in drop and len(w) > 1)


# Publisher categories that are not topics: "national news" means NewsOnAIR's National
# section, whatever the story's genre. Used as a filter only when named as the news type.
SOURCE_CATEGORY_TERMS = {"national": "National", "regional": "Regional"}


def _source_category(query: str) -> Optional[str]:
    q = query.lower()
    for term, cat in SOURCE_CATEGORY_TERMS.items():
        if re.search(r"\b" + term + r"\s+" + NEWS_NOUNS + r"\b", q) or re.search(r"\bin\s+" + term + r"\b", q):
            return cat
    return None


def parse_query(query: str, today: Optional[datetime.date] = None) -> QueryFilters:
    today = today or _anchor_today()
    d_from, d_to = _parse_date_range(query, today)
    h_from, h_to = _parse_time_range(query)
    genre = _explicit_genre(query)
    source_category = _source_category(query)
    topic = _topic_terms(query, genre)
    if source_category:
        topic = " ".join(w for w in topic.split() if w not in SOURCE_CATEGORY_TERMS)
    return QueryFilters(
        date_from=d_from, date_to=d_to, hour_from=h_from, hour_to=h_to, genre=genre,
        genre_hint=None if genre else _extract_genre(query),
        latest=bool(re.search(r"\b(latest|recent|recently|newest|current)\b", query.lower())),
        topic=topic, source_category=source_category,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RRF_K = 60


def _hybrid_search(query: str, candidate_ids: Optional[list[int]], f: QueryFilters) -> list[dict]:
    """Dense + BM25 retrieval fused with RRF, then gated for relevance."""
    import vector_store
    from nlp_pipeline import embed_sentences

    k = config.RAG_CANDIDATES
    t0 = time.perf_counter()
    q_emb = embed_sentences([query])[0]
    dense = vector_store.semantic_search(q_emb, candidate_ids, top_k=k)
    sparse = vector_store.keyword_search(f.topic or query, candidate_ids, top_k=k)
    logger.info("Retrieval: dense=%d hits (top cos=%.3f), keyword/BM25=%d hits (%.2fs)",
                len(dense), dense[0]["score"] if dense else 0.0, len(sparse), time.perf_counter() - t0)

    by_id: dict[int, dict] = {}
    for rank, r in enumerate(dense):
        r = dict(r, dense_rank=rank + 1, bm25=0.0, rrf=1.0 / (RRF_K + rank + 1))
        by_id[r["faiss_id"]] = r
    kw_only = []
    for rank, s in enumerate(sparse):
        if s["faiss_id"] in by_id:
            by_id[s["faiss_id"]]["bm25"] = s["bm25"]
            by_id[s["faiss_id"]]["rrf"] += 1.0 / (RRF_K + rank + 1)
        else:
            kw_only.append((rank, s))
    if kw_only:  # get cosine scores for keyword-only hits so they pass the same gate
        extra = vector_store.semantic_search(q_emb, [s["faiss_id"] for _, s in kw_only], top_k=len(kw_only))
        extra_by = {e["faiss_id"]: e for e in extra}
        for rank, s in kw_only:
            if s["faiss_id"] in extra_by:
                r = dict(extra_by[s["faiss_id"]], dense_rank=None, bm25=s["bm25"], rrf=1.0 / (RRF_K + rank + 1))
                by_id[s["faiss_id"]] = r

    if not by_id:
        return []
    top_dense = max(r["score"] for r in by_id.values())
    if top_dense < config.RAG_MIN_TOP_SCORE:
        logger.info("Relevance gate: best match cos=%.3f is below %.2f -> no relevant source",
                    top_dense, config.RAG_MIN_TOP_SCORE)
        return []
    if top_dense < config.RAG_STRONG_MATCH:
        # A moderate semantic match is not enough when a specific topic word ("bitcoin") occurs in no
        # indexed passage: the match is then about something else ("gold prices").
        specific = [w for w in (f.topic or "").split() if len(w) >= 4]
        absent = [w for w in specific if not vector_store.term_in_corpus(w)]
        if absent and len(absent) * 2 >= len(specific):
            logger.info("Relevance gate: topic term(s) %s occur in no indexed passage and the best match is only "
                        "cos=%.3f (< %.2f) -> not covered", absent, top_dense, config.RAG_STRONG_MATCH)
            return []
    min_dense = config.RAG_MIN_DENSE_SCORE
    gated = []
    for r in by_id.values():
        lexical = r["bm25"] > 0
        if r["score"] >= max(min_dense, top_dense * 0.6) or (lexical and r["score"] >= min_dense * 0.6):
            if f.genre_hint and f.genre_hint in (r.get("genre"), r.get("genre_id")):
                r["rrf"] += 0.5 / RRF_K
            gated.append(r)
    gated.sort(key=lambda r: r["rrf"], reverse=True)
    logger.info("Relevance gate: %d/%d candidates kept (min cos=%.2f, top cos=%.3f)",
                len(gated), len(by_id), min_dense, top_dense)
    return gated


def retrieve(query: str, f: Optional[QueryFilters] = None, top_k: Optional[int] = None,
             date_from: Optional[str] = None, date_to: Optional[str] = None,
             relax_genre: bool = True) -> dict:
    """Return {'hits', 'filters', 'mode'} where mode is 'search' | 'browse' | 'empty-filter'.

    relax_genre=False keeps a genre filter even when it finds nothing (the user
    chose that genre explicitly in the UI, so results from other genres are wrong).
    """
    import vector_store

    f = f or parse_query(query)
    if date_from:
        f.date_from = date_from
    if date_to:
        f.date_to = date_to
    top_k = top_k or config.RAG_TOP_K
    logger.info("Query understanding: %s", f.describe())

    ids = vector_store.metadata_prefilter(f.date_from, f.date_to, f.genre, hour_from=f.hour_from,
                                          hour_to=f.hour_to, source_category=f.source_category)
    if not ids and f.source_category and relax_genre:
        logger.info("No documents in publisher category '%s'; searching all categories", f.source_category)
        f.source_category = None
        ids = vector_store.metadata_prefilter(f.date_from, f.date_to, f.genre, hour_from=f.hour_from,
                                              hour_to=f.hour_to)
    has_filter = any([f.date_from, f.date_to, f.genre, f.hour_from is not None, f.source_category])
    logger.info("Metadata prefilter (SQLite): %d candidate segments%s", len(ids), "" if has_filter else " (whole corpus)")
    if not ids:
        return {"hits": [], "filters": f, "mode": "empty-filter"}

    if f.latest and not (f.date_from or f.date_to):
        newest = vector_store.latest_date(ids)
        if newest:
            f.date_from = f.date_to = newest
            ids = vector_store.metadata_prefilter(newest, newest, f.genre, hour_from=f.hour_from, hour_to=f.hour_to,
                                                  source_category=f.source_category)
            has_filter = True
            logger.info("'latest' resolved to %s: %d candidate segments", newest, len(ids))

    if not f.topic and (has_filter or f.latest):
        hits = vector_store.browse(ids, limit=max(top_k, 8))
        for h in hits:
            h.setdefault("score", None)
        logger.info("No topic words -> chronological browse of the filtered period (%d segments)", len(hits))
        return {"hits": hits, "filters": f, "mode": "browse"}

    hits = _hybrid_search(query, ids, f)
    if not hits and f.genre and relax_genre:
        # The classifier may have filed the story under a neighbouring genre: retry with genre as a hint.
        logger.info("No hits inside genre '%s'; retrying without the genre filter", f.genre)
        f.genre_hint, f.genre = f.genre, None
        ids = vector_store.metadata_prefilter(f.date_from, f.date_to, None, hour_from=f.hour_from, hour_to=f.hour_to,
                                              source_category=f.source_category)
        hits = _hybrid_search(query, ids, f) if ids else []
    return {"hits": _limit_groups(_drop_near_duplicates(hits), top_k), "filters": f, "mode": "search"}


def _drop_near_duplicates(hits: list[dict], threshold: float = 0.8) -> list[dict]:
    """Remove a hit whose wording nearly repeats a better-ranked hit from another document.

    Bulletins repeat the same story across programmes (Midday, News at Nine, Hourly); sending
    every repeat to the generator wastes context without adding facts. Chunks of the same
    article are never dropped (they are different sections).
    """
    kept, seen = [], []
    for h in hits:
        words = set(re.findall(r"[a-z0-9]+", (h.get("text") or "").lower()))
        doc = h.get("doc_id") or h.get("article_id") or h.get("source_file")
        dup = False
        for w, d in seen:
            if d != doc and words and w and len(words & w) / len(words | w) >= threshold:
                dup = True
                break
        if dup:
            continue
        seen.append((words, doc))
        kept.append(h)
    if len(kept) < len(hits):
        logger.info("Removed %d near-duplicate chunk(s) repeated across documents", len(hits) - len(kept))
    return kept


def _limit_groups(hits: list[dict], top_k: int) -> list[dict]:
    """Keep the hits of the first top_k sources, where all chunks of one article count as one source.

    Several matching chunks of the same article are kept together (they tell the
    generator which sections matched) without crowding out other articles.
    """
    keep, groups = [], []
    for h in hits:
        g = h.get("article_id") or h["faiss_id"]
        if g not in groups:
            if len(groups) >= top_k:
                continue
            groups.append(g)
        keep.append(h)
    return keep


def _genre_name(h: dict) -> Optional[str]:
    import taxonomy
    return taxonomy.genre_name(h["genre_id"]) if h.get("genre_id") else h.get("genre")


def _assemble_article(h: dict, number: int) -> dict:
    """One source for a whole article: the matched chunk is the quote, the full article is the context."""
    import vector_store
    art = vector_store.get_article(h["article_id"])
    return {
        "id": f"S{number}",
        "faiss_id": h["faiss_id"],
        "kind": "article",
        "article_id": h["article_id"],
        "doc_id": h.get("doc_id") or h["article_id"],
        "source_category": h.get("source_category"),
        "published_at": h.get("published_at"),
        "headline": art["headline"] if art else h.get("source_name"),
        "section": h.get("section"),
        "matched_sections": [h.get("section")],
        "matched_chunks": [h["text"]],
        "text": h["text"],
        "context": art["full_text"] if art else h["text"],
        "context_ids": [h["faiss_id"]],
        "word_count": art["word_count"] if art else len(h["text"].split()),
        "article": art["data"] if art else None,
        "demo": bool(art and art["data"].get("demo")),
        "genre": h.get("genre"),
        "genre_id": h.get("genre_id"), "genre_name": _genre_name(h), "broadcast_type": h.get("broadcast_type"),
        "date": h.get("date"),
        "hour": h.get("hour"),
        "time": _hhmm(h.get("hour")),
        "source_name": h.get("source_name"),
        "source_file": h.get("source_file"),
        "source_url": h.get("source_url"),
        "ingest_mode": "article",
        "ts_start": None, "audio_start": None, "audio_end": None,
        "context_audio_start": None, "context_audio_end": None,
        "has_audio": False,
        "score": h.get("score"),
        "bm25": round(h.get("bm25", 0.0), 3) if h.get("bm25") else 0.0,
        "rrf": round(h.get("rrf", 0.0), 4) if h.get("rrf") else None,
    }


def _assemble_sources(hits: list[dict], neighbors: int) -> list[dict]:
    """Number hits as S1..Sn.

    Broadcast hits get a sentence-window passage from the same broadcast;
    article hits are merged per article and get the complete article as context.
    """
    import vector_store
    sources, seen = [], set()
    by_article: dict[str, dict] = {}
    for h in hits:
        if h.get("article_id"):
            src = by_article.get(h["article_id"])
            if src is None:
                src = by_article[h["article_id"]] = _assemble_article(h, len(sources) + 1)
                sources.append(src)
            elif h["faiss_id"] not in src["context_ids"]:
                src["matched_sections"].append(h.get("section"))
                src["matched_chunks"].append(h["text"])
                src["context_ids"].append(h["faiss_id"])
            continue
        if h["faiss_id"] in seen:
            continue
        window = vector_store.get_neighbors(h["faiss_id"], neighbors) if neighbors else [h]
        window = window or [h]
        seen.update(w["faiss_id"] for w in window)
        starts = [w["audio_start"] for w in window if w.get("audio_start") is not None]
        ends = [w["audio_end"] for w in window if w.get("audio_end") is not None]
        src = {
            "id": f"S{len(sources) + 1}",
            "faiss_id": h["faiss_id"],
            "kind": "broadcast",
            "doc_id": h.get("doc_id"),
            "headline": h.get("headline"),
            "source_category": h.get("source_category"),
            "published_at": h.get("published_at"),
            "text": h["text"],
            "context": " ".join(w["text"] for w in window),
            "context_ids": [w["faiss_id"] for w in window],
            "genre": h.get("genre"),
            "genre_id": h.get("genre_id"), "genre_name": _genre_name(h), "broadcast_type": h.get("broadcast_type"),
            "date": h.get("date"),
            "hour": h.get("hour"),
            "time": _hhmm(h.get("hour")),
            "source_name": h.get("source_name"),
            "source_file": h.get("source_file"),
            "source_url": h.get("source_url"),
            "ingest_mode": h.get("ingest_mode") or "archive",
            "ts_start": h.get("ts_start"),
            "audio_start": h.get("audio_start"),
            "audio_end": h.get("audio_end"),
            "context_audio_start": min(starts) if starts else None,
            "context_audio_end": max(ends) if ends else None,
            "has_audio": h.get("has_audio", False),
            "score": h.get("score"),
            "bm25": round(h.get("bm25", 0.0), 3) if h.get("bm25") else 0.0,
            "rrf": round(h.get("rrf", 0.0), 4) if h.get("rrf") else None,
        }
        sources.append(src)
    return sources


def _hhmm(hour) -> Optional[str]:
    if hour is None:
        return None
    s = f"{int(hour):04d}"
    return f"{s[:2]}:{s[2:]}"


def _clip_words(text: str, limit: int) -> tuple[str, int]:
    """Cut text to `limit` words with a visible marker. Returns (text, words_cut)."""
    words = text.split()
    if len(words) <= limit:
        return text, 0
    return " ".join(words[:limit]) + f" [... {len(words) - limit} more words of this source not shown]", \
        len(words) - limit


def _source_header(s: dict) -> str:
    """Metadata line for one source: type, title, programme, genre, date/time, URL and audio range."""
    when = f"{s['date']} {s['time'] or ''}".strip()
    genre = s.get("genre_name") or s.get("genre") or "unclassified"
    parts = []
    if s.get("kind") == "article":
        parts += [f"Article \"{s['headline']}\"", s["source_name"], f"published {when}", f"genre: {genre}"]
        matched = ", ".join(dict.fromkeys(x for x in s["matched_sections"] if x))
        if matched:
            parts.append(f"best-matching sections: {matched}")
        if s.get("demo"):
            parts.append("illustrative demo content")
    else:
        parts += [f"Broadcast transcript \"{s.get('headline') or s['source_name']}\"",
                  f"programme: {s.get('broadcast_type') or s['source_name']}", f"aired {when}", f"genre: {genre}"]
        if s.get("context_audio_start") is not None:
            parts.append(f"audio {fmt_ts(s['context_audio_start'])}-{fmt_ts(s['context_audio_end'])}")
    if s.get("source_url"):
        parts.append(f"url: {s['source_url']}")
    return f"[{s['id']}] (" + " | ".join(parts) + ")"


def _build_context(sources: list[dict], stats: Optional[dict] = None) -> str:
    """Numbered sources for the prompt, within RAG_CONTEXT_MAX_WORDS.

    Long sources are cut at RAG_CONTEXT_MAX_WORDS_PER_SOURCE with an explicit marker; once the total
    budget is spent, remaining sources keep their header and matched passage only. Nothing is dropped
    without a marker, and `stats` (if given) receives the words sent and cut for logging.
    """
    budget = config.RAG_CONTEXT_MAX_WORDS
    per_src = config.RAG_CONTEXT_MAX_WORDS_PER_SOURCE
    lines, used, cut, clipped = [], 0, 0, []
    for s in sources:
        body, n_cut = _clip_words(s["context"], min(per_src, max(budget - used, 0)))
        if not body.strip() or n_cut and budget - used < 80:
            # budget spent: keep the matched passage so the citation stays checkable
            body, n_cut = _clip_words(s.get("text") or s["context"], 80)
            body = "[context budget reached; matched passage only] " + body
        if n_cut:
            clipped.append(s["id"])
        cut += n_cut
        used += len(body.split())
        lines.append(f"{_source_header(s)}\n{body}")
    if stats is not None:
        stats.update(words_sent=used, words_cut=cut, clipped_sources=clipped)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_generation_mode: str = "UNKNOWN"

SYSTEM_PROMPT = (
    "You are VaniStream, a news research assistant. You answer questions using ONLY the numbered sources "
    "provided: transcribed All India Radio broadcasts and detailed news articles. Rules:\n"
    "1. Every sentence that states a fact must end with its citation(s) in square brackets, e.g. [S1] or "
    "[S1][S3]. Only cite source ids that appear in the context. Headings need no citation.\n"
    "2. Use only facts, names, numbers, dates and events that are in the sources. Never add statistics, "
    "dates, events or people from general knowledge. Quote a person only with words that appear verbatim "
    "in the sources.\n"
    "3. If you draw a conclusion the sources do not state directly, start that sentence with 'Inference:' "
    "and cite the sources it rests on. Keep source-backed facts and interpretation clearly apart.\n"
    "4. When several sources are used, make clear which source supports which claim, naming the article "
    "headline or broadcast (e.g. 'The grid report [S2] says ...').\n"
    "5. If the sources answer only part of the question, answer that part and say plainly what they do not "
    "cover. Do not fill gaps.\n"
    "6. If the sources do not answer the question at all, or only mention a related name or topic without "
    "the specific event, figure or outcome asked about, reply exactly: "
    "'The broadcasts I have do not contain enough information to answer this.' and nothing else.\n"
    "7. Do not pad. Never restate the same fact in other words to make the answer longer; length must come "
    "from distinct evidence. Match the length to the question as instructed in the answer guidance.\n"
    "8. Format as plain text with '### ' headings and '- ' bullets only (no tables, no bold). Mention "
    "broadcast or publication dates and times where they help. Every heading must be followed by content; "
    "never return only headings or only a Sources list.\n"
    "9. A source marked '[... N more words of this source not shown]' was shortened; do not guess what the "
    "missing part says."
)
INSUFFICIENT_MARKER = "do not contain enough information"
REFUSAL_TEXT = "the broadcasts i have do not contain enough information"

# Question types drive answer length and structure (see answer_guidance).
_GAP_RE = re.compile(r"\b(missing|gaps?|unanswered|left out|unknown|not (?:mentioned|reported|disclosed|said|"
                     r"covered|stated|known)|(?:do|does|did)(?:n't| not) (?:say|tell|mention|cover|explain))\b")
_COMPARE_RE = re.compile(r"\b(compare|comparison|contrast|differ|differs|difference|differences|versus|vs|"
                         r"similar|similarities)\b")
_RELATION_RE = re.compile(r"\b(relate|related|relationship|connection|connected|connect|link|linked)\b")
_EXPLAIN_RE = re.compile(r"\b(why|explain|explanation|background|context|implications?|impacts?|consequences?|"
                         r"effects?|affect|affects|detail|detailed|elaborate|analy[sz]e|analysis|reasons?|causes?|"
                         r"significance|matter|matters)\b|\bhow (?:did|does|do|will|would|could|might)\b")
_OVERVIEW_RE = re.compile(r"\b(summari[sz]e|summary|overview|round-?up|major developments|main developments|"
                          r"key developments|main stories|major stories|what happened)\b")
_SIMPLE_RE = re.compile(r"^\s*(?:by\s+)?(who|when|where|which|how (?:many|much|long)|what score|"
                        r"what (?:is|was|were) the (?:name|date|number|score|price|rate|figure|result|percentage))\b")

STRUCTURE = ("Use this structure with '### ' headings, skipping any section the sources cannot support "
             "or that would only repeat points already made:\n"
             "### Direct answer\n### Detailed explanation\n### Background and context\n"
             "### Key developments\n### Implications\n"
             "### Sources (one line per cited source: '- [S#] headline or broadcast, date, time')")


def classify_question(query: str) -> str:
    """'gap' | 'compare' | 'explain' | 'overview' | 'simple' | 'standard'."""
    q = query.lower()
    if _GAP_RE.search(q):
        return "gap"
    if _COMPARE_RE.search(q):
        return "compare"
    if _EXPLAIN_RE.search(q):
        return "explain"  # "why ... and how is it linked to ..." is an explanation first
    if _RELATION_RE.search(q):
        return "compare"
    if _OVERVIEW_RE.search(q):
        return "overview"
    if _SIMPLE_RE.search(q) and len(q.split()) <= 14:
        return "simple"
    return "standard"


def answer_guidance(kind: str) -> str:
    """Length and structure instructions for one question type."""
    long_form = ("Aim for about 300-600 words. Go longer only if the sources contain enough distinct, "
                 "relevant evidence; stay shorter if they do not. ")
    return {
        "simple": "This is a simple factual question. Answer in 1-3 sentences (about 30-80 words), "
                  "with citations. No headings and no Sources section.",
        "standard": "Answer in about 150-300 words in short paragraphs. Use '### ' headings only if they help, "
                    "and end with a '### Sources' list if you used headings.",
        "explain": "This question needs explanation. " + long_form + STRUCTURE,
        "overview": "Summarise every distinct story in the sources, grouped by story, with its key "
                    "developments and why it matters. " + long_form + STRUCTURE,
        "compare": "Compare the sources explicitly: what each one says, where they agree, differ or connect, "
                   "and how the events relate. Name the article or broadcast for every claim. " + long_form
                   + STRUCTURE.replace("### Implications", "### How the stories compare or connect\n### Implications"),
        "gap": "First summarise what the sources do say about the topic, then add '### What the sources do not "
               "say', listing "
               "specific points the question asks about that the sources leave unanswered (figures, costs, "
               "outcomes, decisions). Do not guess the missing information. " + long_form + STRUCTURE,
    }[kind]


def gemini_models() -> list[str]:
    """Primary model followed by the configured fallbacks, without duplicates."""
    return list(dict.fromkeys([config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS))


# model name -> time.monotonic() until which it is skipped (quota exhausted / unavailable)
_model_cooldown: dict[str, float] = {}
_last_model_used: Optional[str] = None


def _probe_gemini() -> str:
    """Check the key and that at least one configured model exists.

    Uses a metadata lookup, not a generation call, so the probe is fast and does
    not spend generation quota.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        return "FALLBACK (google-generativeai not installed)"
    if not config.GEMINI_API_KEY:
        return "FALLBACK (no GEMINI_API_KEY)"
    genai.configure(api_key=config.GEMINI_API_KEY)
    errors = []
    for name in gemini_models():
        try:
            genai.get_model(f"models/{name}", request_options={"timeout": 20})
            return "LIVE"
        except Exception as e:
            errors.append(f"{name}: {str(e)[:80]}")
    return f"FALLBACK (model error: {'; '.join(errors)[:200]})"


def get_generation_mode() -> str:
    """'LIVE' when Gemini is reachable, otherwise 'FALLBACK (reason)'. Probed once per process."""
    global _generation_mode
    if _generation_mode == "UNKNOWN":
        _generation_mode = _probe_gemini()
        logger.info("Generation mode: %s (models=%s)", _generation_mode, ", ".join(gemini_models()))
    return _generation_mode


def _is_quota_error(e: Exception) -> bool:
    text = f"{type(e).__name__} {e}".lower()
    return "429" in text or "resourceexhausted" in text or "quota" in text


def _generate_with_gemini(context: str, user_query: str, history: Optional[list[dict]] = None,
                          guidance: str = "") -> str:
    """Generate with the first configured model that answers.

    A model that returns a quota error (429) or is unavailable (404) is skipped
    for GEMINI_COOLDOWN_SECONDS and the next model in GEMINI_FALLBACK_MODELS is
    tried, so one exhausted free-tier quota does not disable generation.
    """
    global _last_model_used
    _last_model_used = None
    if get_generation_mode() != "LIVE":
        return ""
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    contents = []
    for turn in (history or [])[-4:]:
        contents.append({"role": turn["role"], "parts": [turn["content"][:1500]]})
    contents.append({"role": "user", "parts": [
        f"Sources:\n{context}\n\nAnswer guidance: {guidance or answer_guidance('standard')}\n\nQuestion: {user_query}"]})
    import ai_health
    # No SDK-level retry: a 429 used to be retried for minutes before the next model was tried.
    opts = {"timeout": config.GEMINI_TIMEOUT_SECONDS}
    try:
        from google.api_core import retry as gretry
        opts["retry"] = gretry.Retry(predicate=lambda e: False, timeout=config.GEMINI_TIMEOUT_SECONDS)
    except ImportError:
        pass
    now = time.monotonic()
    last_error, last_name = None, None
    for name in gemini_models():
        if _model_cooldown.get(name, 0) > now:
            continue
        t0 = time.perf_counter()
        last_name = name
        try:
            model = genai.GenerativeModel(name, system_instruction=SYSTEM_PROMPT)
            resp = model.generate_content(contents, generation_config={
                "temperature": 0.2, "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS},
                request_options=opts)
            text = (resp.text or "").strip()
            if text:
                _last_model_used = name
                ai_health.record_generation(name, True, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
                if name != gemini_models()[0]:
                    logger.info("Answer generated by fallback model %s (primary %s skipped or failed)",
                                name, gemini_models()[0])
                return text
            last_error = None
        except Exception as e:
            last_error = e
            if _is_quota_error(e) or "404" in str(e):
                _model_cooldown[name] = now + config.GEMINI_COOLDOWN_SECONDS
                logger.warning("Gemini model %s unavailable (%s); trying the next model",
                               name, "quota exhausted" if _is_quota_error(e) else "not found")
            else:
                logger.error("Gemini model %s failed: %s", name, ai_health._scrub(str(e))[:200])
    ai_health.record_generation(last_name, False, error=last_error)
    logger.error("No Gemini model produced an answer; using the extractive answer")
    return ""


def _answer_problem(answer: str, sources: list[dict]) -> Optional[str]:
    """Why a generated answer is unusable (None if fine): no citation, or headings without content."""
    if _is_refusal(answer):
        return None
    if not re.search(r"\[S\d+\]", answer, re.I):
        return "no citations"
    body = re.split(r"(?im)^###\s*Sources\b", answer)[0]
    body = re.sub(r"(?m)^###.*$", "", body)
    if not re.search(r"\w.*\[S\d+\]", body, re.I):
        return "no cited content outside headings and the Sources list"
    return None


def validate_citations(answer: str, sources: list[dict]) -> tuple[str, list[str], list[str]]:
    """Remove citations to unknown source ids. Returns (answer, cited_ids, removed_ids)."""
    valid = {s["id"] for s in sources}
    cited, removed = [], []

    def fix(m):
        sid = m.group(1).upper()
        if sid in valid:
            if sid not in cited:
                cited.append(sid)
            return f"[{sid}]"
        removed.append(sid)
        return ""

    # Models sometimes group citations ("[S1, S2]", "[S1; S3]"); split them into "[S1][S2]".
    answer = re.sub(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)+)\s*\]",
                    lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,;]", m.group(1))),
                    answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[(S\d+)\]", fix, answer, flags=re.IGNORECASE)
    return answer, cited, removed


_GAP_SENTENCE_RE = re.compile(r"\b(not (?:yet )?(?:disclosed|published|announced|reported|said|known)|"
                              r"did not disclose|has not|have not|cannot (?:yet )?(?:say|be judged)|"
                              r"no such data|not yet)\b", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text.replace("\n", " ")) if s.strip()]


def _best_sentences(texts: list[str], query: str, n: int) -> list[str]:
    """The n sentences sharing most content words with the query, in reading order."""
    import vector_store
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in vector_store.STOPWORDS and len(w) > 2}
    cands = [s.lstrip("- ") for t in texts for s in _sentences(t)]
    scored = sorted(range(len(cands)), key=lambda i: -len(words & set(re.findall(r"[a-z0-9]+", cands[i].lower()))))
    return [cands[i] for i in sorted(scored[:n])]


def _cite(text: str, sid: str) -> str:
    return f"{text.rstrip()} [{sid}]"


def _words_of(text: str) -> set[str]:
    import vector_store
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in vector_store.STOPWORDS and len(w) > 2}


def _is_new(text: str, said: set[str], threshold: float = 0.6) -> bool:
    """False when most of the item's content words are already in the answer (avoid repeating facts)."""
    w = _words_of(text)
    return not w or len(w & said) / len(w) < threshold


def _lead(article: dict) -> str:
    """The article's own summary, or (for publisher articles without one) its first paragraph."""
    return article.get("summary") or (article.get("body") or [""])[0]


def _extractive_article_answer(arts: list[dict], kind: str, query: str) -> list[str]:
    """Structured answer assembled only from article passages (used when no generator is available).

    Every paragraph and bullet is copied from an article section and cited, so
    the answer can be long without adding anything the sources do not say.
    """
    primary = arts[0]
    a0 = primary["article"]
    if kind == "simple":
        # Pick the best-matching sentences across the top sources that scored close to the best one.
        top = primary.get("score") or 0
        pool = [s for s in arts[:3] if s is primary or (s.get("score") or 0) >= 0.85 * top]
        cands = [(sent, s["id"]) for s in pool for sent in _best_sentences(s["matched_chunks"] + [_lead(s["article"])], query, 2)]
        best = _best_sentences([c for c, _ in cands], query, 2)
        sid = {c: i for c, i in cands}
        return [" ".join(_cite(b, sid.get(b, primary["id"])) for b in best)]

    if kind in ("compare", "overview"):
        covered = arts[:2] if kind == "compare" else arts
    else:
        covered = arts[:1]
    # Extra articles are mentioned only when they are nearly as relevant as the main one.
    top = primary.get("score") or 0
    others = [s for s in arts if s not in covered and (s.get("score") is None or s["score"] >= 0.6 * top)][:2]
    brief = kind == "overview" and len(covered) > 3
    title = lambda s: f"\"{s['headline']}\" ({s['date']})"  # noqa: E731
    out = ["### Direct answer"]
    for s in covered:
        lead = f"{title(s)}: " if len(covered) > 1 else ""
        out.append(_cite(lead + _lead(s["article"]), s["id"]))

    if not brief:
        out.append("### Detailed explanation")
        for s in covered:
            matched = [p for c, sec in zip(s["matched_chunks"], s["matched_sections"]) if sec == "Report"
                       for p in c.split("\n\n")]
            paras = [x for x in (matched or s["article"]["body"]) if x != _lead(s["article"])]
            paras = paras[: 1 if len(covered) > 1 else 2]
            for p in paras:
                out.append(_cite(p, s["id"]))
        if any(s["article"].get("background") for s in covered):
            out.append("### Background and context")
        for s in covered:
            for p in s["article"].get("background", [])[: 1 if len(covered) > 1 else 2]:
                out.append(_cite(p, s["id"]))

    said = _words_of("\n".join(out))
    dev_lines = []
    for s in covered:
        devs = [d for d in s["article"].get("key_developments", []) if _is_new(d, said)]
        if not devs:
            continue
        if len(covered) > 1:
            dev_lines.append(f"From {title(s)}:")
            devs = devs[:2 if brief else 3]
        dev_lines += [f"- {_cite(d, s['id'])}" for d in devs]
    if dev_lines:
        out += ["### Key developments"] + dev_lines

    if not brief and any(s["article"].get("implications") for s in covered):
        out.append("### Implications")
        for s in covered:
            for p in s["article"].get("implications", [])[: 1 if len(covered) > 1 else 2]:
                out.append(_cite(p, s["id"]))

    if kind == "gap":
        gaps = [(g, s["id"]) for s in covered + others for t in s["article"].get("implications", [])
                for g in _sentences(t) if _GAP_SENTENCE_RE.search(g)]
        out.append("### What the sources do not say")
        if gaps:
            out.append("The articles themselves state that these points are not available:")
            out += [f"- {_cite(g, sid)}" for g, sid in gaps]
        else:
            out.append("The articles do not flag missing information explicitly; anything not quoted above "
                       "is not covered by the sources.")

    if others:
        out.append("### Related coverage")
        out += [f"- {_cite(f'{title(s)}: ' + _lead(s['article']), s['id'])}" for s in others]
    return out


def _extractive_answer(sources: list[dict], mode: str, query: str = "") -> str:
    kind = classify_question(query)
    # Without a model to judge relevance, secondary sources are kept only when their match is
    # close to the best one (chronological browse results have no score and are all kept).
    # Comparison and overview questions name several stories on purpose, so the bar is lower.
    top = sources[0].get("score")
    ratio = 0.5 if kind in ("compare", "overview") else 0.75
    sources = [s for s in sources if top is None or s.get("score") is None or s["score"] >= ratio * top]
    arts = [s for s in sources if s.get("kind") == "article" and s.get("article")]
    casts = [s for s in sources if s not in arts]
    lines: list[str] = []
    article_led = bool(arts) and (mode == "browse" or sources[0] in arts)
    if article_led:
        lines += _extractive_article_answer(arts, "overview" if mode == "browse" and kind != "simple" else kind,
                                            query)
    if casts:
        if article_led:
            lines.append("### From the broadcasts")
        else:
            lines.append("Here is what the broadcasts in that period covered:" if mode == "browse"
                         else "Here is what the broadcasts say (direct quotes from the transcripts):")
        for s in casts:
            when = f"{s['date']} {s['time'] or ''}".strip()
            lines.append(f"- {s['text']} [{s['id']}] ({when})")
    if arts and not article_led:
        lines.append("Related articles:")
        lines += [f"- {_cite(chr(34) + s['headline'] + chr(34) + ' (' + s['date'] + '): ' + _lead(s['article']), s['id'])}"
                  for s in arts]
    if article_led and kind != "simple":
        lines.append("### Sources")
        for s in sources:
            label = f"\"{s['headline']}\", {s['source_name']}" if s in arts else s["source_name"]
            lines.append(f"- [{s['id']}] {label}, {s['date']} {s['time'] or ''}".rstrip())
    return "\n".join(lines)


def _is_refusal(answer: str) -> bool:
    """True only for the model's whole-answer refusal, not a partial answer that notes a gap."""
    a = answer.strip().lower()
    return a.startswith(REFUSAL_TEXT) or (INSUFFICIENT_MARKER in a and len(a) < 200)


def _insufficient(f: QueryFilters, reason: str) -> str:
    import vector_store
    lo, hi = vector_store.get_date_range()
    parts = []
    if f.date_from:
        parts.append(f"for {f.date_from}" + (f" to {f.date_to}" if f.date_to and f.date_to != f.date_from else ""))
    if f.hour_from is not None:
        parts.append(f"between {_hhmm(f.hour_from)} and {_hhmm(f.hour_to)}")
    if f.genre:
        parts.append(f"in {f.genre}")
    scope = (" " + " ".join(parts)) if parts else ""
    coverage = f" The corpus currently covers {lo} to {hi}." if lo else " The corpus is currently empty."
    return f"I don't have enough information in the broadcasts{scope} to answer that ({reason}).{coverage}"


def answer_query(user_query: str, date_from: Optional[str] = None, date_to: Optional[str] = None,
                 history: Optional[list[dict]] = None, genre: Optional[str] = None) -> dict:
    """Answer a question (see _answer_query) and record its performance in the query log."""
    t0 = time.perf_counter()
    result, error = None, None
    try:
        result = _answer_query(user_query, date_from, date_to, history, genre)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        raise
    finally:
        if config.QUERY_LOG_ENABLED:
            try:
                _log_query(user_query, genre, date_from, date_to, result, error, time.perf_counter() - t0)
            except Exception as exc:  # analytics must never break answering
                logger.warning("Query log write failed: %s", exc)


def _log_query(user_query: str, genre: Optional[str], date_from: Optional[str], date_to: Optional[str],
               result: Optional[dict], error: Optional[str], elapsed: float) -> None:
    """Timings, filters and retrieved document ids; the question text itself is not stored."""
    import json as _json

    import taxonomy
    import vector_store
    r = result or {}
    t = r.get("timings") or {}
    fa = r.get("filters_applied") or {}
    detected = fa.get("genre") or fa.get("genre_hint")
    detected = taxonomy.genre_id_of(detected) or taxonomy.MODEL_LABEL_GENRE.get(detected or "") or detected
    docs = list(dict.fromkeys(s.get("doc_id") or s.get("source_file") for s in r.get("sources") or []))[:12]
    ms = lambda v: round(v * 1000, 1) if v is not None else None  # noqa: E731
    now = datetime.datetime.now(datetime.timezone.utc)
    trace_id = next((t for t in (r.get("trace_id"), current_trace_id()) if t and t != "-"), "Q")
    vector_store.log_query({
        # Trace ids (REQ-0001, ...) restart with the server, so the stored id adds the time.
        "request_id": f"{trace_id}@{now.strftime('%Y%m%dT%H%M%S.%f')}",
        "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "genre_scope": taxonomy.genre_id_of(genre) or genre, "detected_genre": None if genre else detected,
        "date_from": fa.get("date_from") or date_from, "date_to": fa.get("date_to") or date_to,
        "question_type": r.get("question_type"), "mode": r.get("mode") or ("error" if error else None),
        "model": r.get("model"), "retrieval_mode": r.get("retrieval_mode"), "retrieved_docs": _json.dumps(docs),
        "n_chunks": r.get("n_chunks"), "n_sources": len(r.get("sources") or []),
        "retrieval_ms": ms(t.get("retrieval_s")), "generation_ms": ms(t.get("generation_s")),
        "total_ms": ms(t.get("total_s")) if t.get("total_s") is not None else round(elapsed * 1000, 1),
        "cited_count": len(r.get("cited") or []), "has_citations": int(bool(r.get("cited"))),
        "insufficient": int(bool(r.get("insufficient_context"))) if result else None,
        "error": error, "query_chars": len(user_query or ""),
    })


def _answer_query(user_query: str, date_from: Optional[str] = None, date_to: Optional[str] = None,
                  history: Optional[list[dict]] = None, genre: Optional[str] = None) -> dict:
    """Answer a question with grounded citations.

    `genre` (from the newsroom's genre scope) is a strict filter on both
    broadcasts and articles; without it, a genre named in the question is used.

    Returns: answer, sources (list of S# dicts), filters_applied, mode
    ('generative' | 'extractive' | 'insufficient'), insufficient_context, trace_id, timings.
    """
    t0 = time.perf_counter()
    logger.info("User query received: \"%s\"", user_query[:200])
    f0 = None
    if genre:
        f0 = parse_query(user_query)
        f0.genre, f0.genre_hint = genre, None
        f0.topic = _topic_terms(user_query, genre)
    r = retrieve(user_query, f=f0, date_from=date_from, date_to=date_to, relax_genre=not genre)
    f: QueryFilters = r["filters"]
    filters_applied = {k: v for k, v in asdict(f).items()}
    t_ret = time.perf_counter() - t0

    if not r["hits"]:
        reason = "no segments match those filters" if r["mode"] == "empty-filter" else "no sufficiently relevant segment"
        logger.info("Insufficient context: %s", reason)
        return {"answer": _insufficient(f, reason), "sources": [], "filters_applied": filters_applied,
                "mode": "insufficient", "insufficient_context": True, "trace_id": current_trace_id(),
                "retrieval_mode": r["mode"], "n_chunks": 0,
                "timings": {"retrieval_s": round(t_ret, 3), "total_s": round(time.perf_counter() - t0, 3)}}

    sources = _assemble_sources(r["hits"], config.RAG_CONTEXT_NEIGHBORS)
    logger.info("Retrieved chunks: %d (%s)", len(sources), r["mode"])
    for s in sources:
        rng = (f"{fmt_ts(s['context_audio_start'])}-{fmt_ts(s['context_audio_end'])}"
               if s["context_audio_start"] is not None else "n/a")
        score = f"cos={s['score']:.3f} bm25={s['bm25']:.2f} rrf={s['rrf']}" if s["score"] is not None else "chronological"
        logger.info("  %s | %s | %s | %s %s | %s | audio %s | doc=%s faiss_id=%d | %s | %s",
                    s["id"], s["source_name"], (s.get("headline") or "")[:70], s["date"], s["time"] or "",
                    s.get("genre"), rng, s.get("doc_id") or "-", s["faiss_id"], score, s.get("source_url") or "-")
    ctx_stats: dict = {}
    context = _build_context(sources, ctx_stats)
    kind = classify_question(user_query)
    n_articles = sum(s.get("kind") == "article" for s in sources)
    logger.info("Context assembled: %d sources (%d full articles), %d chars, %d words sent, %d words cut%s "
                "| question type=%s", len(sources), n_articles, len(context), ctx_stats["words_sent"],
                ctx_stats["words_cut"], f" ({', '.join(ctx_stats['clipped_sources'])})"
                if ctx_stats["clipped_sources"] else "", kind)

    t1 = time.perf_counter()
    answer = _generate_with_gemini(context, user_query, history, guidance=answer_guidance(kind))
    problem = _answer_problem(answer, sources) if answer else None
    if problem:
        logger.warning("Generated answer rejected (%s); retrying once with a stricter instruction", problem)
        answer = _generate_with_gemini(context, user_query, history, guidance=answer_guidance(kind) + (
            " Your previous answer was rejected because it had " + problem + ". Write the answer itself, "
            "ending every factual sentence with its [S#] citation."))
        problem = _answer_problem(answer, sources) if answer else None
        if problem:
            logger.warning("Retry also unusable (%s); using the extractive answer", problem)
            answer = ""
    mode = "generative" if answer else "extractive"
    if not answer:
        answer = _extractive_answer(sources, r["mode"], user_query)
    for s in sources:
        s.pop("article", None)  # structured copy only needed for the extractive answer
    if mode == "generative" and _is_refusal(answer):
        logger.info("Model reported insufficient context for this question")
        return {"answer": _insufficient(f, "the retrieved segments do not answer the question"),
                "sources": sources, "filters_applied": filters_applied, "mode": "insufficient",
                "insufficient_context": True, "trace_id": current_trace_id(),
                "retrieval_mode": r["mode"], "n_chunks": len(r["hits"]),
                "timings": {"retrieval_s": round(t_ret, 3), "generation_s": round(time.perf_counter() - t1, 3),
                            "total_s": round(time.perf_counter() - t0, 3)}}
    answer, cited, removed = validate_citations(answer, sources)
    if removed:
        logger.warning("Removed citations to unknown sources: %s", ", ".join(removed))
    if not cited:
        logger.warning("Answer contains no citations")
    t_gen = time.perf_counter() - t1
    logger.info("Answer generated: mode=%s, model=%s, type=%s, %d words, %d chars, %.2fs",
                mode, (_last_model_used or "gemini") if mode == "generative" else "extractive", kind,
                len(re.sub(r"\[S\d+\]", "", answer).split()), len(answer), t_gen)
    logger.info("Sources attached: %s (cited: %s) | total %.2fs",
                ", ".join(s["id"] for s in sources), ", ".join(cited) or "none", time.perf_counter() - t0)
    return {"answer": answer, "sources": sources, "filters_applied": filters_applied, "mode": mode,
            "insufficient_context": False, "cited": cited, "trace_id": current_trace_id(),
            "question_type": kind, "retrieval_mode": r["mode"], "n_chunks": len(r["hits"]),
            "model": (_last_model_used or "gemini") if mode == "generative" else None,
            "timings": {"retrieval_s": round(t_ret, 3), "generation_s": round(t_gen, 3),
                        "total_s": round(time.perf_counter() - t0, 3)}}


def timeline_query(user_query: str, date_from: Optional[str] = None, date_to: Optional[str] = None,
                   genre: Optional[str] = None) -> dict:
    """Relevant segments grouped by date and sorted by broadcast time."""
    f = parse_query(user_query)
    if genre:
        f.genre = genre
    r = retrieve(user_query, f=f, top_k=25, date_from=date_from, date_to=date_to)
    hits = sorted(r["hits"], key=lambda s: (s.get("date") or "", s.get("hour") or 0, s["faiss_id"]))
    groups: dict[str, list] = {}
    for h in hits:
        h = dict(h)
        h.pop("embedding", None)
        groups.setdefault(h.get("date") or "unknown", []).append(h)
    return {"groups": [{"date": d, "segments": s} for d, s in groups.items()],
            "filters_applied": asdict(r["filters"])}


if __name__ == "__main__":
    import sys
    from tracing import trace
    query = " ".join(sys.argv[1:]) or input("Enter your question: ")
    with trace("REQ"):
        result = answer_query(query)
    print("\n" + "=" * 60)
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['id']}] {s['source_name']} {s['date']} {s['time']} faiss_id={s['faiss_id']}: {s['text'][:90]}")
