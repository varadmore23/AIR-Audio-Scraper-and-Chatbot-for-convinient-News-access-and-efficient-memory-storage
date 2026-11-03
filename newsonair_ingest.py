"""
newsonair_ingest.py - Collect real news from NewsOnAIR (newsonair.gov.in) and index it.

Sources (both public; checked against robots.txt, which allows all paths):
  Articles   - the site's public WordPress REST API:
               GET /wp-json/wp/v2/categories?lang=en          (category ids)
               GET /wp-json/wp/v2/posts?lang=en&categories=N  (full article HTML)
  Broadcasts - the server-rendered bulletin archive /national-bulletins/page/N/.
               Each bulletin carries a stable id (share-bulletin?id=N), title,
               programme, language, broadcast date/time, the MP3 URL and its
               /listen-news/ page. English bulletins are downloaded and
               transcribed with the project's configured ASR (transcriber.py).
  The site's audio-archive AJAX endpoint (admin-ajax.php) answers 403 to
  non-browser clients; this module does not try to get around that.

Pipeline (reuses the existing modules, nothing is re-implemented):
  NewsOnAIR -> article/audio collection -> text extraction / transcription
  -> genre (publisher category, else the existing classifier) -> chunking
  (news_articles.chunk_article / nlp_pipeline sentences + MMR) -> MiniLM
  embeddings -> FAISS + SQLite + FTS5 (vector_store) -> RAG (rag_engine)

Incremental: every document has a stable id (noa-post-<wp id>, noa-bulletin-<id>)
and a content hash in the `ingestion_state` table. Unchanged documents are
skipped without re-embedding, changed ones are re-indexed (old chunks are
retired), failed ones are retried on the next run up to NEWSONAIR_MAX_ATTEMPTS.
Every run is recorded in `ingestion_jobs`.

Politeness: one request at a time, NEWSONAIR_DELAY_SECONDS between requests,
retries with backoff on timeouts / 429 / 5xx, an honest User-Agent, audio
files are downloaded once and cached under output/audio/.

CLI:
  python newsonair_ingest.py                          # default batch (config)
  python newsonair_ingest.py --articles 10 --broadcasts 1     # small test batch
  python newsonair_ingest.py --articles 80 --broadcasts 8
  python newsonair_ingest.py --no-audio               # articles only
  python newsonair_ingest.py --status                 # counts + recent jobs
  python newsonair_ingest.py --from 2026-10-01 --to 2026-11-01 --no-audio        # every article in a date range
  python newsonair_ingest.py --from 2026-10-01 --to 2026-11-01 --articles 0 \
         --programs "News At Nine,Midday News" --per-day 2                       # bulletins in a date range
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from typing import Optional

import requests

import config
from tracing import trace

logger = logging.getLogger("newsonair")

API = config.NEWSONAIR_BASE.rstrip("/") + "/wp-json/wp/v2"
BULLETINS_URL = config.NEWSONAIR_BASE.rstrip("/") + "/national-bulletins/"
SOURCE_NAME = "NewsOnAIR"

# NewsOnAIR category slug -> (publisher category label, AIR genre or None = classify the text).
# National / regional / miscellaneous cover many topics, so their genre comes from the existing
# classifier; the publisher category is kept separately (source_category) for "national news".
CATEGORY_MAP = {
    "national": ("National", None),
    "regional-news": ("Regional", None),
    "international": ("International", "International"),
    "business": ("Business", "Economy"),
    "sports": ("Sports", "Sports"),
    "health": ("Health", "Health"),
    "science-and-technologies": ("Science and Technology", "Science & Technology"),
    "entertainment": ("Entertainment", "Society & Culture"),
    "elections": ("Elections", "Politics"),
    "weather": ("Weather", "Environment"),
    "miscellaneous": ("Miscellaneous", None),
}
# Default mix for an article batch (scaled to the requested total).
DEFAULT_PLAN = {"national": 25, "international": 15, "business": 12, "sports": 12, "regional-news": 10,
                "health": 4, "science-and-technologies": 4, "entertainment": 3}

_run_lock = threading.Lock()
_current: dict = {}
_previews: dict[str, dict] = {}     # preview_id -> discovery result awaiting confirmation
_previews_lock = threading.Lock()
PREVIEW_TTL_SECONDS = 1800


class Cancelled(Exception):
    """Raised between documents when cancellation was requested for the running job."""


class PoliteClient:
    """Sequential HTTP client: fixed delay between requests, retries with backoff."""

    def __init__(self, delay: float = None, timeout: float = None, retries: int = 3):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.NEWSONAIR_USER_AGENT
        self.delay = config.NEWSONAIR_DELAY_SECONDS if delay is None else delay
        self.timeout = config.NEWSONAIR_TIMEOUT_SECONDS if timeout is None else timeout
        self.retries = retries
        self._last = 0.0
        self.requests_made = 0

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.monotonic()

    def get(self, url: str, params: Optional[dict] = None, stream: bool = False):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                self.requests_made += 1
                r = self.session.get(url, params=params, timeout=self.timeout, stream=stream)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise IOError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r
            except Exception as exc:  # network errors, timeouts, retryable status codes
                last_exc = exc
                if attempt < self.retries:
                    backoff = self.delay * (2 ** attempt)
                    logger.warning("GET %s failed (%s); retry %d/%d in %.0fs", url, exc, attempt,
                                   self.retries - 1, backoff)
                    time.sleep(backoff)
        raise RuntimeError(f"GET {url} failed after {self.retries} attempts: {last_exc}")

    def robots_allows(self, path: str) -> bool:
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.parse(self.get(config.NEWSONAIR_BASE.rstrip("/") + "/robots.txt").text.splitlines())
        except Exception as exc:
            logger.warning("Could not read robots.txt (%s); proceeding conservatively", exc)
            return True
        return rp.can_fetch(config.NEWSONAIR_USER_AGENT, config.NEWSONAIR_BASE.rstrip("/") + path)


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

_BLOCKS = ["p", "div", "li", "h2", "h3", "h4", "h5", "blockquote", "td"]
# Invisible characters the publisher's editor leaves in titles/text (word joiner, zero-width space, BOM, soft hyphen).
_INVISIBLE = re.compile("[\u2060\u200b\u200c\u200d\ufeff\u00ad]")


def html_to_paragraphs(fragment: str) -> list[str]:
    """Paragraph texts from WordPress post HTML (leaf block elements, whitespace normalised)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(fragment or "", "html.parser")
    for t in soup(["script", "style", "figure", "img", "iframe", "noscript"]):
        t.decompose()
    paras: list[str] = []
    for el in soup.find_all(_BLOCKS):
        if el.find(_BLOCKS):
            continue  # not a leaf block; its children are visited separately
        text = re.sub(r"\s+", " ", _INVISIBLE.sub("", el.get_text(" ", strip=True)).replace("\xa0", " ")).strip()
        text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
        if len(text) > 1 and (not paras or paras[-1] != text):
            paras.append(text)
    if not paras:  # plain text without block tags
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True).replace("\xa0", " ")).strip()
        paras = [text] if text else []
    return paras


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _INVISIBLE.sub("", html.unescape(re.sub(r"<[^>]+>", " ", text or "")))
                  .replace("\xa0", " ")).strip()


def fetch_categories(client: PoliteClient) -> dict[str, int]:
    r = client.get(API + "/categories", {"per_page": 100, "lang": "en", "_fields": "id,slug,count"})
    return {c["slug"]: c["id"] for c in r.json()}


def scale_plan(total: int, plan: Optional[dict] = None) -> dict[str, int]:
    plan = dict(plan or DEFAULT_PLAN)
    base = sum(plan.values())
    out = {k: max(1, round(v * total / base)) for k, v in plan.items()}
    while sum(out.values()) > total:  # trim rounding overshoot from the largest bucket
        k = max(out, key=out.get)
        out[k] -= 1
    return {k: v for k, v in out.items() if v > 0}


def discover_articles(client: PoliteClient, total: int, plan: Optional[dict] = None) -> list[dict]:
    """Newest English posts per category following the plan; de-duplicated by WordPress id."""
    cat_ids = fetch_categories(client)
    seen: dict[int, dict] = {}
    for slug, n in scale_plan(total, plan).items():
        if slug not in cat_ids:
            logger.warning("Category '%s' not found on NewsOnAIR; skipped", slug)
            continue
        r = client.get(API + "/posts", {
            "lang": "en", "categories": cat_ids[slug], "per_page": min(100, n), "orderby": "date", "order": "desc",
            "_fields": "id,date,date_gmt,modified,modified_gmt,link,title,content,excerpt,categories,author,lang"})
        posts = r.json()
        logger.info("Discovered %d posts in category '%s' (requested %d)", len(posts), slug, n)
        id_to_slug = {v: k for k, v in cat_ids.items()}
        for p in posts:
            p["_slug"] = slug
            p["_category_slugs"] = [id_to_slug.get(c) for c in p.get("categories", []) if c in id_to_slug]
            seen.setdefault(p["id"], p)
    return list(seen.values())


# When a post has several categories, the most specific mapped one wins.
_SLUG_PRIORITY = ["business", "sports", "international", "health", "science-and-technologies", "entertainment",
                  "elections", "weather", "national", "regional-news", "miscellaneous"]


def _tag_post(p: dict, id_to_slug: dict, slug_names: dict) -> dict:
    slugs = [id_to_slug.get(c) for c in p.get("categories", []) if c in id_to_slug]
    ranked = sorted(slugs, key=lambda x: _SLUG_PRIORITY.index(x) if x in _SLUG_PRIORITY else len(_SLUG_PRIORITY))
    p["_slug"] = ranked[0] if ranked else "uncategorised"
    p["_category_slugs"] = slugs
    p["_category_name"] = slug_names.get(p["_slug"], p["_slug"].replace("-", " ").title())
    return p


def discover_articles_range(client: PoliteClient, date_from: str, date_to: str,
                            limit: Optional[int] = None) -> list[dict]:
    """Every English post published between date_from and date_to (inclusive, site time), newest first."""
    r = client.get(API + "/categories", {"per_page": 100, "lang": "en", "_fields": "id,slug,name"})
    cats = r.json()
    id_to_slug = {c["id"]: c["slug"] for c in cats}
    slug_names = {c["slug"]: html.unescape(c["name"]) for c in cats}
    params = {"lang": "en", "after": f"{date_from}T00:00:00", "before": f"{date_to}T23:59:59", "per_page": 100,
              "orderby": "date", "order": "desc",
              "_fields": "id,date,date_gmt,modified,modified_gmt,link,title,content,excerpt,categories,author,lang"}
    posts: dict[int, dict] = {}
    page, total_pages = 1, None
    while True:
        r = client.get(API + "/posts", dict(params, page=page))
        total_pages = total_pages or int(r.headers.get("X-WP-TotalPages", 1))
        batch = r.json()
        for p in batch:
            posts.setdefault(p["id"], _tag_post(p, id_to_slug, slug_names))
        logger.info("Articles %s..%s: page %d/%d -> %d posts (total %d, site reports %s)", date_from, date_to, page,
                    total_pages, len(batch), len(posts), r.headers.get("X-WP-Total"))
        if not batch or page >= total_pages or (limit and len(posts) >= limit):
            break
        page += 1
    out = list(posts.values())
    if not limit or len(out) < limit:
        # Every post of every day in the range was listed: record what the source had per date.
        per_date = dict.fromkeys(_complete_days(date_from, date_to), 0)
        for p_ in out:
            d = (p_.get("date") or "")[:10]
            if d in per_date:
                per_date[d] += 1
        import vector_store
        vector_store.record_availability("article", per_date)
    return out[:limit] if limit else out


def count_articles_range(client: PoliteClient, date_from: str, date_to: str) -> int:
    """Number of English posts in the range, from the API's X-WP-Total header (one small request)."""
    r = client.get(API + "/posts", {"lang": "en", "after": f"{date_from}T00:00:00", "before": f"{date_to}T23:59:59",
                                    "per_page": 1, "_fields": "id"})
    return int(r.headers.get("X-WP-Total", 0))


def _complete_days(date_from: str, date_to: str) -> list[str]:
    """Dates in the range that are over (before today in site time): only those can be fully checked."""
    today = config.now_local().date()
    d0, d1 = dt.date.fromisoformat(date_from), min(dt.date.fromisoformat(date_to), today - dt.timedelta(1))
    return [(d0 + dt.timedelta(n)).isoformat() for n in range((d1 - d0).days + 1)] if d1 >= d0 else []


def parse_article(post: dict, retrieved_at: str) -> dict:
    """WordPress post -> article record used by news_articles / vector_store (no invented fields)."""
    slugs = [post["_slug"]] + [s for s in post.get("_category_slugs", []) if s != post["_slug"]]
    label, genre = None, None
    for s in slugs:
        if s in CATEGORY_MAP:
            label, genre = CATEGORY_MAP[s]
            break
    if label is None and post.get("_category_name"):
        label = post["_category_name"]  # unmapped publisher category (e.g. DD News, a state): keep its name
    body = html_to_paragraphs(post.get("content", {}).get("rendered", ""))
    excerpt = _clean(post.get("excerpt", {}).get("rendered", ""))
    headline = _clean(post.get("title", {}).get("rendered", ""))
    words = len(" ".join(body).split())
    published = post.get("date") or ""  # site-local time (Asia/Kolkata, per /wp-json gmt_offset 5.5)
    return {
        "id": f"noa-post-{post['id']}",
        "wp_id": post["id"],
        "headline": headline,
        "genre": genre,                       # None -> classified from the text before indexing
        "genre_source": "publisher category" if genre else "classifier",
        "source_category": label or "Other",
        "published": published[:10],
        "time": published[11:16] or "00:00",
        "hour": int((published[11:16] or "00:00").replace(":", "")),
        "published_at": published,
        "modified": post.get("modified_gmt") or post.get("modified"),
        "source_name": SOURCE_NAME,
        "source_url": post.get("link"),
        "language": post.get("lang") or "en",
        "author_id": post.get("author"),      # WordPress user id; author names are not public in the API
        "retrieved_at": retrieved_at,
        "word_count": words,
        "content_limited": words < config.NEWSONAIR_MIN_ARTICLE_WORDS,
        "excerpt": excerpt,
        "body": body,
        "demo": False,
    }


def content_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(p or "" for p in parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Broadcasts
# ---------------------------------------------------------------------------

_SHARE_ID = re.compile(r"share-bulletin%2F%3Fid%3D(\d+)")
_SHARE_TEXT = re.compile(r"navigator\.clipboard\.writeText\('(.*?)'\)", re.S)
_FIELDS = re.compile(r"^(?P<title>.*?) \| Category: (?P<category>.*?) \| Language: (?P<language>.*?) \| "
                     r"Date: (?P<date>.*?) \| Audio: (?P<audio>\S+)\s+(?P<page>\S+)\s*$", re.S)


def parse_bulletin_page(page_html: str) -> list[dict]:
    """Bulletins listed on one /national-bulletins/ page, in page order."""
    items, current_id = [], None
    events = sorted([(m.start(), "id", m.group(1)) for m in _SHARE_ID.finditer(page_html)] +
                    [(m.start(), "text", m.group(1)) for m in _SHARE_TEXT.finditer(page_html)])
    seen = set()
    for _, kind, value in events:
        if kind == "id":
            current_id = value
            continue
        m = _FIELDS.match(html.unescape(value).strip())
        if not m or current_id is None or current_id in seen:
            continue
        seen.add(current_id)
        try:
            when = dt.datetime.strptime(m["date"].strip(), "%d %B %Y %H:%M")
        except ValueError:
            when = None
        items.append({
            "bulletin_id": current_id, "title": m["title"].strip(), "category": m["category"].strip(),
            "language": m["language"].strip(), "broadcast_at": when.isoformat() if when else None,
            "date_text": m["date"].strip(), "audio_url": m["audio"].strip(), "page_url": m["page"].strip(),
        })
    return items


def discover_bulletins(client: PoliteClient, limit: int, language: str = "English",
                       max_pages: Optional[int] = None) -> list[dict]:
    max_pages = max_pages or config.NEWSONAIR_BULLETIN_PAGES
    found: list[dict] = []
    for page in range(1, max_pages + 1):
        url = BULLETINS_URL + (f"page/{page}/" if page > 1 else "")
        items = parse_bulletin_page(client.get(url).text)
        keep = [b for b in items if b["language"].lower() == language.lower() and b["broadcast_at"]]
        logger.info("Bulletin archive page %d: %d bulletins, %d in %s", page, len(items), len(keep), language)
        found += [b for b in keep if b["bulletin_id"] not in {f["bulletin_id"] for f in found}]
        if len(found) >= limit or not items:
            break
    return found[:limit]


def _archive_page(client: PoliteClient, page: int, cache: dict) -> list[dict]:
    if page not in cache:
        cache[page] = parse_bulletin_page(client.get(BULLETINS_URL + (f"page/{page}/" if page > 1 else "")).text)
    return cache[page]


def _page_oldest(items: list[dict]) -> Optional[str]:
    dates = [b["broadcast_at"][:10] for b in items if b["broadcast_at"]]
    return min(dates) if dates else None


def _find_start_page(client: PoliteClient, date_to: str, cache: dict, max_pages: int = 2000) -> int:
    """First archive page holding a bulletin dated on or before date_to.

    The archive is sorted newest first, so the page is found by doubling then bisecting
    (about 2*log2(pages) requests) instead of reading every newer page.
    """
    def reached(page: int) -> bool:
        items = _archive_page(client, page, cache)
        oldest = _page_oldest(items)
        return not items or (oldest is not None and oldest <= date_to)

    if reached(1):
        return 1
    lo, hi = 1, 2
    while hi < max_pages and not reached(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, max_pages)
    while hi - lo > 1:  # invariant: page lo is newer than date_to, page hi reaches it
        mid = (lo + hi) // 2
        if reached(mid):
            hi = mid
        else:
            lo = mid
    return hi


def scan_archive(client: PoliteClient, date_from: str, date_to: str, language: str = "English",
                 progress=None, max_pages: int = 2000) -> list[dict]:
    """All bulletins in `language` broadcast between date_from and date_to (every programme).

    Reuses the bulletin catalog when every date in the range was checked within
    CATALOG_MAX_AGE_HOURS; otherwise reads the archive pages covering the range (starting one
    page early, because new uploads shift items to later pages while the scan runs). Records
    the listing in the catalog and, for completed days, what the source had per date.
    """
    import taxonomy
    import vector_store
    cached, checked_at = vector_store.catalog_range(date_from, date_to, language)
    if checked_at:
        age_h = (dt.datetime.now(dt.timezone.utc) - dt.datetime.strptime(checked_at, "%Y-%m-%dT%H:%M:%SZ")
                 .replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
        if age_h <= config.CATALOG_MAX_AGE_HOURS:
            logger.info("Bulletins %s..%s: using the catalog checked %.1f h ago (%d bulletins, no archive requests)",
                        date_from, date_to, age_h, len(cached))
            return cached
    cache: dict[int, list[dict]] = {}
    start = max(1, _find_start_page(client, date_to, cache, max_pages) - 1)
    logger.info("Bulletins %s..%s: range starts near archive page %d (%d pages read to find it)", date_from, date_to,
                start + 1, len(cache))
    found: dict[str, dict] = {}
    everything: list[dict] = []
    page = start
    while page <= max_pages:
        items = _archive_page(client, page, cache)
        if not items:
            break
        everything += items
        for b in items:
            d = (b["broadcast_at"] or "")[:10]
            if d and date_from <= d <= date_to and b["language"].lower() == language.lower():
                found.setdefault(b["bulletin_id"], b)
        oldest = _page_oldest(items)
        if progress:
            progress(page, oldest, len(found))
        if page % 10 == 0:
            logger.info("Bulletin archive page %d reached %s; %d bulletins in range so far", page, oldest, len(found))
        if oldest and oldest < date_from:
            break
        page += 1
    vector_store.upsert_catalog([b for b in everything if b["broadcast_at"]])
    days = _complete_days(date_from, date_to)
    per_date = dict.fromkeys(days, 0)
    by_type: dict[str, dict] = {d: {} for d in days}
    for b in found.values():
        d = b["broadcast_at"][:10]
        if d in per_date:
            per_date[d] += 1
            t = taxonomy.broadcast_type_of(b["title"])
            by_type[d][t] = by_type[d].get(t, 0) + 1
    vector_store.record_availability("broadcast", per_date, by_type)
    logger.info("Bulletins %s..%s: %d %s bulletins found on %d archive pages", date_from, date_to, len(found), language,
                len(cache))
    return sorted(found.values(), key=lambda b: b["broadcast_at"], reverse=True)


def select_bulletins(items: list[dict], types: Optional[list[str]] = None, programs: Optional[list[str]] = None,
                     per_day: Optional[int] = None, include_indexed: bool = False) -> dict:
    """Split discovered bulletins into to_process / already_indexed / not_selected.

    `types` are taxonomy broadcast-type ids, `programs` exact titles (CLI); `per_day` caps bulletins
    *in the index* per date, so bulletins indexed by earlier runs count towards it.
    """
    import taxonomy
    import vector_store
    wanted_titles = {p.strip().lower() for p in programs} if programs else None
    per_date: dict[str, int] = vector_store.indexed_broadcasts_per_date() if per_day else {}
    out = {"to_process": [], "already_indexed": [], "not_selected": [], "previously_failed": 0}
    for b in items:
        t = taxonomy.broadcast_type_of(b["title"])
        if (types and t not in types) or (wanted_titles and b["title"].lower() not in wanted_titles):
            out["not_selected"].append(b)
            continue
        st = vector_store.get_state(f"noa-bulletin-{b['bulletin_id']}")
        if st and st["status"] == "indexed" and not include_indexed:
            out["already_indexed"].append(b)
            continue
        d = b["broadcast_at"][:10]
        if per_day and per_date.get(d, 0) >= per_day:
            out["not_selected"].append(b)
            continue
        per_date[d] = per_date.get(d, 0) + 1
        if st and st["status"] == "failed":
            out["previously_failed"] += 1
        out["to_process"].append(b)
    return out


def discover_bulletins_range(client: PoliteClient, date_from: str, date_to: str, language: str = "English",
                             programs: Optional[list[str]] = None, per_day: Optional[int] = None,
                             types: Optional[list[str]] = None, max_pages: int = 2000) -> list[dict]:
    """Bulletins in the range still to be indexed, filtered by programme title or broadcast type."""
    items = scan_archive(client, date_from, date_to, language, max_pages=max_pages)
    return select_bulletins(items, types=types, programs=programs, per_day=per_day)["to_process"]


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "bulletin"


# ---------------------------------------------------------------------------
# Indexing (reuses news_articles, nlp_pipeline, transcriber, vector_store)
# ---------------------------------------------------------------------------

def _classify_missing_genres(records: list[dict]) -> None:
    todo = [r for r in records if not r.get("genre")]
    if not todo:
        return
    from nlp_pipeline import classify_genres_batch
    labels = classify_genres_batch([f"{r['headline']}. {' '.join(r['body'][:2])}"[:1200] for r in todo], fast=True)
    for r, (g, conf) in zip(todo, labels):
        r["genre"], r["genre_confidence"] = g, round(float(conf), 3)
    logger.info("Genre classifier labelled %d articles without a mapped category", len(todo))


def index_article(rec: dict, counts: dict, job_id: str, force: bool = False) -> None:
    import news_articles
    import taxonomy
    import vector_store
    from nlp_pipeline import embed_sentences

    doc_id = rec["id"]
    h = content_hash(rec["headline"], *rec["body"])
    state = vector_store.get_state(doc_id)
    if state and state["status"] == "indexed" and state["content_hash"] == h:
        counts["duplicates_skipped"] += 1
        return
    if state and state["status"] == "failed" and (state["attempts"] or 0) >= config.NEWSONAIR_MAX_ATTEMPTS \
            and state["content_hash"] == h and not force:
        counts["gave_up"] += 1
        return
    updating = bool(state and state["status"] == "indexed")
    try:
        if not rec["body"]:
            raise ValueError("no article text in the API response")
        chunks = news_articles.chunk_article(rec)
        news_articles.verify_coverage(rec, chunks)
        embs = embed_sentences([f"{rec['headline']}. {c['section']}. {c['text']}" for c in chunks])
        counts["chunks"] += len(chunks)
        counts["embeddings"] += len(chunks)
        counts["article_chunks"] += len(chunks)
        counts["article_embeddings"] += len(chunks)
        if updating:
            retired = vector_store.deactivate_doc(doc_id)
            logger.info("Updated article %s: retired %d old chunks", doc_id, retired)
        gid, tags, _ = taxonomy.classify(rec["headline"], news_articles.full_text(rec), rec.get("source_category"),
                                         rec.get("genre"), rec.get("genre_confidence"), "article")
        rec["genre_id"], rec["genre_tags"] = gid, taxonomy.encode_tags(tags)
        segs = [{"text": c["text"], "search_text": f"{rec['headline']}\n{c['text']}", "genre": rec["genre"],
                 "genre_confidence": rec.get("genre_confidence", 1.0), "embedding": embs[i],
                 "article_id": doc_id, "section": c["section"], "chunk_index": i,
                 "genre_id": gid, "genre_tags": rec["genre_tags"]} for i, c in enumerate(chunks)]
        vector_store.upsert_article(rec, news_articles.full_text(rec))
        vector_store.add_segments(segs, rec["published"], rec["hour"], "articles/" + doc_id,
                                  source_name=SOURCE_NAME, source_url=rec["source_url"], ingest_mode="article",
                                  language=rec["language"],
                                  doc={"doc_id": doc_id, "content_type": "article", "headline": rec["headline"],
                                       "source_category": rec["source_category"], "published_at": rec["published_at"]})
        vector_store.set_state(doc_id, kind="article", source_url=rec["source_url"], title=rec["headline"][:300],
                               source_modified=rec.get("modified"), content_hash=h, status="indexed",
                               attempts=0, last_error=None, chunks=len(chunks), indexed_at=vector_store._now_iso(),
                               job_id=job_id, meta_json=json.dumps({"post_id": rec["id"].replace("noa-post-", "")}))
        counts["updated" if updating else "indexed_new"] += 1
        counts["articles_indexed"] += 1
        logger.info("Indexed article %s | %s | %s | %s %s | %d words -> %d chunks%s", doc_id, rec["genre"],
                    rec["source_category"], rec["published"], rec["time"], rec["word_count"], len(chunks),
                    " (short: publisher text is brief)" if rec["content_limited"] else "")
    except Exception as exc:
        counts["failed"] += 1
        vector_store.set_state(doc_id, kind="article", source_url=rec.get("source_url"), title=rec["headline"][:300],
                               content_hash=h, status="failed", attempts=((state or {}).get("attempts") or 0) + 1,
                               last_error=str(exc)[:500], job_id=job_id,
                               meta_json=json.dumps({"post_id": rec["id"].replace("noa-post-", "")}))
        logger.error("Article %s failed: %s", doc_id, exc)


def index_bulletin(b: dict, client: PoliteClient, counts: dict, job_id: str, transcribe: bool = True,
                   force: bool = False, stage=None) -> None:
    """Download, transcribe, chunk, embed and index one bulletin. `stage(name)` reports progress."""
    import taxonomy
    import vector_store
    stage = stage or (lambda name: None)
    from ingest_pipeline import _load_sentence_timestamps
    from nlp_pipeline import process_transcript, segment_sentences

    doc_id = f"noa-bulletin-{b['bulletin_id']}"
    h = content_hash(b["audio_url"], b["broadcast_at"])
    state = vector_store.get_state(doc_id)
    if state and state["status"] == "indexed" and state["content_hash"] == h:
        counts["duplicates_skipped"] += 1
        return
    if state and state["status"] == "failed" and (state["attempts"] or 0) >= config.NEWSONAIR_MAX_ATTEMPTS and not force:
        counts["gave_up"] += 1
        return
    meta = json.dumps({k: b.get(k) for k in ("bulletin_id", "title", "category", "language", "broadcast_at",
                                              "audio_url", "page_url")})
    when = dt.datetime.fromisoformat(b["broadcast_at"])
    date, hhmm = when.date().isoformat(), when.strftime("%H%M")
    name = f"NOA-{_safe(b['title'])}-{hhmm}-{b['bulletin_id']}"
    audio_path = os.path.join(config.OUTPUT_DIR, "audio", date, name + ".mp3")
    text_rel = f"text/{date}/{name}.txt"
    text_path = os.path.join(config.OUTPUT_DIR, "text", date, name + ".txt")
    headline = f"{b['title']} ({b['language']}) {when.strftime('%d %b %Y %H:%M')}"
    try:
        if not (os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0):
            stage("downloading")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            tmp = audio_path + ".part"
            # The server sometimes drops the connection mid-file; retry the whole file a bounded number of times.
            for attempt in range(1, config.NEWSONAIR_MAX_ATTEMPTS + 1):
                try:
                    r = client.get(b["audio_url"], stream=True)
                    with open(tmp, "wb") as fh:
                        for part in r.iter_content(1 << 16):
                            fh.write(part)
                    break
                except requests.exceptions.RequestException as exc:
                    if attempt == config.NEWSONAIR_MAX_ATTEMPTS:
                        raise
                    logger.warning("Download of %s interrupted (%s); retry %d/%d", b["audio_url"], exc, attempt,
                                   config.NEWSONAIR_MAX_ATTEMPTS - 1)
                    time.sleep(client.delay * (2 ** attempt))
            os.replace(tmp, audio_path)
            counts["broadcasts_downloaded"] += 1
            logger.info("Downloaded %s (%.1f MB) -> %s", b["audio_url"], os.path.getsize(audio_path) / 1e6, audio_path)
        else:
            counts["audio_cached"] += 1
            logger.info("Audio already cached: %s", audio_path)
        if not transcribe:
            raise RuntimeError("transcription disabled (--no-transcribe); bulletin not indexed")
        counts["audio_processed"] += 1
        if os.path.isfile(text_path) and os.path.isfile(os.path.splitext(text_path)[0] + ".segments.json"):
            with open(text_path, encoding="utf-8") as fh:
                text = fh.read()
            logger.info("Transcript already on disk, not re-transcribing: %s", text_path)
        else:
            stage("transcribing")
            from transcriber import get_transcriber
            result = get_transcriber().transcribe(audio_path)
            text = result.text
            if not text.strip():
                raise RuntimeError("ASR produced no text (silent or unreadable audio)")
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            with open(text_path, "w", encoding="utf-8") as fh:
                fh.write(text)
            with open(os.path.splitext(text_path)[0] + ".segments.json", "w", encoding="utf-8") as fh:
                json.dump(result.to_json(), fh, ensure_ascii=False)
        counts["transcriptions_ok"] += 1
        stage("chunking")
        ts = _load_sentence_timestamps(text_path, text)
        pool_emb, pool_txt = vector_store.get_pool("audio_transcript")
        segments, rejected = process_transcript(text, existing_embeddings=pool_emb, existing_texts=pool_txt,
                                                fast=True, detailed=True, sentence_timestamps=ts)
        n_sent = len(segment_sentences(text))
        counts["chunks"] += len(segments)
        counts["embeddings"] += n_sent
        counts["transcript_chunks"] += len(segments)
        counts["transcript_embeddings"] += n_sent
        counts["mmr_repeats_removed"] += len(rejected)
        stage("indexing")
        if state and state["status"] == "indexed":
            vector_store.deactivate_doc(doc_id)
        for i, s in enumerate(segments):
            s["chunk_index"] = i
            gid, tags, _ = taxonomy.classify("", s["text"], None, s.get("genre"), s.get("genre_confidence"),
                                             "audio_transcript")
            s["genre_id"], s["genre_tags"] = gid, taxonomy.encode_tags(tags)
        if segments:
            vector_store.add_segments(segments, date, int(hhmm), text_rel, source_name=f"{SOURCE_NAME} {b['title']}",
                                      source_url=b["page_url"], ingest_mode="archive", language="en",
                                      doc={"doc_id": doc_id, "content_type": "audio_transcript", "headline": headline,
                                           "source_category": b["category"], "published_at": b["broadcast_at"],
                                           "broadcast_type": taxonomy.broadcast_type_of(b["title"])})
        vector_store.set_state(doc_id, kind="broadcast", source_url=b["page_url"], title=headline,
                               source_modified=b["broadcast_at"], content_hash=h, status="indexed", attempts=0,
                               last_error=None, chunks=len(segments), indexed_at=vector_store._now_iso(), job_id=job_id,
                               meta_json=meta)
        if segments:
            vector_store.sync_broadcasts(only_source=text_rel)  # again: the state (meta, status) is now recorded
        counts["broadcasts_indexed"] += 1
        counts["indexed_new"] += 1
        timed = sum(1 for s in segments if s.get("audio_start") is not None)
        logger.info("Indexed broadcast %s | %s | %s sentences -> %d new segments (%d repeats removed by MMR), "
                    "%d with audio timestamps | audio %s | page %s", doc_id, headline, n_sent, len(segments),
                    len(rejected), timed, b["audio_url"], b["page_url"])
    except Exception as exc:
        counts["failed"] += 1
        vector_store.set_state(doc_id, kind="broadcast", source_url=b["page_url"], title=headline, content_hash=h,
                               status="failed", attempts=((state or {}).get("attempts") or 0) + 1,
                               last_error=str(exc)[:500], job_id=job_id, meta_json=meta)
        logger.error("Broadcast %s failed: %s", doc_id, exc)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

COUNT_KEYS = ("articles_discovered", "broadcasts_discovered", "articles_downloaded", "broadcasts_downloaded",
              "audio_cached", "audio_processed", "transcriptions_ok", "chunks", "embeddings", "article_chunks",
              "article_embeddings", "transcript_chunks", "transcript_embeddings", "indexed_new", "updated",
              "articles_indexed", "broadcasts_indexed", "already_indexed", "duplicates_skipped", "failed", "gave_up",
              "mmr_repeats_removed")


def validate_range(date_from: Optional[str], date_to: Optional[str], types: Optional[list[str]] = None,
                   include_articles: bool = False, max_days: Optional[int] = None) -> tuple[list[str], list[str]]:
    """Check a dashboard ingestion request. Returns (errors, normalised type ids); dates are never adjusted."""
    import taxonomy
    errors: list[str] = []
    try:
        type_ids = taxonomy.broadcast_type_ids(types)
    except ValueError as exc:
        errors.append(str(exc))
        type_ids = []
    d0 = d1 = None
    for label, value in (("Start date", date_from), ("End date", date_to)):
        if not value:
            errors.append(f"{label} is required")
            continue
        try:
            parsed = dt.date.fromisoformat(value)
        except ValueError:
            errors.append(f"{label} '{value}' is not a valid date (use YYYY-MM-DD)")
            continue
        if label == "Start date":
            d0 = parsed
        else:
            d1 = parsed
    today = config.now_local().date()
    if d0 and d1:
        if d0 > d1:
            errors.append(f"Start date {d0} is after end date {d1}")
        days = (d1 - d0).days + 1
        limit = max_days or config.INGEST_MAX_RANGE_DAYS
        if days > limit:
            errors.append(f"The range covers {days} days; one job may cover at most {limit} days")
    if d1 and d1 > today:
        errors.append(f"End date {d1} is in the future (today is {today} in {config.TIMEZONE})")
    if d0 and d0.isoformat() < config.INGEST_EARLIEST_DATE:
        errors.append(f"Start date {d0} is before {config.INGEST_EARLIEST_DATE}, the earliest date supported")
    if "other" in type_ids:
        errors.append("'Other / Unclassified' cannot be selected for ingestion; choose named programmes")
    if not type_ids and not include_articles:
        errors.append("Select at least one broadcast type, or include articles")
    return errors, type_ids


def _overlapping_job(date_from: Optional[str], date_to: Optional[str], bulletin_types: Optional[set],
                     articles: bool) -> Optional[dict]:
    """A running job (any process) that overlaps this request in dates and content.

    bulletin_types: None = this request processes no bulletins; empty set = every type.
    """
    import vector_store
    for j in vector_store.running_jobs():
        p = j.get("params") or {}
        jf, jt = p.get("date_from"), p.get("date_to") or p.get("date_from")
        if jf and date_from and (jt < date_from or jf > date_to):
            continue  # both ranged and disjoint
        retry = p.get("kind") == "retry"
        j_bul = None if (p.get("broadcasts") == 0 and not p.get("preselected") and not retry)             else set() if (retry or p.get("programs") or not p.get("types")) else set(p["types"])
        j_art = retry or (p.get("articles") != 0 and p.get("include_articles") is not False)
        bul_clash = bulletin_types is not None and j_bul is not None and             (not bulletin_types or not j_bul or bool(bulletin_types & j_bul))
        if bul_clash or (articles and j_art):
            return j
    return None


def preview(date_from: str, date_to: str, types: Optional[list[str]] = None, include_articles: bool = False,
            include_indexed: bool = False) -> str:
    """Start discovery for a range in the background; returns a preview id (see get_preview)."""
    errors, type_ids = validate_range(date_from, date_to, types, include_articles)
    if errors:
        raise ValueError("; ".join(errors))
    pid = "PRV-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    rec = {"preview_id": pid, "status": "discovering", "date_from": date_from, "date_to": date_to,
           "types": type_ids, "include_articles": include_articles, "include_indexed": include_indexed,
           "created": time.time(), "progress": "starting", "error": None}
    with _previews_lock:
        for k in [k for k, v in _previews.items() if time.time() - v["created"] > PREVIEW_TTL_SECONDS]:
            _previews.pop(k, None)
        _previews[pid] = rec

    def work():
        import taxonomy
        import vector_store
        try:
            with trace("NOA", trace_id=pid):
                client = PoliteClient()
                if type_ids:
                    def prog(page, oldest, n):
                        rec["progress"] = f"archive page {page} (reached {oldest}); {n} bulletins in range"
                    rec["progress"] = "locating the date range in the bulletin archive"
                    items = scan_archive(client, date_from, date_to, progress=prog)
                    sel = select_bulletins(items, types=type_ids, include_indexed=include_indexed)
                    by_type = {}
                    for key in ("to_process", "already_indexed"):
                        for b in sel[key]:
                            t = taxonomy.broadcast_type_of(b["title"])
                            row = by_type.setdefault(t, {"id": t, "name": taxonomy.broadcast_type_name(t),
                                                         "discovered": 0, "already_indexed": 0, "to_process": 0})
                            row["discovered"] += 1
                            row["already_indexed" if key == "already_indexed" else "to_process"] += 1
                    other_types = {}
                    for b in sel["not_selected"]:
                        t = taxonomy.broadcast_type_name(taxonomy.broadcast_type_of(b["title"]))
                        other_types[t] = other_types.get(t, 0) + 1
                    rec["broadcasts"] = {"discovered": len(sel["to_process"]) + len(sel["already_indexed"]),
                                         "already_indexed": len(sel["already_indexed"]),
                                         "to_process": len(sel["to_process"]),
                                         "previously_failed": sel["previously_failed"],
                                         "by_type": [by_type[t] for t in taxonomy.BROADCAST_TYPES if t in by_type],
                                         "other_types_in_range": other_types,
                                         "dates": sorted({b["broadcast_at"][:10] for b in sel["to_process"]})}
                    rec["_items"] = sel["to_process"]
                if include_articles:
                    rec["progress"] = "counting articles"
                    total = count_articles_range(client, date_from, date_to)
                    db = vector_store._get_db()
                    try:
                        have = db.execute("SELECT COUNT(*) FROM articles WHERE date BETWEEN ? AND ? "
                                          "AND article_id LIKE 'noa-post-%'", (date_from, date_to)).fetchone()[0]
                    finally:
                        db.close()
                    rec["articles"] = {"discovered": total, "already_indexed": have,
                                       "to_process_at_most": max(0, total - have),
                                       "note": "Exact new/updated counts are known after the listing is read; "
                                               "unchanged articles are skipped."}
                n_b = rec.get("broadcasts", {}).get("to_process", 0)
                n_a = rec.get("articles", {}).get("to_process_at_most", 0)
                problems = []
                if n_b > config.INGEST_MAX_JOB_BROADCASTS:
                    problems.append(f"{n_b} bulletins to process exceeds the per-job limit of "
                                    f"{config.INGEST_MAX_JOB_BROADCASTS}; narrow the range or the types")
                if n_a > config.INGEST_MAX_JOB_ARTICLES:
                    problems.append(f"Up to {n_a} articles exceeds the per-job limit of {config.INGEST_MAX_JOB_ARTICLES}")
                rec["blocking"] = problems
                rec["nothing_to_do"] = n_b == 0 and (not include_articles or n_a == 0)
                rec["requests_made"] = client.requests_made
                rec["status"] = "ready"
                rec["progress"] = "done"
                logger.info("Preview %s ready: %s..%s types=%s | bulletins %s | articles %s | %d requests", pid,
                            date_from, date_to, type_ids, {k: v for k, v in rec.get("broadcasts", {}).items()
                                                           if k in ("discovered", "already_indexed", "to_process")},
                            rec.get("articles"), client.requests_made)
        except Exception as exc:
            rec["status"], rec["error"] = "failed", str(exc)[:300]
            logger.exception("Preview %s failed: %s", pid, exc)

    threading.Thread(target=work, daemon=True, name="newsonair-preview").start()
    return pid


def get_preview(preview_id: str) -> Optional[dict]:
    rec = _previews.get(preview_id)
    return {k: v for k, v in rec.items() if not k.startswith("_")} if rec else None


def is_running() -> bool:
    return _run_lock.locked()


def current_job() -> dict:
    return dict(_current)


def _fetch_post(client: PoliteClient, post_id: str) -> Optional[dict]:
    r = client.get(f"{API}/posts/{post_id}", {"_fields": "id,date,date_gmt,modified,modified_gmt,link,title,content,"
                                                       "excerpt,categories,author,lang"})
    return r.json() if r.status_code == 200 else None


def run(articles: Optional[int] = None, broadcasts: Optional[int] = None, transcribe: bool = True,
        trigger: str = "cli", plan: Optional[dict] = None, date_from: Optional[str] = None,
        date_to: Optional[str] = None, programs: Optional[list[str]] = None, per_day: Optional[int] = None,
        types: Optional[list[str]] = None, items: Optional[list[dict]] = None, retry_failed: bool = False,
        include_articles: Optional[bool] = None, force: bool = False) -> dict:
    """One ingestion job. Only one job runs at a time per process, and never two overlapping jobs.

    Modes
      latest  - no dates: the newest `articles` posts (category mix) and `broadcasts` bulletins.
      range   - date_from/date_to: every post in the range (unless articles=0 / include_articles=False)
                and the bulletins in the range, filtered by broadcast `types` or exact `programs`
                titles and `per_day`; `items` (from a confirmed preview) replaces bulletin discovery.
      retry   - retry_failed: every document currently marked failed (within the range if given),
                ignoring the attempt limit.
    Progress (stage + counts) is written to ingestion_jobs after every step; a cancel request is
    honoured between documents and already indexed documents are kept.
    """
    import vector_store
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError("An ingestion job is already running")
    kind = "retry" if retry_failed else ("range" if date_from else "latest")
    date_to = date_to or date_from
    if kind == "range":
        if include_articles is None:
            include_articles = articles is None or articles > 0
        if not include_articles:
            articles = 0
    elif kind == "latest":
        articles = config.NEWSONAIR_ARTICLE_LIMIT if articles is None else articles
        broadcasts = config.NEWSONAIR_BROADCAST_LIMIT if broadcasts is None else broadcasts
    with_bulletins = kind == "retry" or items is not None or broadcasts is None or broadcasts > 0
    clash = _overlapping_job(date_from, date_to,
                             (set(types or []) if types and not programs else set()) if with_bulletins else None,
                             kind == "retry" or articles is None or articles > 0)
    if clash:
        _run_lock.release()
        raise RuntimeError(f"Job {clash['job_id']} is already ingesting an overlapping date range / type "
                           f"(stage: {clash.get('stage')})")
    job_id = _new_job_id()
    counts = {k: 0 for k in COUNT_KEYS}
    t0 = time.perf_counter()
    status, error = "completed", None
    _current.clear()
    _current.update({"job_id": job_id, "counts": counts, "stage": "starting", "started": time.time(),
                     "date_from": date_from, "date_to": date_to, "types": types or [], "kind": kind})
    last_write = [0.0]

    def stage(name: str, force_write: bool = False) -> None:
        """Report progress; raises Cancelled when a cancel was requested."""
        _current["stage"] = name
        if force_write or time.time() - last_write[0] > 2:
            last_write[0] = time.time()
            if vector_store.update_job_progress(job_id, name, counts):
                raise Cancelled()

    try:
        with trace("NOA", trace_id=job_id):
            params = {"kind": kind, "articles": articles, "broadcasts": broadcasts, "transcribe": transcribe,
                      "date_from": date_from, "date_to": date_to, "programs": programs, "per_day": per_day,
                      "types": list(types or []), "include_articles": include_articles, "force": force,
                      "preselected": items is not None}
            vector_store.start_job(job_id, trigger, params)
            logger.info("Ingestion job %s started (trigger=%s, kind=%s) | source=%s | range=%s..%s | articles=%s "
                        "broadcasts=%s types=%s programs=%s per_day=%s transcribe=%s", job_id, trigger, kind,
                        config.NEWSONAIR_BASE, date_from, date_to, "all" if articles is None else articles,
                        "all" if broadcasts is None else broadcasts, types, programs, per_day, transcribe)
            client = PoliteClient()
            if not client.robots_allows("/wp-json/wp/v2/posts") or not client.robots_allows("/national-bulletins/"):
                raise RuntimeError("robots.txt disallows the NewsOnAIR paths used by this collector")
            retrieved_at = vector_store._now_iso()
            if kind == "retry":
                _retry_failed(client, counts, job_id, date_from, date_to, transcribe, stage, retrieved_at)
            else:
                if articles is None or articles > 0:
                    stage("discovering articles", True)
                    logger.info("Discovering articles via %s/posts (lang=en)", API)
                    posts = (discover_articles_range(client, date_from, date_to, articles) if kind == "range"
                             else discover_articles(client, articles, plan))
                    counts["articles_discovered"] = len(posts)
                    counts["articles_downloaded"] = len(posts)  # full content arrives with the listing call
                    recs = [parse_article(p, retrieved_at) for p in posts]
                    stage("classifying articles", True)
                    _classify_missing_genres(recs)
                    for i, rec in enumerate(recs, 1):
                        stage(f"indexing articles {i}/{len(recs)}")
                        index_article(rec, counts, job_id, force=force)
                        if i % 100 == 0:
                            logger.info("Articles progress: %d/%d (new %d, skipped %d, failed %d)", i, len(recs),
                                        counts["indexed_new"], counts["duplicates_skipped"], counts["failed"])
                if items is not None or broadcasts is None or broadcasts > 0:
                    stage("discovering broadcasts", True)
                    if items is not None:
                        bl = list(items)
                        logger.info("Processing %d bulletins selected in the confirmed preview", len(bl))
                    elif kind == "range":
                        found = scan_archive(client, date_from, date_to)
                        sel = select_bulletins(found, types=types, programs=programs, per_day=per_day,
                                               include_indexed=force)
                        counts["already_indexed"] = len(sel["already_indexed"])
                        bl = sel["to_process"][:broadcasts] if broadcasts else sel["to_process"]
                        logger.info("Selected %d bulletins in %s..%s (%d already indexed)", len(bl), date_from,
                                    date_to, len(sel["already_indexed"]))
                    else:
                        bl = discover_bulletins(client, broadcasts)
                    counts["broadcasts_discovered"] = len(bl) + counts["already_indexed"]
                    for i, b in enumerate(bl, 1):
                        stage(f"bulletin {i}/{len(bl)}", True)
                        index_bulletin(b, client, counts, job_id, transcribe=transcribe, force=force,
                                       stage=lambda name, i=i, n=len(bl): stage(f"{name} {i}/{n}"))
            if counts["failed"]:
                status = "completed_with_errors"
    except Cancelled:
        status, error = "cancelled", "Cancelled on request; documents indexed before the cancel are kept"
        logger.warning("Ingestion job %s cancelled at stage '%s'", job_id, _current.get("stage"))
    except Exception as exc:
        status, error = "failed", str(exc)[:500]
        logger.exception("Ingestion job %s failed: %s", job_id, exc)
    finally:
        counts["duration_s"] = round(time.perf_counter() - t0, 1)
        try:
            vector_store.finish_job(job_id, status, counts, error)
        finally:
            with trace("NOA", trace_id=job_id):
                logger.info("Ingestion job %s %s in %.1fs | discovered: %d articles, %d broadcasts | downloaded: "
                            "%d articles, %d audio files | transcribed: %d/%d | chunks: %d | embeddings: %d | "
                            "indexed: %d new, %d updated | already indexed: %d | duplicates skipped: %d | failed: %d"
                            " | MMR repeats: %d", job_id, status, counts["duration_s"], counts["articles_discovered"],
                            counts["broadcasts_discovered"], counts["articles_downloaded"],
                            counts["broadcasts_downloaded"], counts["transcriptions_ok"], counts["audio_processed"],
                            counts["chunks"], counts["embeddings"], counts["indexed_new"], counts["updated"],
                            counts["already_indexed"], counts["duplicates_skipped"], counts["failed"],
                            counts["mmr_repeats_removed"])
            _current.clear()
            _run_lock.release()
    return {"job_id": job_id, "status": status, "error": error, "counts": counts}


def _new_job_id() -> str:
    """NOA-<date>-<time>, with a suffix when a job already started in the same second."""
    import vector_store
    base = "NOA-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    taken = {j["job_id"] for j in vector_store.list_jobs(20)}
    n, job_id = 1, base
    while job_id in taken:
        n += 1
        job_id = f"{base}-{n}"
    return job_id


def _retry_failed(client: PoliteClient, counts: dict, job_id: str, date_from: Optional[str],
                  date_to: Optional[str], transcribe: bool, stage, retrieved_at: str) -> None:
    """Re-run every failed document (bulletins from their stored listing, articles re-fetched by post id)."""
    import vector_store
    db = vector_store._get_db()
    try:
        rows = [dict(r) for r in db.execute("SELECT * FROM ingestion_state WHERE status = 'failed' ORDER BY doc_id")]
        catalog = {r["bulletin_id"]: dict(r) for r in db.execute("SELECT * FROM bulletin_catalog")}
    finally:
        db.close()
    stage("discovering failed documents", True)
    todo = []
    for r in rows:
        meta = json.loads(r.get("meta_json") or "{}")
        if r["kind"] == "broadcast":
            bid = meta.get("bulletin_id") or r["doc_id"].replace("noa-bulletin-", "")
            b = meta if meta.get("audio_url") else catalog.get(bid)
            if b and not meta.get("audio_url"):
                b = {k: b.get(k) for k in ("bulletin_id", "title", "category", "language", "broadcast_at",
                                           "audio_url", "page_url")}
            d = (b or {}).get("broadcast_at", "")[:10] if b else None
            if date_from and d and not (date_from <= d <= date_to):
                continue
            todo.append(("broadcast", r, b))
        else:
            todo.append(("article", r, meta.get("post_id") or r["doc_id"].replace("noa-post-", "")))
    counts["broadcasts_discovered"] = sum(1 for k, _, _ in todo if k == "broadcast")
    counts["articles_discovered"] = sum(1 for k, _, _ in todo if k == "article")
    logger.info("Retrying %d failed documents (%d bulletins, %d articles)", len(todo), counts["broadcasts_discovered"],
                counts["articles_discovered"])
    for i, (k, r, ref) in enumerate(todo, 1):
        stage(f"retry {i}/{len(todo)}", True)
        if k == "broadcast":
            if not ref or not ref.get("audio_url"):
                counts["failed"] += 1
                logger.error("Cannot retry %s: its listing (audio URL) was not recorded; run a date-range job "
                             "covering %s", r["doc_id"], r.get("title"))
                continue
            index_bulletin(ref, client, counts, job_id, transcribe=transcribe, force=True,
                           stage=lambda name, i=i, n=len(todo): stage(f"{name} {i}/{n}"))
        else:
            post = _fetch_post(client, ref)
            if not post:
                counts["failed"] += 1
                logger.error("Cannot retry %s: post %s not returned by the API", r["doc_id"], ref)
                continue
            cats = client.get(API + "/categories", {"per_page": 100, "lang": "en", "_fields": "id,slug,name"}).json()
            post = _tag_post(post, {c["id"]: c["slug"] for c in cats},
                             {c["slug"]: html.unescape(c["name"]) for c in cats})
            if date_from and not (date_from <= post["date"][:10] <= date_to):
                continue
            counts["articles_downloaded"] += 1
            rec = parse_article(post, retrieved_at)
            _classify_missing_genres([rec])
            index_article(rec, counts, job_id, force=True)


def run_in_background(**kwargs) -> str:
    """Start a job on a daemon thread (dashboard trigger). Returns the job id, or raises on a conflict."""
    if is_running():
        raise RuntimeError("An ingestion job is already running")
    errors: list = []

    def target():
        try:
            run(**kwargs)
        except Exception as exc:  # conflict / lock errors raised before the job registered itself
            errors.append(exc)

    th = threading.Thread(target=target, daemon=True, name="newsonair-ingest")
    th.start()
    for _ in range(250):  # wait until the job registered itself (or refused) so the caller can report it
        if _current.get("job_id") or errors or not th.is_alive():
            break
        time.sleep(0.02)
    if errors:
        raise errors[0]
    return _current.get("job_id", "starting")


def cancel(job_id: str) -> bool:
    import vector_store
    return vector_store.request_cancel(job_id)


def availability_from_jobs() -> int:
    """Record per-date article availability for ranges a past job listed completely.

    A completed date-range job with no article limit and no failures indexed (or found already
    indexed) every post it listed, so the posts now stored for those dates are what the site had.
    """
    import vector_store
    n = 0
    for j in vector_store.list_jobs(500):
        p, c = j.get("params") or {}, j.get("counts") or {}
        if not (p.get("date_from") and p.get("articles") is None and j["status"] == "completed"
                and not c.get("failed") and c.get("articles_discovered")):
            continue
        days = _complete_days(p["date_from"], p.get("date_to") or p["date_from"])
        if not days:
            continue
        db = vector_store._get_db()
        try:
            have = {r[0]: r[1] for r in db.execute(
                "SELECT date, COUNT(*) FROM articles WHERE date BETWEEN ? AND ? AND article_id LIKE 'noa-post-%' "
                "GROUP BY date", (days[0], days[-1]))}
        finally:
            db.close()
        vector_store.record_availability("article", {d: have.get(d, 0) for d in days})
        n += len(days)
    return n


def recover_meta_from_log(log_path: Optional[str] = None) -> int:
    """Fill ingestion_state.meta_json for bulletins indexed before it existed, from the ingestion log.

    The log line "Indexed broadcast <doc> | <title> (<lang>) <dd Mon yyyy HH:MM> | ... | audio <url> | page <url>"
    holds everything needed to retry or re-describe the bulletin.
    """
    import vector_store
    path = log_path or config.LOG_FILE
    if not os.path.isfile(path):
        return 0
    pat = re.compile(r"Indexed broadcast (noa-bulletin-(\d+)) \| (.+?) \((\w+)\) (\d\d \w{3} \d{4} \d\d:\d\d) \|.*?"
                     r"\| audio (\S+) \| page (\S+)")
    found = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                when = dt.datetime.strptime(m[5], "%d %b %Y %H:%M")
                found[m[1]] = {"bulletin_id": m[2], "title": m[3], "category": m[3], "language": m[4],
                               "broadcast_at": when.isoformat(), "audio_url": m[6], "page_url": m[7]}
    n = 0
    for doc_id, meta in found.items():
        st = vector_store.get_state(doc_id)
        if st and not st.get("meta_json"):
            vector_store.set_state(doc_id, meta_json=json.dumps(meta), last_attempt=st.get("last_attempt"))
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and index real NewsOnAIR articles and bulletins.")
    parser.add_argument("--articles", type=int, default=None, help=f"Articles to fetch (default {config.NEWSONAIR_ARTICLE_LIMIT})")
    parser.add_argument("--broadcasts", type=int, default=None,
                        help=f"English bulletins to fetch (default {config.NEWSONAIR_BROADCAST_LIMIT})")
    parser.add_argument("--no-audio", action="store_true", help="Articles only (same as --broadcasts 0)")
    parser.add_argument("--no-transcribe", action="store_true", help="Download audio but do not transcribe/index it")
    parser.add_argument("--status", action="store_true", help="Print dataset counts and recent jobs, then exit")
    parser.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD (site time); enables range mode")
    parser.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD, inclusive (default: --from)")
    parser.add_argument("--programs", help='Range mode: only these bulletin titles, e.g. "News At Nine,Midday News"')
    parser.add_argument("--per-day", type=int, default=None, help="Range mode: at most N bulletins per day")
    parser.add_argument("--types", help="Range mode: broadcast type ids, e.g. midday,news-at-nine,hourly")
    parser.add_argument("--retry-failed", action="store_true", help="Retry every document marked failed")
    parser.add_argument("--force", action="store_true", help="Reprocess documents even if already indexed")
    parser.add_argument("--cancel", metavar="JOB_ID", help="Request cancellation of a running job, then exit")
    parser.add_argument("--recover-meta", action="store_true",
                        help="Fill missing bulletin metadata from the log and source availability from past jobs")
    args = parser.parse_args()
    import vector_store
    if args.cancel:
        print("cancel requested" if cancel(args.cancel) else "no running job with that id")
        return 0
    if args.recover_meta:
        print(f"{recover_meta_from_log()} bulletin records updated; article availability recorded for "
              f"{availability_from_jobs()} dates")
        return 0
    if args.status:
        print(json.dumps({"dataset": vector_store.get_dataset_stats(), "jobs": vector_store.list_jobs(5)},
                         indent=2, default=str))
        return 0
    res = run(articles=args.articles, broadcasts=0 if args.no_audio else args.broadcasts,
              transcribe=not args.no_transcribe, date_from=args.date_from, date_to=args.date_to,
              programs=[p for p in args.programs.split(",") if p.strip()] if args.programs else None,
              per_day=args.per_day, retry_failed=args.retry_failed, force=args.force,
              types=[t.strip() for t in args.types.split(",") if t.strip()] if args.types else None)
    print(json.dumps(res, indent=2))
    return 0 if res["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
