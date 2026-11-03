"""
news_articles.py - Detailed news articles: loading, chunking and RAG ingestion.

The landing page shows a set of long-form articles (data/demo_articles.json).
They are ILLUSTRATIVE DEMO CONTENT: fictional organisations, people and
figures, attributed to an in-house demo desk, never to AIR or another real
news organisation.

Chunking keeps the article's structure:
  - one chunk per section for the short sections (summary, key developments,
    people and organisations, key takeaways)
  - prose sections (report, background, implications) are packed paragraph by
    paragraph into chunks of up to CHUNK_WORDS words; a paragraph is never
    split, and nothing is dropped
Every chunk carries the article id, section, genre, date and time. Ingestion
checks that the chunks together contain exactly the words of the full
article, so an article can never be truncated silently. The complete article
is also stored in the `articles` table, and answers use the whole article as
context, not only the matched chunk.

CLI:
  python news_articles.py --list
  python news_articles.py --chunks <article_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import config

logger = logging.getLogger("articles")

ARTICLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "demo_articles.json")
SOURCE_PREFIX = "articles/"
# all-MiniLM-L6-v2 reads 256 word-pieces; 150 words plus the headline stays inside that window.
CHUNK_WORDS = 150

SECTIONS = [  # (key in the JSON, display title, kind)
    ("summary", "Summary", "text"),
    ("body", "Report", "paragraphs"),
    ("background", "Background", "paragraphs"),
    ("key_developments", "Key developments", "list"),
    ("entities", "People and organisations", "entities"),
    ("implications", "Implications", "paragraphs"),
    ("takeaways", "Key takeaways", "list"),
]

_cache: dict[str, tuple[float, dict]] = {}


def load_dataset(path: str = ARTICLES_PATH) -> dict:
    """The whole dataset (cached until the file changes)."""
    mtime = os.path.getmtime(path)
    if path in _cache and _cache[path][0] == mtime:
        return _cache[path][1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for a in data["articles"]:
        a.setdefault("source_name", data.get("source_name", "Demo desk"))
        a.setdefault("demo", True)
        a["hour"] = int(a.get("time", "00:00").replace(":", ""))
        a["word_count"] = len(full_text(a).split())
    _cache[path] = (mtime, data)
    return data


def load_articles(path: str = ARTICLES_PATH) -> list[dict]:
    return load_dataset(path)["articles"]


def get_article(article_id: str, path: str = ARTICLES_PATH) -> Optional[dict]:
    return next((a for a in load_articles(path) if a["id"] == article_id), None)


def _entity_line(e: dict) -> str:
    return f"{e['name']}: {e['role']}."


def section_items(article: dict, key: str, kind: str) -> list[str]:
    """The text items of one section in reading order."""
    value = article.get(key)
    if not value:
        return []
    if kind == "text":
        return [value]
    if kind == "entities":
        return [_entity_line(e) for e in value]
    return list(value)


def full_text(article: dict) -> str:
    """The complete article as plain text with section headings (used as RAG context)."""
    parts = [article["headline"]]
    for key, title, kind in SECTIONS:
        items = section_items(article, key, kind)
        if not items:
            continue
        if kind in ("list", "entities"):
            parts.append(f"{title}:\n" + "\n".join(f"- {i}" for i in items))
        else:
            parts.append(f"{title}:\n" + "\n\n".join(items))
    return "\n\n".join(parts)


def chunk_article(article: dict, max_words: int = CHUNK_WORDS) -> list[dict]:
    """Section-aware chunks: [{'section', 'text', 'chunk_index'}] covering the whole article."""
    chunks: list[dict] = []
    for key, title, kind in SECTIONS:
        items = section_items(article, key, kind)
        if not items:
            continue
        if kind in ("list", "entities"):
            chunks.append({"section": title, "text": "\n".join(f"- {i}" for i in items)})
            continue
        buf: list[str] = []
        for para in items:
            if buf and len(" ".join(buf + [para]).split()) > max_words:
                chunks.append({"section": title, "text": "\n\n".join(buf)})
                buf = []
            buf.append(para)  # an over-long paragraph becomes its own chunk, never cut
        if buf:
            chunks.append({"section": title, "text": "\n\n".join(buf)})
    for i, c in enumerate(chunks):
        c["chunk_index"] = i
    return chunks


def _content_words(text: str) -> list[str]:
    return [w.lstrip("-") for w in text.split() if w.lstrip("-")]


def verify_coverage(article: dict, chunks: list[dict]) -> None:
    """Raise if the chunks do not contain every word of the article body (no silent truncation)."""
    expected = []
    for key, _title, kind in SECTIONS:
        for item in section_items(article, key, kind):
            expected += _content_words(item)
    got = []
    for c in chunks:
        got += _content_words(c["text"])
    if got != expected:
        raise ValueError(f"Chunking lost content for article {article['id']}: "
                         f"{len(expected)} words expected, {len(got)} in chunks")


def ingest(processed: Optional[set[str]] = None, path: str = ARTICLES_PATH,
           mark_processed=None) -> dict:
    """Embed and store every article not yet in `processed`. Returns counts."""
    import vector_store
    from nlp_pipeline import embed_sentences
    from tracing import trace

    processed = processed or set()
    if not os.path.isfile(path):
        logger.info("No article dataset at %s; skipped", path)
        return {"articles": 0, "chunks": 0}
    n_articles = n_chunks = 0
    for article in load_articles(path):
        source_file = SOURCE_PREFIX + article["id"]
        if source_file in processed:
            continue
        with trace("ART"):
            chunks = chunk_article(article)
            verify_coverage(article, chunks)
            # Embedding input carries headline + section so a chunk is understood in context.
            embs = embed_sentences([f"{article['headline']}. {c['section']}. {c['text']}" for c in chunks])
            segments = [{
                "text": c["text"],
                "search_text": f"{article['headline']}\n{c['section']}\n{c['text']}",
                "genre": article["genre"], "genre_confidence": 1.0, "embedding": embs[i],
                "article_id": article["id"], "section": c["section"],
            } for i, c in enumerate(chunks)]
            text = full_text(article)
            vector_store.upsert_article(article, text)
            vector_store.add_segments(segments, article["published"], article["hour"], source_file,
                                      source_name=article["source_name"], ingest_mode="article",
                                      language="en")
            logger.info("Article ingested: %s | %s | %s %s | %d words -> %d chunks (%s)",
                        article["id"], article["genre"], article["published"], article["time"],
                        len(text.split()), len(chunks), ", ".join(c["section"] for c in chunks))
            if mark_processed:
                mark_processed(source_file)
            n_articles += 1
            n_chunks += len(chunks)
    if n_articles:
        logger.info("Articles complete: %d articles, %d chunks stored (no MMR dedup on articles)",
                    n_articles, n_chunks)
    return {"articles": n_articles, "chunks": n_chunks}


REAL_ATTRIBUTION = ("Articles and bulletins sourced from NewsOnAIR (newsonair.gov.in), the news platform of "
                    "Akashvani / All India Radio. Content belongs to its publisher; VaniStream is an independent "
                    "academic project and is not affiliated with Prasar Bharati.")


def indexed_articles() -> list[dict]:
    """Articles that are actually indexed (the `articles` table), newest first.

    This is what the newsroom shows: real NewsOnAIR articles after an ingestion
    run, or the demo set if that is what was ingested.
    """
    import vector_store
    arts = vector_store.list_articles()
    for a in arts:
        a.setdefault("hour", int((a.get("time") or "00:00").replace(":", "")))
        a.setdefault("word_count", len(full_text(a).split()))
    arts.sort(key=lambda a: (a.get("published") or "", a.get("hour") or 0), reverse=True)
    return arts


def find_article(article_id: str) -> Optional[dict]:
    return next((a for a in indexed_articles() if a["id"] == article_id), None)


def disclaimer_for(articles: list[dict]) -> Optional[str]:
    if not articles:
        return None
    if all(a.get("demo") for a in articles):
        return load_dataset().get("disclaimer")
    return REAL_ATTRIBUTION


def _teaser(article: dict, words: int = 45) -> str:
    """Card text: the article's own summary/excerpt, else its opening words (display only)."""
    text = article.get("summary") or article.get("excerpt") or " ".join(article.get("body") or [])
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def preview(article: dict) -> dict:
    """Landing-page card data: metadata and a short teaser only, never the body."""
    keys = ("id", "headline", "genre", "published", "time", "source_name", "source_url", "source_category",
            "word_count", "demo", "tags", "content_limited")
    d = {k: article.get(k) for k in keys}
    d["summary"] = _teaser(article)
    d["read_minutes"] = max(1, round((article.get("word_count") or 0) / 200))
    return d


def related(article: dict, limit: int = 3, pool: Optional[list[dict]] = None) -> list[dict]:
    """Other articles in the same genre, newest first."""
    pool = pool if pool is not None else indexed_articles()
    same = [a for a in pool if a.get("genre") == article.get("genre") and a["id"] != article["id"]]
    same.sort(key=lambda a: (a.get("published") or "", a.get("hour") or 0), reverse=True)
    return [preview(a) for a in same[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the demo article dataset.")
    parser.add_argument("--list", action="store_true", help="List articles with word counts")
    parser.add_argument("--chunks", metavar="ARTICLE_ID", help="Show how one article is chunked")
    args = parser.parse_args()
    if args.chunks:
        a = get_article(args.chunks)
        if not a:
            print(f"No article {args.chunks}")
            return 1
        chunks = chunk_article(a)
        verify_coverage(a, chunks)
        for c in chunks:
            print(f"--- [{c['chunk_index']}] {c['section']} ({len(c['text'].split())} words)\n{c['text']}\n")
    else:
        for a in load_articles():
            print(f"{a['published']} {a['time']}  {a['genre']:<22} {a['word_count']:>4} words  {a['id']}")
    return 0


if __name__ == "__main__":
    _ = config  # logging setup
    raise SystemExit(main())
