import argparse
import calendar
import csv
import datetime as dt
import hashlib
import os
import random
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import requests
from bs4 import BeautifulSoup

AJAX_URL = "https://www.newsonair.gov.in/wp-admin/admin-ajax.php"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]
DEFAULT_UA = USER_AGENTS[0]


def rotate_user_agent(session: requests.Session) -> None:
    """Randomly rotate user agent on session."""
    session.headers["User-Agent"] = random.choice(USER_AGENTS)


def check_robots_txt(session: requests.Session, target_url: str = AJAX_URL) -> None:
    """Check robots.txt and warn if target URL is disallowed."""
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url("https://www.newsonair.gov.in/robots.txt")
        resp = session.get("https://www.newsonair.gov.in/robots.txt", timeout=10)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            ua = session.headers.get("User-Agent", "*")
            if not rp.can_fetch(ua, target_url):
                print(f"WARNING: robots.txt disallows fetching {target_url} for user agent.")
            else:
                print("robots.txt check passed: scraping allowed.")
    except Exception as e:
        print(f"Note: Could not verify robots.txt: {e}")


@dataclass(frozen=True)
class BulletinItem:
    title: str
    date_text: str
    date_iso: Optional[str]
    audio_url: str
    time_hint: Optional[int]


def sleep_jitter(min_delay: float, max_delay: float) -> None:
    if max_delay <= 0:
        return
    time.sleep(random.uniform(min_delay, max_delay))


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_date_text(date_text: str) -> Optional[str]:
    try:
        parsed = dt.datetime.strptime(date_text.strip(), "%d %B %Y").date()
        return parsed.isoformat()
    except ValueError:
        return None


def extract_time_hint(audio_url: str) -> Optional[int]:
    base = os.path.splitext(os.path.basename(audio_url))[0]
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", base)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def post_ajax(
    session: requests.Session,
    payload: dict,
    timeout: int,
    max_retries: int,
    min_delay: float,
    max_delay: float,
) -> str:
    rotate_user_agent(session)
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(AJAX_URL, data=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            sleep_jitter(min_delay, max_delay)
    raise RuntimeError("Unreachable retry loop")


def fetch_categories(
    session: requests.Session,
    language: str,
    timeout: int,
    max_retries: int,
    min_delay: float,
    max_delay: float,
) -> list[tuple[str, str]]:
    html = post_ajax(
        session,
        {"action": "get_listen_news_categories", "language": language},
        timeout,
        max_retries,
        min_delay,
        max_delay,
    )
    soup = BeautifulSoup(html, "html.parser")
    options = []
    for opt in soup.select("option"):
        value = (opt.get("value") or "").strip()
        label = opt.get_text(strip=True)
        if not value:
            continue
        options.append((value, label))
    return options


def parse_bulletins(html: str) -> tuple[list[BulletinItem], bool]:
    soup = BeautifulSoup(html, "html.parser")
    no_posts = soup.get_text(" ", strip=True).lower().startswith("no posts found")
    if no_posts:
        return [], False

    items: list[BulletinItem] = []
    for li in soup.select("li"):
        source = li.select_one("audio source")
        if not source:
            continue
        audio_url = (source.get("src") or "").strip()
        if not audio_url:
            continue
        title_el = li.select_one("h3")
        date_el = li.select_one("p")
        title = title_el.get_text(strip=True) if title_el else ""
        date_text = date_el.get_text(strip=True) if date_el else ""
        date_iso = parse_date_text(date_text)
        time_hint = extract_time_hint(audio_url)
        items.append(
            BulletinItem(
                title=title,
                date_text=date_text,
                date_iso=date_iso,
                audio_url=audio_url,
                time_hint=time_hint,
            )
        )

    has_next = soup.select_one("nav.pagination a.next") is not None
    return items, has_next


def fetch_bulletins(
    session: requests.Session,
    language: str,
    category: str,
    date_from: str,
    date_to: str,
    timeout: int,
    max_retries: int,
    min_delay: float,
    max_delay: float,
    max_pages: int,
) -> list[BulletinItem]:
    all_items: list[BulletinItem] = []
    page = 1
    while True:
        payload = {
            "action": "filter_bulletins_audio",
            "language": language,
            "category": category,
            "date_from": date_from,
            "date_to": date_to,
            "paged": str(page),
        }
        html = post_ajax(session, payload, timeout, max_retries, min_delay, max_delay)
        items, has_next = parse_bulletins(html)
        if not items:
            break
        all_items.extend(items)
        if not has_next or page >= max_pages:
            break
        page += 1
        sleep_jitter(min_delay, max_delay)
    return all_items


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("_") or "bulletin"


def download_audio(
    session: requests.Session,
    url: str,
    dest_path: str,
    timeout: int,
    max_retries: int,
    min_delay: float,
    max_delay: float,
) -> None:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return

    rotate_user_agent(session)
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            sleep_jitter(min_delay, max_delay)


def strip_overlap(prev_text: str, curr_text: str, min_overlap: int = 80) -> str:
    if not prev_text or not curr_text:
        return curr_text

    prev_norm = " ".join(prev_text.split())
    curr_norm = " ".join(curr_text.split())
    max_scan = min(len(prev_norm), len(curr_norm), 2000)

    for size in range(max_scan, min_overlap - 1, -1):
        if prev_norm[-size:] == curr_norm[:size]:
            trimmed = curr_norm[size:].lstrip()
            return trimmed

    return curr_text


def transcribe_audio(
    audio_path: str,
    transcriber: str,
    model_name: str,
    device: str,
) -> str:
    """Transcribe one bulletin. Uses the shared, cached model (see transcriber.py)."""
    return transcribe_audio_detailed(audio_path, transcriber, model_name, device).text


def transcribe_audio_detailed(audio_path: str, transcriber: str, model_name: str, device: str):
    """Return a transcriber.TranscriptionResult with segment and word timestamps."""
    from transcriber import get_transcriber
    return get_transcriber(backend=transcriber, model=model_name, device=device).transcribe(audio_path)


def save_segments_json(transcript_path: str, result) -> str:
    """Write `<transcript>.segments.json` next to the .txt so ingestion can
    attach audio start/end times to every sentence."""
    import json
    seg_path = os.path.splitext(transcript_path)[0] + ".segments.json"
    with open(seg_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_json(), fh, ensure_ascii=False)
    return seg_path


def parse_month_arg(month_arg: str) -> tuple[str, str]:
    parts = month_arg.split("-")
    if len(parts) != 2:
        raise ValueError("month must be in YYYY-MM format")
    year = int(parts[0])
    month = int(parts[1])
    last_day = calendar.monthrange(year, month)[1]
    start = dt.date(year, month, 1).isoformat()
    end = dt.date(year, month, last_day).isoformat()
    return start, end


def load_existing_urls(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    urls = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("audio_url") or "").strip()
            if url:
                urls.add(url)
    return urls


def write_csv_rows(csv_path: str, rows: Iterable[dict]) -> None:
    fieldnames = [
        "title",
        "date",
        "language",
        "category",
        "audio_url",
        "audio_file",
        "transcript_file",
        "sha256",
        "downloaded_at",
    ]
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_category(options: list[tuple[str, str]], user_value: str) -> str:
    values = {value: label for value, label in options}
    if user_value in values:
        return user_value
    lowered = user_value.strip().lower()
    for value, label in options:
        if lowered == label.lower():
            return value
    for value, label in options:
        if lowered in label.lower():
            return value
    raise ValueError("Category not found. Use --list-categories to see options.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape AIR main audio bulletins.")
    parser.add_argument("--language", default="english", help="english|hindi|urdu")
    parser.add_argument("--category", default="hourly-news", help="Category slug or label")
    parser.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD")
    parser.add_argument("--month", help="Month in YYYY-MM format")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--csv", default="output/metadata.csv", help="CSV path")
    parser.add_argument("--min-delay", type=float, default=1.0, help="Min delay between requests")
    parser.add_argument("--max-delay", type=float, default=3.0, help="Max delay between requests")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count")
    parser.add_argument("--max-pages", type=int, default=200, help="Max pagination pages")
    parser.add_argument("--list-categories", action="store_true", help="List categories and exit")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe audio to text")
    parser.add_argument("--transcriber", default=None, help="faster-whisper|whisper (default: VANI_ASR_BACKEND)")
    parser.add_argument("--model", default=None, help="Whisper model name (default: VANI_ASR_MODEL)")
    parser.add_argument("--device", default=None, help="cpu|cuda (default: VANI_ASR_DEVICE)")
    parser.add_argument("--dedupe", action="store_true", help="Remove overlap vs previous bulletin")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="HTTP user agent")
    args = parser.parse_args()

    if args.month:
        date_from, date_to = parse_month_arg(args.month)
    else:
        if not args.date_from or not args.date_to:
            parser.error("Provide --from and --to or --month")
        date_from = args.date_from
        date_to = args.date_to

    session = make_session(args.user_agent)
    check_robots_txt(session)

    categories = fetch_categories(
        session,
        args.language,
        args.timeout,
        args.max_retries,
        args.min_delay,
        args.max_delay,
    )

    if args.list_categories:
        for value, label in categories:
            print(f"{value}\t{label}")
        return 0

    category = resolve_category(categories, args.category)

    bulletins = fetch_bulletins(
        session,
        args.language,
        category,
        date_from,
        date_to,
        args.timeout,
        args.max_retries,
        args.min_delay,
        args.max_delay,
        args.max_pages,
    )

    if not bulletins:
        print("No bulletins found for the given range.")
        return 0

    existing_urls = load_existing_urls(args.csv)

    def sort_key(item: BulletinItem) -> tuple:
        date_key = item.date_iso or "0000-00-00"
        time_key = item.time_hint if item.time_hint is not None else 9999
        return (date_key, time_key, item.audio_url)

    saved_count = 0
    prev_text = ""
    for item in sorted(bulletins, key=sort_key):
        if item.audio_url in existing_urls:
            continue

        date_folder = item.date_iso or "unknown_date"
        audio_name = safe_name(os.path.splitext(os.path.basename(item.audio_url))[0]) + ".mp3"
        audio_path = os.path.join(args.out_dir, "audio", date_folder, audio_name)

        download_audio(
            session,
            item.audio_url,
            audio_path,
            args.timeout,
            args.max_retries,
            args.min_delay,
            args.max_delay,
        )

        transcript_path = ""
        transcript_text = ""
        if args.transcribe:
            result = None
            try:
                result = transcribe_audio_detailed(audio_path, args.transcriber, args.model, args.device)
                transcript_text = result.text
            except Exception as exc:  # one bad file must not stop the whole scrape
                print(f"WARNING: transcription failed for {audio_path}: {exc}")
            raw_text = transcript_text
            if args.dedupe:
                transcript_text = strip_overlap(prev_text, transcript_text)
            prev_text = transcript_text

            if transcript_text:
                transcript_name = safe_name(os.path.splitext(audio_name)[0]) + ".txt"
                transcript_path = os.path.join(args.out_dir, "text", date_folder, transcript_name)
                os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
                with open(transcript_path, "w", encoding="utf-8") as handle:
                    handle.write(transcript_text)
                # Timestamps stay valid only if the text was not trimmed.
                if result is not None and transcript_text == raw_text:
                    save_segments_json(transcript_path, result)

        file_hash = sha256_file(audio_path)
        row = {
            "title": item.title,
            "date": item.date_iso or item.date_text,
            "language": args.language,
            "category": category,
            "audio_url": item.audio_url,
            "audio_file": os.path.relpath(audio_path, args.out_dir),
            "transcript_file": os.path.relpath(transcript_path, args.out_dir)
            if transcript_path
            else "",
            "sha256": file_hash,
            "downloaded_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        # Write immediately to CSV after each item is processed
        write_csv_rows(args.csv, [row])
        existing_urls.add(item.audio_url)
        saved_count += 1

        sleep_jitter(args.min_delay, args.max_delay)

    if saved_count > 0:
        print(f"Saved {saved_count} items to {args.csv}")
    else:
        print("No new items to save.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
