"""
live_stream_ingest.py - Continuous live-radio ingestion for VaniStream.

LIVE_STREAM_URL -> ffmpeg (16 kHz mono PCM) -> rolling windows (default 30 s,
5 s overlap) -> ASR with word timestamps -> overlap-aware stitching ->
the SAME nlp_pipeline (sentences, embeddings, MMR dedup, genre) -> the SAME
vector_store (SQLite + FAISS) -> immediately searchable by rag_engine.

Overlap handling (no lost or doubled words at window boundaries):
  * Each window is transcribed with word timestamps.
  * Words that start inside the trailing overlap are *deferred*: the next
    window hears them again with full context and commits them.
  * Words that start before the last committed word ended are dropped as
    already transcribed.
  * A sentence that is still unfinished at the window boundary is carried
    into the next window, so sentences are not split in two.
  * Remaining near-duplicates (AIR repeating a headline at 09:00 and 09:30)
    are removed by the existing MMR novelty gate, using a rolling window of
    recent segments (config.LIVE_DEDUP_WINDOW_HOURS).

Long-running safety:
  * Any failure in one window (ASR, DB, FAISS) is logged and skipped.
  * Disconnects, ffmpeg exits and stalled streams trigger reconnects with
    exponential backoff (5 s -> 10 s -> ... capped at 5 min). By default it
    retries forever (LIVE_MAX_RETRIES=0), so an overnight run survives
    network drops.
  * Nothing is wiped on start; a restart resumes appending to the corpus.

Audio storage: each window is saved as a small MP3 under
output/audio/<YYYY-MM-DD>/Live-<Lang>-<HHMMSS>-<n>.mp3; every stored sentence
records that file plus its start/end seconds inside it and its absolute IST
wall-clock time. Windows with no speech are not kept.

CLI:
  python live_stream_ingest.py                      # uses LIVE_STREAM_URL
  python live_stream_ingest.py --url <stream> --chunk-seconds 30 --overlap-seconds 5
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import logging
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional

import numpy as np

import config
from tracing import fmt_ts, set_trace_id, trace

logger = logging.getLogger("live")

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2  # 16-bit mono
_SENT_END = re.compile(r"[.!?।]['\")\]]?$")
MAX_CARRY_SECONDS = 45.0  # flush an unfinished sentence after this long
# A connection only counts as healthy (and resets the backoff) after this much audio;
# a stream that connects and drops repeatedly still backs off instead of flapping.
HEALTHY_CONNECTION_SECONDS = 60.0


def audio_backend() -> Optional[str]:
    """'ffmpeg' (binary on PATH), 'pyav' (FFmpeg libraries bundled with faster-whisper) or None."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import av  # noqa: F401
        return "pyav"
    except ImportError:
        return None


def check_ffmpeg() -> None:
    """Verify that audio can be decoded: the ffmpeg binary, or PyAV as a fallback."""
    if audio_backend() is None:
        raise RuntimeError(
            "Neither ffmpeg nor PyAV is available to decode audio. Install ffmpeg "
            "('winget install Gyan.FFmpeg' on Windows, 'sudo apt install ffmpeg' on Linux) "
            "and check with 'ffmpeg -version'."
        )


def public_url(url: Optional[str]) -> Optional[str]:
    """A display/storage form of a stream URL without credentials, query tokens or local directories."""
    if not url:
        return None
    from urllib.parse import urlsplit
    u = urlsplit(url)
    if u.scheme in ("http", "https", "rtmp", "rtsp", "icecast"):
        name = u.path.rstrip("/").rsplit("/", 1)[-1]
        return f"{u.scheme}://{u.hostname or ''}{'/.../' + name if name else ''}"
    return os.path.basename(url) or url


STALE_PLAYLIST_SECONDS = 120


def probe_stream(url: str, timeout: float = 15.0) -> dict:
    """Check a stream URL before starting: syntax, reachability, and that audio (not a web page) is served.

    Returns {"ok": bool, "kind": ..., "detail": str}. For HLS it also checks that the live playlist is
    still being updated and that its newest segment can be fetched (a frozen playlist whose segments
    return 404 means the stream is offline). Nothing is retried or worked around.
    """
    from urllib.parse import urljoin, urlsplit

    import requests
    url = (url or "").strip()
    if os.path.isfile(url):  # checked before URL parsing: a Windows path (C:/...) parses as scheme "c"
        return {"ok": True, "kind": "file", "detail": "local audio file (played through the live pipeline)"}
    parts = urlsplit(url)
    if parts.scheme in ("rtmp", "rtsp"):
        return {"ok": True, "kind": parts.scheme, "detail": f"{parts.scheme.upper()} stream (not probed; the decoder connects directly)"}
    if parts.scheme not in ("http", "https"):
        if parts.scheme in ("", "file") and os.path.isfile(parts.path if parts.scheme else url):
            return {"ok": True, "kind": "file", "detail": "local audio file (played through the live pipeline)"}
        return {"ok": False, "kind": "invalid",
                "detail": "Enter an http(s) stream URL (Icecast/MP3/AAC or an HLS .m3u8 playlist)"}
    if not parts.hostname:
        return {"ok": False, "kind": "invalid", "detail": "The URL has no host name"}
    headers = {"User-Agent": config.NEWSONAIR_USER_AGENT}
    host = parts.hostname
    try:
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "kind": "unreachable", "detail": f"Could not connect to {host}: {type(exc).__name__}"}
    try:
        if r.status_code != 200:
            return {"ok": False, "kind": "http", "detail": f"{host} answered HTTP {r.status_code} for this URL"}
        ctype = (r.headers.get("content-type") or "").lower()
        head = next(r.iter_content(65536), b"")
    finally:
        r.close()
    text = head[:4096].decode("utf-8", "ignore")
    if "mpegurl" in ctype or text.lstrip().startswith("#EXTM3U"):
        playlist = text
        if "#EXT-X-STREAM-INF" in playlist:  # master playlist -> first variant
            variant = next((ln for ln in playlist.splitlines() if ln and not ln.startswith("#")), None)
            if not variant:
                return {"ok": False, "kind": "hls", "detail": "The HLS master playlist lists no streams"}
            url = urljoin(url, variant)
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
            except requests.exceptions.RequestException as exc:
                return {"ok": False, "kind": "hls", "detail": f"Variant playlist unreachable: {type(exc).__name__}"}
            if r.status_code != 200:
                return {"ok": False, "kind": "hls", "detail": f"Variant playlist answered HTTP {r.status_code}"}
            playlist = r.text
            modified = r.headers.get("last-modified")
        else:
            modified = None
        segs = [ln for ln in playlist.splitlines() if ln and not ln.startswith("#")]
        if not segs:
            return {"ok": False, "kind": "hls", "detail": "The HLS playlist lists no audio segments"}
        if "#EXT-X-ENDLIST" in playlist:
            return {"ok": True, "kind": "hls-vod", "detail": "HLS recording (not live): it will be ingested once and then end"}
        try:
            seg = requests.get(urljoin(url, segs[-1]), headers=headers, timeout=timeout, stream=True)
            seg_status = seg.status_code
            seg.close()
        except requests.exceptions.RequestException as exc:
            return {"ok": False, "kind": "hls", "detail": f"Audio segment unreachable: {type(exc).__name__}"}
        stale = ""
        if modified:
            import email.utils
            age = (dt.datetime.now(dt.timezone.utc) - email.utils.parsedate_to_datetime(modified)).total_seconds()
            if age > STALE_PLAYLIST_SECONDS:
                stale = f"the playlist has not been updated since {email.utils.parsedate_to_datetime(modified):%Y-%m-%d %H:%M} UTC"
        if seg_status != 200 or stale:
            why = "; ".join(x for x in (stale, f"its newest audio segment answers HTTP {seg_status}" if seg_status != 200 else "") if x)
            return {"ok": False, "kind": "hls-offline", "detail": f"The stream at {host} appears to be offline: {why}"}
        return {"ok": True, "kind": "hls", "detail": f"Live HLS stream at {host}"}
    if "text/html" in ctype or text.lstrip().lower().startswith(("<!doctype", "<html")):
        return {"ok": False, "kind": "html",
                "detail": "This URL returns a web page, not audio. Use the direct stream address (.mp3/.aac/.m3u8)"}
    if not head:
        return {"ok": False, "kind": "empty", "detail": f"{host} returned no data"}
    return {"ok": True, "kind": "audio", "detail": f"Audio stream at {host} ({ctype or 'unknown type'})"}


class _PyAVProcess:
    """Decodes a stream with PyAV and exposes it like `ffmpeg ... -f s16le -` (stdout, poll, terminate).

    Used when the ffmpeg binary is not installed. The open/read timeout makes a dead stream end the
    'process' instead of hanging, so the normal reconnect logic applies.
    """

    def __init__(self, url: str, stderr: "collections.deque[str]"):
        r, w = os.pipe()
        self.stdout = os.fdopen(r, "rb", buffering=0)
        self._w = os.fdopen(w, "wb", buffering=0)
        self._stderr = stderr
        self._stop = threading.Event()
        self.returncode: Optional[int] = None
        self._t = threading.Thread(target=self._pump, args=(url,), daemon=True, name="pyav-decode")
        self._t.start()

    def _pump(self, url: str) -> None:
        import av
        code = 0
        try:
            opts = {"rw_timeout": str(int(config.LIVE_READ_TIMEOUT_SECONDS * 1e6))}
            with av.open(url, options=opts, timeout=(30.0, config.LIVE_READ_TIMEOUT_SECONDS)) as c:
                stream = c.streams.audio[0]
                rs = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
                for frame in c.decode(stream):
                    if self._stop.is_set():
                        break
                    for out in rs.resample(frame):
                        self._w.write(bytes(out.planes[0])[: out.samples * 2])
        except (BrokenPipeError, OSError, ValueError) as exc:
            if not self._stop.is_set():
                code = 1
                self._stderr.append(f"pyav: {exc}")
        except Exception as exc:  # av errors (connection, decode, timeout)
            code = 1
            self._stderr.append(f"pyav: {exc}")
        finally:
            self.returncode = code
            try:
                self._w.close()
            except OSError:
                pass

    def poll(self) -> Optional[int]:
        return self.returncode

    def terminate(self) -> None:
        self._stop.set()
        try:
            self.stdout.close()
        except OSError:
            pass

    kill = terminate

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        self._t.join(timeout)
        return self.returncode


def save_pcm_as_mp3(pcm_bytes: bytes, mp3_path: str, sample_rate: int = SAMPLE_RATE) -> bool:
    """Encode raw 16 kHz mono s16le PCM to a compact MP3 (48 kbps mono, speech quality)."""
    os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
    if not shutil.which("ffmpeg"):
        return _save_pcm_pyav(pcm_bytes, mp3_path, sample_rate)
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate),
           "-ac", "1", "-i", "-", "-codec:a", "libmp3lame", "-b:a", "48k", mp3_path]
    proc = subprocess.run(cmd, input=pcm_bytes, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        logger.warning("MP3 encode failed for %s: %s", mp3_path, proc.stderr.decode("utf-8", "ignore")[-200:])
        return False
    return True


def _save_pcm_pyav(pcm_bytes: bytes, mp3_path: str, sample_rate: int) -> bool:
    """MP3 encode with PyAV (ffmpeg binary missing)."""
    try:
        import av
        import numpy as _np
        samples = _np.frombuffer(pcm_bytes, dtype=_np.int16).reshape(1, -1)
        with av.open(mp3_path, "w", format="mp3") as out:
            st = out.add_stream("mp3", rate=sample_rate)
            st.bit_rate = 48000
            st.layout = "mono"
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = sample_rate
            for pkt in st.encode(frame):
                out.mux(pkt)
            for pkt in st.encode(None):
                out.mux(pkt)
        return True
    except Exception as exc:
        logger.warning("MP3 encode (PyAV) failed for %s: %s", mp3_path, exc)
        return False


class _PipeReader:
    """Reads ffmpeg stdout on a background thread so reads can time out.

    A plain pipe.read() blocks forever when a stream stalls without closing;
    the timeout lets the main loop detect that and reconnect.
    """

    def __init__(self, pipe):
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._buf = bytearray()
        self.eof = False
        self._t = threading.Thread(target=self._pump, args=(pipe,), daemon=True)
        self._t.start()

    def _pump(self, pipe):
        try:
            while True:
                data = pipe.read(8192)
                if not data:
                    break
                self._q.put(data)
        except (OSError, ValueError):
            pass
        finally:
            self._q.put(b"")

    def read(self, n: int, timeout: float, stop: threading.Event) -> bytes:
        """Return exactly n bytes, or fewer on EOF / timeout / stop."""
        deadline = time.monotonic() + timeout
        while len(self._buf) < n and not self.eof and not stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                data = self._q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if not data:
                self.eof = True
                break
            self._buf.extend(data)
            deadline = time.monotonic() + timeout  # timeout is per stall, not per window
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


def _strip_repeated_prefix(prev_words: list, new_words: list, max_n: int = 6) -> list:
    """Drop leading words of new_words that exactly repeat the tail of prev_words (>= 2 words)."""
    norm = lambda w: re.sub(r"\W+", "", w.text.lower())  # noqa: E731
    for n in range(min(max_n, len(prev_words), len(new_words)), 1, -1):
        if [norm(w) for w in prev_words[-n:]] == [norm(w) for w in new_words[:n]]:
            return new_words[n:]
    return new_words


def select_window_words(words: list, window_start: float, window_len: float, overlap: float,
                        committed_until: float, final: bool) -> tuple[list, list]:
    """Split one window's words (times relative to the window) into (commit, defer).

    commit: words heard for the first time whose start lies before the overlap tail
    defer : words in the overlap tail, which the next window will hear again
    Words starting before `committed_until` (connection-relative seconds) were
    already committed by the previous window and are dropped.
    """
    cutoff = window_len if final else max(0.0, window_len - overlap)
    commit, defer = [], []
    for w in words:
        abs_start = window_start + w.start
        if abs_start < committed_until - 0.05:
            continue
        (commit if w.start < cutoff else defer).append(w)
    return commit, defer


class LiveStreamManager:
    """Runs live ingestion in a background thread and exposes health/status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_stderr: collections.deque[str] = collections.deque(maxlen=20)
        self._reset_state()

    def _reset_state(self):
        self.running = False
        self.url: Optional[str] = None
        self.stream_id: Optional[str] = None
        self.started_at: Optional[str] = None
        self.stream_connected = False
        self.last_audio_received: Optional[str] = None
        self.last_transcription: Optional[str] = None
        self.last_chunk_at: Optional[str] = None
        self.last_segment_id: Optional[int] = None
        self.last_error: Optional[str] = None
        self.segments_ingested = 0
        self.duplicates_rejected = 0
        self.chunks_processed = 0
        self.chunks_failed = 0
        self.reconnects = 0
        self.url_source: Optional[str] = None

    # ------------------------------------------------------------------ API
    def get_status(self) -> dict:
        with self._lock:
            if not self.running:
                state = "error" if self.last_error else "off"
            elif self.stream_connected:
                state = "on"
            else:
                state = "reconnecting" if self.reconnects else "connecting"
            uptime = None
            if self.running and self.started_at:
                uptime = round((config.now_local() - dt.datetime.fromisoformat(self.started_at)).total_seconds())
            return {
                "running": self.running,
                "state": state,               # off | connecting | on | reconnecting | error
                "stream_id": self.stream_id if self.running else None,
                "url": public_url(self.url) if self.running else None,   # never the raw URL (may hold tokens)
                "url_source": self.url_source if self.running else None,  # configured | manual
                "started_at": self.started_at if self.running else None,
                "uptime_seconds": uptime,
                "audio_backend": audio_backend(),
                "configured_url": public_url(config.LIVE_STREAM_URL),
                "stream_connected": self.stream_connected and self.running,
                "last_audio_received": self.last_audio_received,
                "last_transcription": self.last_transcription,
                "last_chunk_at": self.last_chunk_at,
                "last_segment_id": self.last_segment_id,
                "segments_ingested": self.segments_ingested if self.running else 0,
                "duplicates_rejected": self.duplicates_rejected,
                "chunks_processed": self.chunks_processed,
                "chunks_failed": self.chunks_failed,
                "reconnects": self.reconnects,
                "last_error": self.last_error,
            }

    def start(self, url: Optional[str] = None, language: Optional[str] = None,
              chunk_seconds: Optional[int] = None, overlap_seconds: Optional[int] = None,
              fast: bool = True, transcriber: Optional[str] = None, model: Optional[str] = None,
              device: Optional[str] = None, out_dir: Optional[str] = None, csv_path: Optional[str] = None,
              category: str = "live", source_name: Optional[str] = None,
              max_retries: Optional[int] = None, verify: bool = False, programme: Optional[str] = None,
              **_ignored) -> str:
        """Start ingestion in a background thread. Raises ValueError/RuntimeError on bad config.

        verify=True runs probe_stream first and refuses to start a stream that is not serving audio.
        """
        source = "manual" if (url or "").strip() else "configured"
        url = config.require_stream_url(url)
        check_ffmpeg()
        if verify:
            probe = probe_stream(url)
            if not probe["ok"]:
                raise ValueError(probe["detail"])
        chunk = int(chunk_seconds or config.LIVE_CHUNK_SECONDS)
        overlap = int(config.LIVE_OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds)
        if chunk < 5:
            raise ValueError("chunk_seconds must be at least 5")
        overlap = max(0, min(overlap, chunk // 2))
        with self._lock:
            if self.running:
                raise RuntimeError(f"A stream is already active with ID '{self.stream_id}'. Stop it first.")
            self._reset_state()
            self.stream_id = uuid.uuid4().hex[:8]
            self.running = True
            self.url = url
            self.url_source = source
            self.started_at = config.now_local().isoformat(timespec="seconds")
            self._stop_event.clear()
            opts = dict(
                url=url, language=language or config.ASR_LANGUAGE, chunk=chunk, overlap=overlap,
                fast=fast, backend=transcriber, model=model, device=device,
                out_dir=out_dir or config.OUTPUT_DIR,
                csv_path=csv_path or os.path.join(out_dir or config.OUTPUT_DIR, "metadata.csv"),
                category=category, source_name=source_name or config.LIVE_SOURCE_NAME,
                max_retries=config.LIVE_MAX_RETRIES if max_retries is None else max_retries,
                programme=programme,
            )
            self._thread = threading.Thread(target=self._run, kwargs=opts, daemon=True, name="live-ingest")
            self._thread.start()
        logger.info("Live stream %s started (%s url %s, decoder=%s, chunk=%ds, overlap=%ds)", self.stream_id, source,
                    public_url(url), audio_backend(), chunk, overlap)
        return self.stream_id

    def stop(self, stream_id: Optional[str] = None) -> dict:
        """Stop cleanly: the last window is flushed before the thread exits."""
        with self._lock:
            if not self.running:
                return {"status": "not_running", "stream_id": None}
            sid = self.stream_id
            if stream_id and stream_id != sid:
                raise ValueError(f"Stream ID mismatch: active stream is '{sid}', got '{stream_id}'")
            self._stop_event.set()
        self._kill_ffmpeg()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=120)  # allow the final window to be transcribed
        with self._lock:
            self.running = False
        return {"status": "stopped", "stream_id": sid}

    def wait(self) -> None:
        while self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------ internals
    def _spawn_ffmpeg(self, url: str):
        if not shutil.which("ffmpeg"):
            self._ffmpeg_stderr.clear()
            self._ffmpeg_proc = _PyAVProcess(url, self._ffmpeg_stderr)
            return self._ffmpeg_proc
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
        if url.startswith(("http://", "https://")):
            cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                    "-rw_timeout", "30000000"]
        cmd += ["-i", url, "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._ffmpeg_stderr.clear()

        def drain():
            for line in iter(proc.stderr.readline, b""):
                self._ffmpeg_stderr.append(line.decode("utf-8", "ignore").strip())
        threading.Thread(target=drain, daemon=True).start()
        self._ffmpeg_proc = proc
        return proc

    def _kill_ffmpeg(self):
        proc = self._ffmpeg_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _run(self, **opt) -> None:
        set_trace_id(f"LIVE-{self.stream_id}")
        try:
            self._run_inner(**opt)
        except Exception as exc:  # never die silently
            logger.exception("Live ingestion crashed: %s", exc)
            self._set(last_error=str(exc))
        finally:
            self._kill_ffmpeg()
            self._set(running=False, stream_connected=False)
            logger.info("Live loop ended: %d segments stored, %d duplicates rejected, %d windows, %d failed, %d reconnects",
                        self.segments_ingested, self.duplicates_rejected, self.chunks_processed,
                        self.chunks_failed, self.reconnects)

    def _run_inner(self, url, language, chunk, overlap, fast, backend, model, device, out_dir,
                   csv_path, category, source_name, max_retries, programme=None) -> None:
        from transcriber import get_transcriber
        import vector_store

        asr = get_transcriber(backend=backend, model=model, device=device, language=language)
        try:
            asr._load()
        except Exception as exc:
            logger.error("Cannot load ASR model %s: %s", asr.label, exc)
            self._set(last_error=f"ASR model load failed: {exc}")
            return
        logger.info("Models: ASR=%s | embeddings=%s | dedup window=%dh, threshold=%.2f",
                    asr.label, config.EMBEDDING_MODEL, config.LIVE_DEDUP_WINDOW_HOURS, config.MMR_THRESHOLD)

        try:
            vector_store.check_consistency()
            pool_emb, pool_txt = vector_store.get_recent_pool(config.LIVE_DEDUP_WINDOW_HOURS)
        except Exception as exc:
            logger.warning("Could not load dedup pool, starting empty: %s", exc)
            pool_emb, pool_txt = None, []
        self._pool = [pool_emb, list(pool_txt)]
        logger.info("Dedup pool loaded: %d recent segments", len(pool_txt))

        # programme: what the operator says is on air (e.g. "Midday News"); stored as the publisher category so
        # the windows get that broadcast type. Without it, live windows are Other / Unclassified.
        self._ctx = dict(asr=asr, language=language, fast=fast, out_dir=out_dir, csv_path=csv_path,
                         category=category, source_name=source_name, url=url, programme=programme)
        attempt = 0
        window_idx = 0
        while not self._stop_event.is_set():
            proc = self._spawn_ffmpeg(url)
            reader = _PipeReader(proc.stdout)
            conn_wall = None             # wall-clock time of the first byte on this connection
            consumed = 0.0               # seconds of audio received on this connection
            committed_until = 0.0
            overlap_pcm = b""
            carry: list = []             # unfinished-sentence words carried forward
            prev_tail: list = []
            logger.info("Connecting to stream (attempt %d)", attempt + 1)

            while not self._stop_event.is_set():
                need = (chunk - overlap) * BYTES_PER_SEC if overlap_pcm else chunk * BYTES_PER_SEC
                data = reader.read(need, config.LIVE_READ_TIMEOUT_SECONDS, self._stop_event)
                final = len(data) < need
                if not data and not overlap_pcm:
                    break
                if conn_wall is None:
                    if not data:
                        break
                    conn_wall = config.now_local()
                    self._set(stream_connected=True)
                    logger.info("Stream connected")
                now_iso = config.now_local().isoformat(timespec="seconds")
                if data:
                    self._set(last_audio_received=now_iso)
                window_pcm = overlap_pcm + data
                window_start = consumed - len(overlap_pcm) / BYTES_PER_SEC
                consumed += len(data) / BYTES_PER_SEC
                if attempt and consumed >= HEALTHY_CONNECTION_SECONDS:
                    attempt = 0
                if len(window_pcm) < BYTES_PER_SEC:  # < 1 s: nothing useful
                    break
                window_idx += 1
                try:
                    committed_until, carry, prev_tail = self._process_window(
                        window_pcm, window_start, conn_wall, window_idx, overlap,
                        committed_until, carry, prev_tail, final or self._stop_event.is_set())
                    self._set(chunks_processed=self.chunks_processed + 1)
                except Exception as exc:
                    logger.error("Window #%d failed and was skipped: %s", window_idx, exc)
                    self._set(chunks_failed=self.chunks_failed + 1, last_error=f"window {window_idx}: {exc}")
                overlap_pcm = window_pcm[-overlap * BYTES_PER_SEC:] if overlap else b""
                if final:
                    break

            rc = proc.poll()
            self._kill_ffmpeg()
            self._set(stream_connected=False)
            if self._stop_event.is_set():
                break
            err = self._ffmpeg_stderr[-1] if self._ffmpeg_stderr else "no error output"
            reason = "stalled (no audio)" if (rc is None and not reader.eof) else f"ended (ffmpeg rc={rc}: {err})"
            attempt += 1
            self._set(reconnects=self.reconnects + 1)
            if max_retries and attempt > max_retries:
                logger.error("Stream disconnected: %s. Max retries (%d) reached, stopping.", reason, max_retries)
                self._set(last_error=f"stream {reason}; retries exhausted")
                break
            delay = min(config.LIVE_RETRY_BASE_SECONDS * 2 ** (attempt - 1), config.LIVE_RETRY_MAX_SECONDS)
            delay *= random.uniform(0.8, 1.2)
            logger.warning("Stream disconnected: %s. Retry %d in %.0fs", reason, attempt, delay)
            self._set(last_error=f"stream {reason}")
            if self._stop_event.wait(delay):
                break

    def _process_window(self, pcm: bytes, window_start: float, conn_wall: dt.datetime, idx: int,
                        overlap: int, committed_until: float, carry: list, prev_tail: list,
                        final: bool):
        """Transcribe one window, commit its new sentences, and return updated stitching state."""
        import nlp_pipeline
        import vector_store
        from air_bulletins_scraper import sha256_file, write_csv_rows
        from ingest_pipeline import _mark_processed
        from transcriber import Word, align_sentences, pcm16_to_float

        c = self._ctx
        window_len = len(pcm) / BYTES_PER_SEC
        wall_start = conn_wall + dt.timedelta(seconds=window_start)
        date_str = wall_start.strftime("%Y-%m-%d")
        slug = f"Live-{c['language'].capitalize()}-{wall_start.strftime('%H%M%S')}-{idx:04d}"
        audio_path = os.path.join(c["out_dir"], "audio", date_str, f"{slug}.mp3")
        logger.info("Audio window #%d received: %.1fs (%s-%s)", idx, window_len,
                    wall_start.strftime("%H:%M:%S"), (wall_start + dt.timedelta(seconds=window_len)).strftime("%H:%M:%S"))

        result = c["asr"].transcribe(pcm16_to_float(pcm))
        self._set(last_transcription=config.now_local().isoformat(timespec="seconds"))
        words = result.words
        commit, defer = select_window_words(words, window_start, window_len, overlap, committed_until, final)
        commit = _strip_repeated_prefix(prev_tail, commit)
        if commit:
            committed_until = window_start + commit[-1].end
        elif not words:
            committed_until = max(committed_until, window_start + (window_len if final else window_len - overlap))

        # Words carried from the previous window were already re-timed relative to this window's audio file.
        carried = [Word(w.start, w.end, w.text) for w in carry]
        all_words = carried + commit
        text = "".join(w.text for w in all_words).strip()
        logger.info("Transcription completed: %d words heard, %d new, %d deferred to next window",
                    len(words), len(commit), len(defer))

        # Hold back an unfinished last sentence (unless final or it has been carried too long).
        sentences = nlp_pipeline.segment_sentences(text) if text else []
        new_carry: list = []
        if sentences and not final and not _SENT_END.search(sentences[-1]):
            span = (all_words[-1].end - all_words[0].start) if all_words else 0
            if span < MAX_CARRY_SECONDS or len(sentences) > 1:
                n_last = len(re.sub(r"\W+", "", sentences[-1]))
                acc, k = 0, len(all_words)
                while k > 0 and acc < n_last:
                    k -= 1
                    acc += len(re.sub(r"\W+", "", all_words[k].text))
                # carry words are re-timed for the next window (which starts window_len - overlap later)
                step = window_len - overlap
                new_carry = [Word(w.start - step, w.end - step, w.text) for w in all_words[k:]]
                all_words = all_words[:k]
                sentences = sentences[:-1]
                text = "".join(w.text for w in all_words).strip()

        if not sentences:
            if os.path.isfile(audio_path):
                os.remove(audio_path)
            return committed_until, new_carry, (commit or prev_tail)[-6:]

        if not save_pcm_as_mp3(pcm, audio_path):
            logger.warning("Audio for window #%d could not be saved; segments will have no playback", idx)
        rel_audio = os.path.relpath(audio_path, c["out_dir"]).replace("\\", "/")
        text_path = os.path.join(c["out_dir"], "text", date_str, f"{slug}.txt")
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(" ".join(sentences))
        rel_text = os.path.relpath(text_path, c["out_dir"]).replace("\\", "/")
        write_csv_rows(c["csv_path"], [{
            "title": f"{c['source_name']} {date_str} {wall_start.strftime('%H:%M:%S')}",
            "date": date_str, "language": c["language"], "category": c["category"],
            "audio_url": c["url"], "audio_file": rel_audio, "transcript_file": rel_text,
            "sha256": sha256_file(audio_path) if os.path.isfile(audio_path) else "",
            "downloaded_at": wall_start.isoformat(timespec="seconds"),
        }])

        # Sentence timing inside this window's audio file (clamped: carried words may start before it).
        stamps = [(max(0.0, s) if s is not None else None, max(0.0, e) if e is not None else None)
                  for s, e in align_sentences(sentences, all_words)]
        segments, rejections = nlp_pipeline.process_transcript(
            " ".join(sentences),
            existing_embeddings=self._pool[0], existing_texts=self._pool[1],
            fast=c["fast"], mmr_threshold=config.MMR_THRESHOLD, detailed=True,
            sentence_timestamps=stamps,
        )
        for seg in segments:
            a = seg.get("audio_start") or 0.0
            b = seg.get("audio_end") or a
            ts0 = wall_start + dt.timedelta(seconds=a)
            seg["ts_start"] = ts0.isoformat(timespec="seconds")
            seg["ts_end"] = (wall_start + dt.timedelta(seconds=b)).isoformat(timespec="seconds")
            seg["date"] = ts0.strftime("%Y-%m-%d")
            seg["hour"] = int(ts0.strftime("%H%M"))
        if segments:
            vector_store.add_segments(segments, date_str, int(wall_start.strftime("%H%M")), rel_text,
                                      source_name=c["source_name"], source_url=public_url(c["url"]),
                                      ingest_mode="live", language=c["language"],
                                      doc={"source_category": c.get("programme")} if c.get("programme") else None)
            new_embs = np.vstack([s["embedding"] for s in segments])
            self._pool[0] = new_embs if self._pool[0] is None else np.vstack([self._pool[0], new_embs])
            self._pool[1].extend(s["text"] for s in segments)
            for s in segments:
                logger.info("Segment accepted: FAISS ID=%d | genre=%s | %s-%s in %s | \"%s\"",
                            s["faiss_id"], s["genre"], fmt_ts(s.get("audio_start")), fmt_ts(s.get("audio_end")),
                            os.path.basename(audio_path), s["text"][:70])
        for r in rejections:
            logger.info("Duplicate rejected (cos=%.2f): \"%s\"", r["cosine_score"], r["rejected_text"][:70])
        _mark_processed(rel_text)
        self._set(segments_ingested=self.segments_ingested + len(segments),
                  duplicates_rejected=self.duplicates_rejected + len(rejections),
                  last_chunk_at=config.now_local().isoformat(timespec="seconds"),
                  last_segment_id=segments[-1]["faiss_id"] if segments else self.last_segment_id)
        return committed_until, new_carry, commit[-6:] if commit else prev_tail


# Shared instance used by the API (/stream/start, /stream/stop, /health).
live_manager = LiveStreamManager()


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous live-stream transcription & ingestion.")
    parser.add_argument("--url", default=None, help="Stream URL (default: LIVE_STREAM_URL from env/.env)")
    parser.add_argument("--language", default=None, help="Spoken language code, e.g. en, hi (default: VANI_ASR_LANGUAGE)")
    parser.add_argument("--chunk-seconds", type=int, default=None, help="Window length (default LIVE_CHUNK_SECONDS=30)")
    parser.add_argument("--overlap-seconds", type=int, default=None, help="Window overlap (default LIVE_OVERLAP_SECONDS=5)")
    parser.add_argument("--fast", action="store_true", default=True, help="Use the light genre classifier (default)")
    parser.add_argument("--full-classifier", dest="fast", action="store_false", help="Use bart-large-mnli")
    parser.add_argument("--transcriber", default=None, choices=["faster-whisper", "whisper"])
    parser.add_argument("--model", default=None, help="ASR model (default VANI_ASR_MODEL)")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--source-name", default=None, help="Label stored with every segment")
    parser.add_argument("--max-retries", type=int, default=None, help="0 = reconnect forever (default)")
    args = parser.parse_args()

    manager = LiveStreamManager()
    try:
        with trace("LIVE"):
            manager.start(url=args.url, language=args.language, chunk_seconds=args.chunk_seconds,
                          overlap_seconds=args.overlap_seconds, fast=args.fast, transcriber=args.transcriber,
                          model=args.model, device=args.device, source_name=args.source_name,
                          max_retries=args.max_retries)
    except (ValueError, RuntimeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("Live ingestion running. Press Ctrl+C to stop (the current window is flushed first).")
    try:
        manager.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
