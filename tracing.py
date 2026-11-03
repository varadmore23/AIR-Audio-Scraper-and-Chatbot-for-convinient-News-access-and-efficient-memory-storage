"""
tracing.py - Readable, traceable logging for VaniStream.

Every log line carries a trace ID so a faculty reviewer can follow one
request (a chat question, an ingested file, a live audio window) from start
to finish in the terminal:

  [2026-11-02 17:45:18] [INFO ] [REQ-0007] [rag    ] Retrieved chunks: 5

Use `with trace("REQ"):` (or `new_trace_id`) to start a trace, and `timed()`
to log how long a stage took.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import logging
import os
import sys
import time

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_counters: dict[str, itertools.count] = {}
_configured = False


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        name = record.name.replace("vanistream.", "")
        record.short_name = name[:12]
        return True


FORMAT = "[%(asctime)s] [%(levelname)-5s] [%(trace_id)s] [%(short_name)-12s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger once (safe to call repeatedly)."""
    global _configured
    if _configured:
        return
    handlers: list[logging.Handler] = []
    # Never let an unencodable character (e.g. a zero-width joiner in a headline) break a log line
    # on consoles with a legacy code page such as Windows cp1252.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    console = logging.StreamHandler(sys.stdout)
    handlers.append(console)
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            from logging.handlers import RotatingFileHandler
            handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"))
        except OSError:
            pass
    fmt = logging.Formatter(FORMAT, DATEFMT)
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(_TraceFilter())
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=handlers, force=True)
    # Quiet noisy third-party loggers.
    for noisy in ("faiss.loader", "httpx", "urllib3", "sentence_transformers", "transformers",
                  "faster_whisper", "uvicorn.access", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_trace_id(prefix: str = "REQ") -> str:
    counter = _counters.setdefault(prefix, itertools.count(1))
    return f"{prefix}-{next(counter):04d}"


def current_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(trace_id: str) -> contextvars.Token:
    return _trace_id.set(trace_id)


@contextlib.contextmanager
def trace(prefix: str = "REQ", trace_id: str | None = None):
    """Context manager that sets a trace ID for everything logged inside it."""
    tid = trace_id or new_trace_id(prefix)
    token = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


@contextlib.contextmanager
def timed(logger: logging.Logger, stage: str):
    """Log '<stage> completed in X.XXs' (or the failure) around a block."""
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        logger.error("%s failed after %.2fs: %s", stage, time.perf_counter() - t0, exc)
        raise
    logger.info("%s completed in %.2fs", stage, time.perf_counter() - t0)


def fmt_ts(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS (or MM:SS when under an hour)."""
    if seconds is None:
        return "--:--"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
