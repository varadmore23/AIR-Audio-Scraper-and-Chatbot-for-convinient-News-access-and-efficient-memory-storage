"""
transcriber.py - Single speech-recognition interface for VaniStream.

Both the archive scraper and the live stream use this module, so the model is
loaded once per process and the backend can be swapped from config without
touching the rest of the pipeline.

Pipeline per audio input:
  ffmpeg decode -> 16 kHz mono float32 -> light normalisation (high-pass +
  dynamic loudness) -> silence/empty check -> ASR with VAD -> segments with
  word-level timestamps.

Long audio: faster-whisper processes audio in 30 s windows internally and the
Silero VAD filter skips silence/music, so a full 10-15 min bulletin can be
passed directly. Timestamps are relative to the start of the audio (plus an
optional `offset`).

Backends:
  faster-whisper (default) - CTranslate2 Whisper, int8 on CPU (MIT licence)
  whisper                  - original openai-whisper (MIT), kept for parity
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

import config

logger = logging.getLogger("asr")

SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.5
SILENCE_RMS = 1e-4
# High-pass removes hum/rumble; dynaudnorm evens out quiet/loud presenters.
PREPROCESS_FILTER = "highpass=f=60,dynaudnorm=f=250:g=15"


class AudioError(Exception):
    """Raised for unreadable, unsupported or corrupt audio input."""


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: Optional[str]
    duration: float
    model: str
    elapsed: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip()).strip()

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def to_json(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 2),
            "model": self.model,
            "segments": [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                 "words": [[round(w.start, 2), round(w.end, 2), w.text] for w in s.words]}
                for s in self.segments
            ],
        }


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def load_audio(path: str, preprocess: bool = True) -> np.ndarray:
    """Decode any ffmpeg-readable file to 16 kHz mono float32 in [-1, 1]."""
    if not os.path.isfile(path):
        raise AudioError(f"Audio file not found: {path}")
    if os.path.getsize(path) == 0:
        raise AudioError(f"Audio file is empty: {path}")
    if not shutil.which("ffmpeg"):
        return _load_audio_pyav(path)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-vn"]
    if preprocess:
        cmd += ["-af", PREPROCESS_FILTER]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "ignore").strip().splitlines()
        raise AudioError(f"Unsupported or corrupt audio '{os.path.basename(path)}': {err[-1] if err else 'no audio stream'}")
    return pcm16_to_float(proc.stdout)


def _load_audio_pyav(path: str) -> np.ndarray:
    """Fallback decoder when the ffmpeg binary is missing: PyAV (bundled with faster-whisper).

    It decodes and resamples to 16 kHz mono, but the ffmpeg high-pass/loudness
    preprocessing is skipped, which is logged once.
    """
    try:
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        raise AudioError("ffmpeg is not installed and PyAV (faster-whisper) is unavailable") from exc
    global _warned_no_ffmpeg
    if not _warned_no_ffmpeg:
        logger.warning("ffmpeg not found on PATH: decoding with PyAV, audio preprocessing skipped")
        _warned_no_ffmpeg = True
    try:
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        raise AudioError(f"Unsupported or corrupt audio '{os.path.basename(path)}': {exc}") from exc
    if audio.size == 0:
        raise AudioError(f"No audio stream in '{os.path.basename(path)}'")
    return audio.astype(np.float32)


_warned_no_ffmpeg = False


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def is_silent(audio: np.ndarray) -> bool:
    if audio.size < SAMPLE_RATE * MIN_AUDIO_SECONDS:
        return True
    return float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """Loads one ASR model and transcribes audio files or arrays."""

    def __init__(self, backend: str, model: str, device: str, compute_type: str,
                 language: Optional[str], beam_size: int = 5):
        self.backend = backend
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = None if (language or "auto").lower() == "auto" else _lang_code(language)
        self.beam_size = beam_size
        self._model = None
        self._lock = threading.Lock()  # models are not guaranteed thread-safe

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.model_name}"

    def _load(self):
        if self._model is not None:
            return self._model
        t0 = time.perf_counter()
        if self.backend == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("faster-whisper is not installed (pip install faster-whisper)") from exc
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        elif self.backend == "whisper":
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError("openai-whisper is not installed (pip install openai-whisper)") from exc
            self._model = whisper.load_model(self.model_name, device=self.device)
        else:
            raise ValueError(f"Unknown ASR backend: {self.backend}")
        logger.info("ASR model loaded: %s on %s (%s) in %.1fs", self.label, self.device,
                    self.compute_type, time.perf_counter() - t0)
        return self._model

    def transcribe(self, audio: Union[str, np.ndarray], offset: float = 0.0,
                   word_timestamps: bool = True) -> TranscriptionResult:
        """Transcribe a file path or a 16 kHz float32 array.

        Returns an empty result (no exception) for silent/too-short audio.
        Raises AudioError for unreadable input and RuntimeError for ASR failures.
        """
        t0 = time.perf_counter()
        if isinstance(audio, str):
            audio = load_audio(audio)
        duration = audio.size / SAMPLE_RATE
        if is_silent(audio):
            logger.warning("Skipping ASR: audio is empty or silent (%.1fs)", duration)
            return TranscriptionResult([], self.language, duration, self.label, 0.0)

        model = self._load()
        segments: list[Segment] = []
        language = self.language
        with self._lock:
            try:
                if self.backend == "faster-whisper":
                    it, info = model.transcribe(
                        audio,
                        language=self.language,
                        beam_size=self.beam_size,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 500},
                        word_timestamps=word_timestamps,
                        condition_on_previous_text=False,  # avoids repetition loops on long audio
                    )
                    language = info.language
                    for s in it:
                        words = [Word(w.start + offset, w.end + offset, w.word) for w in (s.words or [])]
                        segments.append(Segment(s.start + offset, s.end + offset, s.text.strip(), words))
                else:
                    res = model.transcribe(audio, language=self.language, word_timestamps=word_timestamps,
                                           condition_on_previous_text=False, fp16=(self.device == "cuda"))
                    language = res.get("language", language)
                    for s in res.get("segments", []):
                        words = [Word(w["start"] + offset, w["end"] + offset, w["word"]) for w in s.get("words", [])]
                        segments.append(Segment(s["start"] + offset, s["end"] + offset, s["text"].strip(), words))
            except Exception as exc:
                raise RuntimeError(f"ASR failed ({self.label}): {exc}") from exc
        segments = [s for s in segments if s.text]
        elapsed = time.perf_counter() - t0
        logger.info("Transcribed %.1fs of audio -> %d segments in %.1fs (%s, lang=%s)",
                    duration, len(segments), elapsed, self.label, language)
        return TranscriptionResult(segments, language, duration, self.label, elapsed)


_LANG_ALIASES = {"english": "en", "hindi": "hi", "urdu": "ur", "bengali": "bn", "tamil": "ta",
                 "telugu": "te", "marathi": "mr", "gujarati": "gu", "kannada": "kn",
                 "malayalam": "ml", "punjabi": "pa", "odia": "or", "assamese": "as"}


def _lang_code(lang: str) -> str:
    return _LANG_ALIASES.get(lang.lower(), lang.lower())


_instances: dict[tuple, Transcriber] = {}
_instances_lock = threading.Lock()


def get_transcriber(backend: Optional[str] = None, model: Optional[str] = None,
                    device: Optional[str] = None, language: Optional[str] = None) -> Transcriber:
    """Return a cached Transcriber (one model load per configuration per process)."""
    backend = backend or config.ASR_BACKEND
    model = model or config.ASR_MODEL
    device = device or config.ASR_DEVICE
    language = language or config.ASR_LANGUAGE
    compute = config.ASR_COMPUTE_TYPE if device == "cpu" else ("float16" if config.ASR_COMPUTE_TYPE == "int8" else config.ASR_COMPUTE_TYPE)
    key = (backend, model, device, compute, language)
    with _instances_lock:
        if key not in _instances:
            _instances[key] = Transcriber(backend, model, device, compute, language, config.ASR_BEAM_SIZE)
        return _instances[key]


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-zऀ-෿]+", "", text.lower())


def align_sentences(sentences: list[str], words: list[Word]) -> list[tuple[Optional[float], Optional[float]]]:
    """Map each sentence to (start, end) seconds using word timestamps.

    Walks the word list once, consuming words until their normalised characters
    cover the normalised sentence. Robust to punctuation/spacing differences
    introduced by sentence segmentation.
    """
    out: list[tuple[Optional[float], Optional[float]]] = []
    wi = 0
    for sent in sentences:
        target = len(_norm(sent))
        if target == 0 or wi >= len(words):
            out.append((None, None))
            continue
        start = words[wi].start
        end = words[wi].end
        acc = 0
        while wi < len(words) and acc < target:
            acc += len(_norm(words[wi].text))
            end = words[wi].end
            wi += 1
        out.append((round(start, 2), round(end, 2)))
    return out


def words_from_segments(segments: list[dict]) -> list[Word]:
    """Rebuild Word objects from a saved `.segments.json` payload.

    Falls back to one pseudo-word per segment when word timings were not saved.
    """
    words: list[Word] = []
    for s in segments:
        if s.get("words"):
            words.extend(Word(float(a), float(b), t) for a, b, t in s["words"])
        else:
            words.append(Word(float(s["start"]), float(s["end"]), s.get("text", "")))
    return words
