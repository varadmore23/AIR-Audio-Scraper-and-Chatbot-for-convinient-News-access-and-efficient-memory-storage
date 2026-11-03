"""
nlp_pipeline.py - NLP Pipeline for AIR Broadcast News Intelligence System (Phase 1)

Takes a transcript .txt file and produces classified, deduplicated, embedded
segments ready for FAISS indexing.

CLI usage:
  python nlp_pipeline.py --transcript output/text/2026-05-01/file.txt --date 2026-05-01 --hour 0800
  python nlp_pipeline.py --transcript ... --fast --mmr-threshold 0.90
  python nlp_pipeline.py --transcript ... --classifier finetuned

Classifier backends:
  zero-shot  : facebook/bart-large-mnli (default, highest quality zero-shot)
  fast       : cross-encoder/nli-MiniLM2-L6-H768 (lighter, --fast flag)
  finetuned  : models/air-genre-classifier/ (DistilBERT fine-tuned on AIR corpus);
               falls back to 'fast' with a WARNING if checkpoint not present.
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

import re

import nltk
import numpy as np

import config

logger = logging.getLogger("nlp")

_PUNKT_OK = True
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer data...")
    try:
        _PUNKT_OK = bool(nltk.download("punkt_tab", quiet=True))
    except Exception:
        _PUNKT_OK = False
    if not _PUNKT_OK:
        logger.warning("NLTK punkt_tab unavailable; using regex sentence splitter")

GENRE_LABELS = [
    "Politics", "Economy", "Science & Technology", "Sports",
    "International", "Defence & Security", "Health",
    "Environment", "Society & Culture", "Other",
]

DEFAULT_CLASSIFIER_MODEL = "facebook/bart-large-mnli"
FAST_CLASSIFIER_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
EMBEDDING_MODEL = config.EMBEDDING_MODEL
FINETUNED_MODEL_PATH = os.path.join("models", "air-genre-classifier")
DEFAULT_MMR_THRESHOLD = config.MMR_THRESHOLD

_classifier_pipeline = None
_embedding_model = None
_finetuned_classifier: Optional[tuple] = None  # (model, tokenizer, id2label) or None


def segment_sentences(text: str) -> list[str]:
    """Split raw transcript text into individual sentences using NLTK."""
    text = " ".join(text.lstrip('\ufeff').split())
    if not text:
        return []
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?\u0964])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _get_classifier(fast: bool = False):
    """Lazily load and cache the zero-shot classification pipeline."""
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline
    from transformers import pipeline as hf_pipeline
    model_name = FAST_CLASSIFIER_MODEL if fast else DEFAULT_CLASSIFIER_MODEL
    logger.info("Loading zero-shot classifier: %s", model_name)
    _classifier_pipeline = hf_pipeline("zero-shot-classification", model=model_name, device=-1)
    return _classifier_pipeline


def classify_genre(sentence: str, fast: bool = False) -> tuple[str, float]:
    """Classify a single sentence into one of the predefined genre labels."""
    classifier = _get_classifier(fast=fast)
    result = classifier(sentence, candidate_labels=GENRE_LABELS)
    return result["labels"][0], float(result["scores"][0])


def classify_genres_batch(sentences: list[str], fast: bool = False) -> list[tuple[str, float]]:
    """Batch-classify sentences for better throughput."""
    if not sentences:
        return []
    classifier = _get_classifier(fast=fast)
    results = classifier(sentences, candidate_labels=GENRE_LABELS, batch_size=8)
    if isinstance(results, dict):
        results = [results]
    return [(r["labels"][0], float(r["scores"][0])) for r in results]


# ---------------------------------------------------------------------------
# Fine-tuned classifier backend
# ---------------------------------------------------------------------------

def _get_finetuned_classifier() -> Optional[tuple]:
    """Lazily load the fine-tuned genre classifier from FINETUNED_MODEL_PATH.

    Returns (model, tokenizer, id2label) if the checkpoint exists, or None
    if the checkpoint is missing (caller must fallback to zero-shot).
    This mirrors the _probe_gemini() graceful-degradation pattern in rag_engine.py.
    """
    global _finetuned_classifier
    if _finetuned_classifier is not None:
        return _finetuned_classifier

    if not os.path.isdir(FINETUNED_MODEL_PATH):
        logger.warning(
            "Fine-tuned model not found at '%s'. "
            "Run 'python train_genre_classifier.py' to create it. "
            "Falling back to zero-shot fast classifier.",
            FINETUNED_MODEL_PATH,
        )
        return None

    try:
        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Loading fine-tuned classifier from %s", FINETUNED_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
        model.eval()

        # Load id2label from saved label_map.json (authoritative) or model config
        label_map_path = os.path.join(FINETUNED_MODEL_PATH, "label_map.json")
        if os.path.isfile(label_map_path):
            with open(label_map_path, "r") as f:
                data = json.load(f)
            id2label = {int(k): v for k, v in data["id2label"].items()}
        else:
            id2label = {int(k): v for k, v in model.config.id2label.items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        _finetuned_classifier = (model, tokenizer, id2label, device)
        logger.info("Fine-tuned classifier loaded (%d classes)", len(id2label))
        return _finetuned_classifier

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load fine-tuned classifier from '%s': %s. "
            "Falling back to zero-shot fast classifier.",
            FINETUNED_MODEL_PATH,
            exc,
        )
        return None


def classify_genres_batch_finetuned(
    sentences: list[str],
    max_len: int = 128,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Classify sentences using the fine-tuned DistilBERT model.

    Returns the exact same ``list[tuple[str, float]]`` shape as
    ``classify_genres_batch()`` so all downstream callers are unchanged.

    If the fine-tuned model is unavailable, transparently falls back to the
    zero-shot fast model and logs a WARNING — no exception is raised.
    """
    if not sentences:
        return []

    classifier = _get_finetuned_classifier()
    if classifier is None:
        # Graceful degradation: fall back to zero-shot fast
        return classify_genres_batch(sentences, fast=True)

    import torch
    model, tokenizer, id2label, device = classifier

    results: list[tuple[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start: start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probs.max(dim=-1)
            for pid, pconf in zip(top_ids.cpu().tolist(), top_probs.cpu().tolist()):
                label = id2label.get(pid, GENRE_LABELS[-1])  # fallback to "Other"
                results.append((label, float(pconf)))

    return results


def _get_embedding_model():
    """Lazily load and cache the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_sentences(sentences: list[str]) -> np.ndarray:
    """Generate embeddings for a list of sentences. Returns shape (N, 384)."""
    model = _get_embedding_model()
    return model.encode(sentences, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def mmr_novelty_filter_detailed(
    sentences: list[str],
    embeddings: np.ndarray,
    existing_embeddings: Optional[np.ndarray] = None,
    existing_texts: Optional[list[str]] = None,
    threshold: float = DEFAULT_MMR_THRESHOLD,
) -> tuple[list[int], list[dict]]:
    """MMR novelty gate: streaming cosine-similarity deduplication.

    Returns:
        (novel_indices, rejections)
        where rejections is a list of dicts:
        {"rejected_text": str, "matched_text": str, "cosine_score": float, "matched_source_idx": int}
    """
    def _unit(m: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    normed = _unit(np.asarray(embeddings, dtype=np.float32))

    # Similarity of every new sentence to the stored corpus, computed in one matmul.
    if existing_embeddings is not None and len(existing_embeddings) > 0:
        existing = _unit(np.asarray(existing_embeddings, dtype=np.float32))
        sims_existing = normed @ existing.T
        if existing_texts and len(existing_texts) == len(existing):
            existing_labels = list(existing_texts)
        else:
            existing_labels = [f"Existing segment #{idx+1}" for idx in range(len(existing))]
    else:
        sims_existing = None
        existing_labels = []

    novel_indices: list[int] = []
    rejections: list[dict] = []

    for i in range(len(normed)):
        best_sim, best_text, best_idx = -1.0, "", -1
        if sims_existing is not None and sims_existing.shape[1]:
            j = int(np.argmax(sims_existing[i]))
            best_sim, best_text, best_idx = float(sims_existing[i, j]), existing_labels[j], j
        if novel_indices:  # also compare with sentences accepted earlier in this batch
            batch_sims = normed[novel_indices] @ normed[i]
            k = int(np.argmax(batch_sims))
            if float(batch_sims[k]) > best_sim:
                best_sim = float(batch_sims[k])
                best_text = sentences[novel_indices[k]]
                best_idx = len(existing_labels) + k

        if best_sim >= threshold:
            logger.debug("DUPLICATE (sim=%.3f): %s", best_sim, sentences[i][:80])
            rejections.append({
                "rejected_text": sentences[i],
                "matched_text": best_text,
                "cosine_score": round(best_sim, 4),
                "matched_source_idx": best_idx,
            })
        else:
            novel_indices.append(i)

    return novel_indices, rejections


def mmr_novelty_filter(
    sentences: list[str],
    embeddings: np.ndarray,
    existing_embeddings: Optional[np.ndarray] = None,
    threshold: float = DEFAULT_MMR_THRESHOLD,
) -> list[int]:
    """MMR novelty gate: streaming cosine-similarity deduplication.

    Returns indices of novel (non-duplicate) sentences.
    """
    novel_indices, _ = mmr_novelty_filter_detailed(
        sentences, embeddings, existing_embeddings, threshold=threshold
    )
    return novel_indices


def process_transcript(
    text: str,
    existing_embeddings: Optional[np.ndarray] = None,
    fast: bool = False,
    mmr_threshold: float = DEFAULT_MMR_THRESHOLD,
    detailed: bool = False,
    existing_texts: Optional[list[str]] = None,
    classifier: str = "zero-shot",
    sentence_timestamps: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
):
    """Run the full NLP pipeline on a transcript string.

    Args:
        text: Raw transcript text.
        existing_embeddings: Embeddings already in FAISS for MMR dedup.
        fast: If True, equivalent to classifier='fast' (backward compat).
        mmr_threshold: Cosine similarity threshold for novelty gate.
        detailed: If True, returns (segments, rejections).
        existing_texts: Corresponding texts for existing_embeddings.
        classifier: Backend selector — 'zero-shot' | 'fast' | 'finetuned'.
                    If 'finetuned' and model missing, falls back to 'fast'.
        sentence_timestamps: Optional list of (audio_start, audio_end) tuples matching
                             segmented sentences in order.

    Returns:
        segments list, or (segments, rejections) when detailed=True.
    """
    # Backward compat: --fast flag maps to classifier='fast'
    if fast and classifier == "zero-shot":
        classifier = "fast"

    sentences = segment_sentences(text)
    logger.info("Segmented transcript into %d sentences", len(sentences))
    if not sentences:
        return ([], []) if detailed else []

    embeddings = embed_sentences(sentences)
    logger.info("Generated embeddings: shape %s", embeddings.shape)

    novel_indices, rejections = mmr_novelty_filter_detailed(
        sentences,
        embeddings,
        existing_embeddings=existing_embeddings,
        existing_texts=existing_texts,
        threshold=mmr_threshold,
    )
    logger.info("MMR filter: %d novel / %d total (threshold=%.2f)", len(novel_indices), len(sentences), mmr_threshold)
    if not novel_indices:
        return ([], rejections) if detailed else []

    novel_sentences = [sentences[i] for i in novel_indices]
    novel_embeddings = embeddings[novel_indices]
    logger.info("Classifying %d novel sentences (backend=%s)...", len(novel_sentences), classifier)

    if classifier == "finetuned":
        genre_results = classify_genres_batch_finetuned(novel_sentences)
    elif classifier == "fast":
        genre_results = classify_genres_batch(novel_sentences, fast=True)
    else:
        # "zero-shot" default
        genre_results = classify_genres_batch(novel_sentences, fast=False)

    segments = []
    for idx, (genre, confidence) in enumerate(genre_results):
        orig_i = novel_indices[idx]
        a_start, a_end = None, None
        if sentence_timestamps and orig_i < len(sentence_timestamps):
            a_start, a_end = sentence_timestamps[orig_i]

        segments.append({
            "text": novel_sentences[idx],
            "genre": genre,
            "genre_confidence": confidence,
            "embedding": novel_embeddings[idx],
            "audio_start": a_start,
            "audio_end": a_end,
        })
    return (segments, rejections) if detailed else segments


def main() -> int:
    """CLI entry point for standalone pipeline execution."""
    parser = argparse.ArgumentParser(description="NLP pipeline for AIR transcripts.")
    parser.add_argument("--transcript", required=True, help="Path to transcript .txt file")
    parser.add_argument("--date", required=True, help="Broadcast date YYYY-MM-DD")
    parser.add_argument("--hour", required=True, type=int, help="Broadcast hour e.g. 0800")
    parser.add_argument("--fast", action="store_true",
                        help="Use lighter classifier model (shorthand for --classifier fast)")
    parser.add_argument(
        "--classifier",
        choices=["zero-shot", "fast", "finetuned"],
        default="zero-shot",
        help="Classifier backend: zero-shot (default), fast, or finetuned",
    )
    parser.add_argument("--mmr-threshold", type=float, default=DEFAULT_MMR_THRESHOLD,
                        help=f"Cosine sim threshold for dedup (default {DEFAULT_MMR_THRESHOLD})")
    args = parser.parse_args()

    # --fast overrides --classifier for backward compat
    classifier = args.classifier
    if args.fast:
        classifier = "fast"

    if not os.path.isfile(args.transcript):
        logger.error("Transcript file not found: %s", args.transcript)
        return 1

    with open(args.transcript, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        logger.warning("Transcript is empty: %s", args.transcript)
        return 0

    existing_embeddings = None
    try:
        from vector_store import get_all_embeddings
        existing_embeddings = get_all_embeddings()
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            logger.info("Loaded %d existing embeddings for dedup", len(existing_embeddings))
    except Exception:
        logger.info("No existing FAISS index found - starting fresh")

    segments = process_transcript(
        text,
        existing_embeddings=existing_embeddings,
        fast=args.fast,
        mmr_threshold=args.mmr_threshold,
        classifier=classifier,
    )
    logger.info("Pipeline complete: %d novel segments produced", len(segments))
    for seg in segments:
        print(json.dumps({
            "text": seg["text"], "genre": seg["genre"],
            "genre_confidence": round(seg["genre_confidence"], 4),
            "embedding_shape": list(seg["embedding"].shape),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
