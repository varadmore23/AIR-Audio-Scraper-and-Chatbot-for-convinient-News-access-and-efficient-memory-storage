"""
train_genre_classifier.py - Fine-tune DistilBERT for AIR news genre classification.

Trains an AutoModelForSequenceClassification on genre_training_data.csv rows
where 'true_genre' is filled in. Saves the best checkpoint (by val accuracy)
to models/air-genre-classifier/.

Usage:
    python train_genre_classifier.py
    python train_genre_classifier.py --csv genre_training_data.csv --epochs 5 --batch-size 16
    python train_genre_classifier.py --model bert-base-uncased --lr 2e-5

Requires: scikit-learn, transformers, torch (all in requirements.txt)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_genre_classifier")

DEFAULT_CSV = "genre_training_data.csv"
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_OUT_DIR = os.path.join("models", "air-genre-classifier")
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 3e-5
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_MAX_LEN = 128


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labeled_rows(csv_path: str) -> tuple[list[str], list[str]]:
    """Read CSV and return (texts, labels) for rows where true_genre is set."""
    if not os.path.isfile(csv_path):
        logger.error("Training CSV not found: %s", csv_path)
        logger.error("Run: python build_genre_training_data.py --augment")
        sys.exit(1)

    texts, labels = [], []
    skipped = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_genre = row.get("true_genre", "").strip()
            text = row.get("text", "").strip()
            if not true_genre or not text:
                skipped += 1
                continue
            texts.append(text)
            labels.append(true_genre)

    logger.info("Loaded %d labeled rows (%d skipped — empty true_genre)", len(texts), skipped)
    if not texts:
        logger.error("No labeled rows found. Fill in 'true_genre' column first.")
        sys.exit(1)
    return texts, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    csv_path: str = DEFAULT_CSV,
    model_name: str = DEFAULT_MODEL_NAME,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    val_split: float = DEFAULT_VAL_SPLIT,
    max_len: int = DEFAULT_MAX_LEN,
    seed: int = 42,
) -> int:
    """Fine-tune and save the genre classifier. Returns 0 on success."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    from nlp_pipeline import GENRE_LABELS

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    texts, raw_labels = load_labeled_rows(csv_path)

    # Validate labels
    invalid = [l for l in raw_labels if l not in GENRE_LABELS]
    if invalid:
        logger.error("Unknown labels found (not in GENRE_LABELS): %s", set(invalid))
        logger.error("Valid labels: %s", GENRE_LABELS)
        return 1

    label2id = {g: i for i, g in enumerate(GENRE_LABELS)}
    id2label = {i: g for g, i in label2id.items()}
    y = [label2id[l] for l in raw_labels]

    # -----------------------------------------------------------------------
    # Train/val split (stratified)
    # -----------------------------------------------------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(texts, y))

    train_texts = [texts[i] for i in train_idx]
    train_labels = [y[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [y[i] for i in val_idx]

    logger.info("Train: %d | Val: %d | Classes: %d", len(train_texts), len(val_texts), len(GENRE_LABELS))

    # Label distribution
    from collections import Counter
    dist = Counter(raw_labels)
    logger.info("Label distribution: %s", dict(dist.most_common()))

    # -----------------------------------------------------------------------
    # Class weights for imbalance
    # -----------------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(GENRE_LABELS)),
        y=np.array(train_labels),
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Tokenizer + model
    # -----------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model: %s (%d classes)", model_name, len(GENRE_LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(GENRE_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    model = model.to(device)
    weight_tensor = weight_tensor.to(device)

    # -----------------------------------------------------------------------
    # Dataset helpers
    # -----------------------------------------------------------------------
    def tokenize(batch_texts: list[str]) -> dict:
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    def iter_batches(tx, ty, bsz, shuffle=True):
        indices = list(range(len(tx)))
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(indices)
        for start in range(0, len(indices), bsz):
            batch_idx = indices[start: start + bsz]
            yield [tx[i] for i in batch_idx], [ty[i] for i in batch_idx]

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    best_val_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_texts, batch_labels in iter_batches(train_texts, train_labels, batch_size):
            enc = tokenize(batch_texts)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels_t = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels_t).sum().item()
            total_samples += len(batch_labels)

        train_acc = total_correct / total_samples * 100
        avg_loss = total_loss / total_samples

        # Validation
        model.eval()
        val_correct = 0
        val_preds_all: list[int] = []
        val_labels_all: list[int] = []

        with torch.no_grad():
            for batch_texts, batch_labels in iter_batches(val_texts, val_labels, batch_size * 2, shuffle=False):
                enc = tokenize(batch_texts)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                labels_t = torch.tensor(batch_labels, dtype=torch.long).to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels_t).sum().item()
                val_preds_all.extend(preds.cpu().tolist())
                val_labels_all.extend(batch_labels)

        val_acc = val_correct / len(val_labels) * 100
        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d | loss=%.4f | train_acc=%.1f%% | val_acc=%.1f%% | %.0fs elapsed",
            epoch, epochs, avg_loss, train_acc, val_acc, elapsed,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            os.makedirs(out_dir, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            # Save label mapping as standalone JSON (for lazy-load in nlp_pipeline.py)
            with open(os.path.join(out_dir, "label_map.json"), "w") as f:
                json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)
            logger.info("  --> New best val_acc=%.1f%% — checkpoint saved to %s", best_val_acc, out_dir)

    # -----------------------------------------------------------------------
    # Final evaluation report (per-class F1, matching eval_genre.py style)
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  FINE-TUNED CLASSIFIER TRAINING REPORT")
    print("=" * 70)
    print(f"  Best checkpoint: epoch {best_epoch} / {epochs}")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to:  {out_dir}")
    print()

    # Per-class metrics using final val predictions
    tp: dict[int, int] = {i: 0 for i in range(len(GENRE_LABELS))}
    fp: dict[int, int] = {i: 0 for i in range(len(GENRE_LABELS))}
    fn: dict[int, int] = {i: 0 for i in range(len(GENRE_LABELS))}

    for p, t in zip(val_preds_all, val_labels_all):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    print(f"  {'Genre':<28} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}")
    print("  " + "-" * 64)
    for i, genre in enumerate(GENRE_LABELS):
        support = tp[i] + fn[i]
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {genre:<28} {precision:>9.3f} {recall:>9.3f} {f1:>9.3f} {support:>9d}")

    overall_correct = sum(1 for p, t in zip(val_preds_all, val_labels_all) if p == t)
    overall_acc = overall_correct / len(val_labels_all) * 100 if val_labels_all else 0.0
    print("  " + "-" * 64)
    print(f"  {'OVERALL':<28} {'':>9} {'':>9} {'':>9} {len(val_labels_all):>9d}")
    print(f"  Accuracy: {overall_acc:.2f}%")
    print("=" * 70)
    print()
    print("  To use the fine-tuned classifier:")
    print("    python ingest_pipeline.py --classifier finetuned")
    print("    python nlp_pipeline.py --transcript <file> --classifier finetuned")
    print("  To compare vs zero-shot:")
    print("    python eval_genre.py --compare")
    print("=" * 70)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for AIR news genre classification."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"Training CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"Base HF model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Checkpoint output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                        help=f"Validation fraction (default: {DEFAULT_VAL_SPLIT})")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                        help=f"Max token length (default: {DEFAULT_MAX_LEN})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return train(
        csv_path=args.csv,
        model_name=args.model,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        max_len=args.max_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
