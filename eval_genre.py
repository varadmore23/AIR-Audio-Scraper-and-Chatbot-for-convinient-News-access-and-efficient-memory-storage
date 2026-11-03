"""
eval_genre.py - Zero-shot Genre Classification Evaluation Harness

Evaluates zero-shot genre predictions against ground-truth human labels
without external dependencies (no sklearn, no pandas).

Usage:
  1. Generate evaluation sample:
     python eval_genre.py --sample 10

  2. Fill in the 'true_genre' column in eval_labels.csv

  3. Compute accuracy and confusion matrix:
     python eval_genre.py --evaluate

  4. Compare zero-shot vs fine-tuned side by side:
     python eval_genre.py --compare
"""

import argparse
import csv
import os
import sqlite3
import sys

SQLITE_DB_PATH = "news_store.db"
EVAL_CSV_PATH = "eval_labels.csv"
TRAINING_CSV_PATH = "genre_training_data.csv"


def _get_training_texts(training_csv: str = TRAINING_CSV_PATH) -> set[str]:
    """Return normalized texts present in genre_training_data.csv to prevent data leakage."""
    if not os.path.isfile(training_csv):
        return set()
    try:
        with open(training_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {r["text"].strip().lower() for r in reader if r.get("text")}
    except Exception:
        return set()


def sample_segments(
    n: int,
    db_path: str = SQLITE_DB_PATH,
    out_path: str = EVAL_CSV_PATH,
    training_csv: str = TRAINING_CSV_PATH,
) -> int:
    """Sample N random segments from SQLite and save to eval_labels.csv,
    excluding any segments already used as training data to prevent leakage."""
    if not os.path.isfile(db_path):
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        return 1

    training_texts = _get_training_texts(training_csv)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT faiss_id, text, genre FROM segments ORDER BY RANDOM()"
    ).fetchall()
    conn.close()

    if not all_rows:
        print("Error: No segments found in database to sample.", file=sys.stderr)
        return 1

    # Exclude any segments already used in training data
    candidate_rows = [
        r for r in all_rows if r["text"].strip().lower() not in training_texts
    ]
    if training_texts:
        excluded = len(all_rows) - len(candidate_rows)
        if excluded > 0:
            print(f"Excluded {excluded} segment(s) already present in training data ({training_csv}).")

    if not candidate_rows:
        print(
            f"Error: All {len(all_rows)} segments in database are already in training data ({training_csv}). "
            "No held-out evaluation samples available. Ingest more bulletins first.",
            file=sys.stderr,
        )
        return 1

    rows = candidate_rows[:n]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["faiss_id", "text", "predicted_genre", "true_genre"])
        for r in rows:
            writer.writerow([r["faiss_id"], r["text"], r["genre"], ""])

    print(f"Sampled {len(rows)} held-out segments to {out_path}.")
    print("Fill in the 'true_genre' column and run: python eval_genre.py --evaluate")
    return 0


def _build_matrix(
    labeled_rows: list[dict],
    pred_key: str,
    all_genres: list[str],
) -> tuple[dict, int, int]:
    """Build a confusion matrix dict and return (matrix, correct, total)."""
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in all_genres} for g in all_genres}
    correct = 0
    for r in labeled_rows:
        t = r["true_genre"].strip()
        p = r.get(pred_key, "").strip()
        if t in matrix:
            if p in matrix[t]:
                matrix[t][p] += 1
            if t.lower() == p.lower():
                correct += 1
    return matrix, correct, len(labeled_rows)


def _print_matrix(title: str, matrix: dict, all_genres: list[str], correct: int, total: int) -> None:
    """Print a labelled confusion matrix block consistent with evaluate_labels() style."""
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"Total labeled samples: {total}")
    print(f"Correct predictions:   {correct}")
    print(f"Accuracy:              {accuracy:.2f}%\n")

    col_width = max(max(len(g) for g in all_genres) + 2, 12)
    label_title = "True \ Pred"
    header = f"{label_title:<{col_width}}" + "".join(f"{g[:col_width-1]:<{col_width}}" for g in all_genres)
    print("Confusion Matrix:")
    print(header)
    print("-" * len(header))
    for t in all_genres:
        row_str = f"{t:<{col_width}}"
        for p in all_genres:
            cnt = matrix[t][p]
            row_str += f"{cnt:<{col_width}}"
        print(row_str)
    print("=" * 70)


def evaluate_labels(csv_path: str = EVAL_CSV_PATH) -> int:
    """Read labeled evaluation CSV and print accuracy and confusion matrix."""
    if not os.path.isfile(csv_path):
        print(f"Error: Evaluation file not found: {csv_path}", file=sys.stderr)
        print("Run 'python eval_genre.py --sample N' first.", file=sys.stderr)
        return 1

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labeled_rows = [r for r in rows if r.get("true_genre", "").strip()]
    if not labeled_rows:
        print(f"No labeled rows found in {csv_path} (true_genre is empty).", file=sys.stderr)
        print("Please fill in ground truth values in the 'true_genre' column first.", file=sys.stderr)
        return 1

    # Unique genre labels
    all_genres = sorted(list(set(
        [r["true_genre"].strip() for r in labeled_rows] +
        [r["predicted_genre"].strip() for r in labeled_rows]
    )))

    matrix, correct, total = _build_matrix(labeled_rows, "predicted_genre", all_genres)
    _print_matrix("GENRE CLASSIFICATION EVALUATION REPORT", matrix, all_genres, correct, total)
    return 0


def compare_classifiers(csv_path: str = EVAL_CSV_PATH) -> int:
    """Run zero-shot and fine-tuned classifiers over eval_labels.csv rows and
    print two side-by-side confusion matrices plus the accuracy delta.

    If the fine-tuned model is absent, only zero-shot results are shown.
    """
    if not os.path.isfile(csv_path):
        print(f"Error: Evaluation file not found: {csv_path}", file=sys.stderr)
        print("Run 'python eval_genre.py --sample N' first.", file=sys.stderr)
        return 1

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labeled_rows = [r for r in rows if r.get("true_genre", "").strip()]
    if not labeled_rows:
        print(f"No labeled rows found in {csv_path}.", file=sys.stderr)
        return 1

    training_texts = _get_training_texts()
    overlap = [r for r in labeled_rows if r["text"].strip().lower() in training_texts]
    if overlap:
        print(
            f"[WARNING] {len(overlap)} of {len(labeled_rows)} evaluation rows in '{csv_path}' "
            "are present in training data (genre_training_data.csv).\n"
            "          These evaluate training performance rather than held-out generalization.\n"
        )

    texts = [r["text"] for r in labeled_rows]

    # --- Zero-shot fast predictions ---
    print("Running zero-shot (fast) classifier on evaluation set...")
    from nlp_pipeline import classify_genres_batch, classify_genres_batch_finetuned, FINETUNED_MODEL_PATH
    zs_preds = classify_genres_batch(texts, fast=True)
    for i, row in enumerate(labeled_rows):
        row["_zs_pred"] = zs_preds[i][0]

    # --- Fine-tuned predictions (if available) ---
    has_finetuned = os.path.isdir(FINETUNED_MODEL_PATH)
    if has_finetuned:
        print("Running fine-tuned classifier on evaluation set...")
        ft_preds = classify_genres_batch_finetuned(texts)
        for i, row in enumerate(labeled_rows):
            row["_ft_pred"] = ft_preds[i][0]
    else:
        print(
            f"[NOTE] Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'. "
            "Showing zero-shot results only.\n"
            "Run 'python train_genre_classifier.py' to enable --compare."
        )

    # Build genre label set from ground truth + both prediction sets
    all_genres = sorted(set(
        r["true_genre"].strip() for r in labeled_rows
    ) | set(
        r["_zs_pred"] for r in labeled_rows
    ) | (set(r["_ft_pred"] for r in labeled_rows) if has_finetuned else set()))

    # --- Print zero-shot matrix ---
    print()
    zs_matrix, zs_correct, total = _build_matrix(labeled_rows, "_zs_pred", all_genres)
    _print_matrix("ZERO-SHOT (fast) CLASSIFIER", zs_matrix, all_genres, zs_correct, total)

    if has_finetuned:
        print()
        ft_matrix, ft_correct, _ = _build_matrix(labeled_rows, "_ft_pred", all_genres)
        _print_matrix("FINE-TUNED CLASSIFIER", ft_matrix, all_genres, ft_correct, total)

        # --- Accuracy delta ---
        zs_acc = zs_correct / total * 100.0
        ft_acc = ft_correct / total * 100.0
        delta = ft_acc - zs_acc
        sign = "+" if delta >= 0 else ""
        print()
        print("=" * 70)
        print("  ACCURACY COMPARISON")
        print("=" * 70)
        print(f"  Zero-shot accuracy  : {zs_acc:.2f}%")
        print(f"  Fine-tuned accuracy : {ft_acc:.2f}%")
        print(f"  Delta               : {sign}{delta:.2f} pp")
        verdict = "Fine-tuned BETTER" if delta > 0 else ("Tie" if delta == 0 else "Zero-shot BETTER")
        print(f"  Verdict             : {verdict}")
        print("=" * 70)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Genre evaluation harness for AIR News Intelligence.")
    parser.add_argument("--sample", type=int, metavar="N", help="Sample N random segments into eval_labels.csv")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate accuracy from eval_labels.csv")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare zero-shot vs fine-tuned classifier on eval_labels.csv (side-by-side matrices + delta)",
    )
    args = parser.parse_args()

    if args.sample:
        return sample_segments(args.sample)
    elif args.evaluate:
        return evaluate_labels()
    elif args.compare:
        return compare_classifiers()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
