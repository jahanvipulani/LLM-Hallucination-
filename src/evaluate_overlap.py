
import json
import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_FILE = DATA_DIR / "source_info.jsonl"

# Automatically locate the response dataset.
POSSIBLE_RESPONSE_FILES = [
    DATA_DIR / "response.jsonl",
    DATA_DIR / "response(1).jsonl",
]

RESPONSE_FILE = next(
    (
        path
        for path in POSSIBLE_RESPONSE_FILES
        if path.exists()
    ),
    None,
)

# Threshold selected using validation data.
THRESHOLD_FILE = (
    PROJECT_ROOT / "overlap_threshold_validation.json"
)

# Output files.
OUTPUT_FILE = PROJECT_ROOT / "overlap_test_results.csv"
METRICS_FILE = PROJECT_ROOT / "overlap_test_metrics.json"


# ============================================================
# 2. READ JSONL FILES
# ============================================================

def read_jsonl(file_path):
    """Read a JSON Lines file and return its records."""

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, "
                    f"line {line_number}: {error}"
                ) from error

    return records


# ============================================================
# 3. CONVERT SOURCE DATA INTO TEXT
# ============================================================

def source_to_text(value):
    """Normalize source and response values consistently with validation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [source_to_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, (list, tuple)):
        parts = [source_to_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


# ============================================================
# 4. TOKENIZE TEXT
# ============================================================

def tokenize(text):
    """Convert text into lowercase word tokens."""

    tokens = []

    for word in text.lower().split():
        word = word.strip(".,!?;:\"'()[]{}")

        if word:
            tokens.append(word)

    return tokens


# ============================================================
# 5. CALCULATE LEXICAL OVERLAP
# ============================================================

def calculate_overlap(source, response):
    """
    Calculate the proportion of response tokens that occur
    in the source.

    Overlap = matching response tokens / total response tokens
    Unsupported score = 1 - overlap
    """

    source_tokens = set(tokenize(source))
    response_tokens = tokenize(response)

    if not response_tokens:
        return 0.0

    matching_tokens = sum(
        1
        for token in response_tokens
        if token in source_tokens
    )

    return matching_tokens / len(response_tokens)


# ============================================================
# 6. MAIN EVALUATION
# ============================================================

def main():
    print("Lexical overlap evaluation started.", flush=True)

    # Verify that all required files exist.
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    if RESPONSE_FILE is None:
        raise FileNotFoundError(
            "Neither response.jsonl nor response(1).jsonl "
            f"was found in {DATA_DIR}"
        )

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Validation threshold file not found: "
            f"{THRESHOLD_FILE}\n"
            "Run tune_overlap_validation.py first."
        )

    # Load the validation-selected threshold.
    with open(
        THRESHOLD_FILE, "r", encoding="utf-8"
    ) as file:
        threshold_info = json.load(file)

    threshold = float(threshold_info["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid threshold: {threshold}"
        )

    print(f"Selected threshold: {threshold:.2f}")
    print("Loading source records...", flush=True)

    source_records = read_jsonl(SOURCE_FILE)

    print("Loading response records...", flush=True)

    response_records = read_jsonl(RESPONSE_FILE)

    # --------------------------------------------------------
    # Create a source lookup using source_id.
    # Use both source and source_info consistently.
    # --------------------------------------------------------

    source_lookup = {}

    for record in source_records:
        source_id = record.get("source_id")

        source_text = " ".join(
            text
            for text in [
                source_to_text(record.get("source", "")),
                source_to_text(record.get("source_info", "")),
            ]
            if text.strip()
        )

        source_lookup[str(source_id)] = source_text

    # --------------------------------------------------------
    # Select only the held-out test records.
    # --------------------------------------------------------

    test_records = [
        record
        for record in response_records
        if record.get("split") == "test"
    ]

    if not test_records:
        raise ValueError(
            "No test records found. Check the split field "
            "in the response dataset."
        )

    results = []

    print(
        f"Evaluating {len(test_records)} test responses...",
        flush=True,
    )

    for record in test_records:
        source_id = str(record.get("source_id"))

        source_text = source_lookup.get(source_id, "")

        response_text = source_to_text(
            record.get("response", "")
        )

        # RAGTruth response-level label:
        # a non-empty labels array indicates an annotated
        # hallucination span.
        labels = record.get("labels", [])

        actual_label = int(
            isinstance(labels, list)
            and len(labels) > 0
        )

        # Calculate lexical overlap and unsupported score.
        overlap_score = calculate_overlap(
            source_text,
            response_text,
        )

        unsupported_score = 1.0 - overlap_score

        # Predict positive if unsupported score meets threshold.
        predicted_label = int(
            unsupported_score >= threshold
        )

        results.append({
            "id": record.get("id", ""),
            "source_id": source_id,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "overlap_score": overlap_score,
            "unsupported_score": unsupported_score,
            "source_missing": int(
                not source_text.strip()
            ),
        })

    # ========================================================
    # 7. CALCULATE EVALUATION METRICS
    # ========================================================

    y_true = [
        row["actual_label"] for row in results
    ]

    y_pred = [
        row["predicted_label"] for row in results
    ]

    y_score = [
        row["unsupported_score"] for row in results
    ]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    # AUROC uses continuous scores, not thresholded predictions.
    auroc = (
        roc_auc_score(y_true, y_score)
        if len(set(y_true)) == 2
        else None
    )

    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count

    missing_source_count = sum(
        row["source_missing"] for row in results
    )

    # ========================================================
    # 8. SAVE TEST METRICS
    # ========================================================

    metrics = {
        "method": "lexical_overlap",
        "dataset": "RAGTruth",
        "split": "test",
        "test_records": len(results),
        "positive_labels": positive_count,
        "negative_labels": negative_count,
        "threshold": threshold,
        "threshold_status": (
            "selected using validation F1"
        ),
        "validation_f1": threshold_info.get(
            "validation_f1"
        ),
        "source_text_fields": [
            "source",
            "source_info",
        ],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix.tolist(),
        "auroc": auroc,
        "missing_source_records": missing_source_count,
    }

    # Save each test response and its scores.
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # Save aggregate metrics.
    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # ========================================================
    # 9. DISPLAY RESULTS
    # ========================================================

    print("\nLEXICAL OVERLAP TEST RESULTS")
    print("=" * 45)

    print(f"Test records: {len(results)}")
    print(f"Positive labels: {positive_count}")
    print(f"Negative labels: {negative_count}")
    print(f"Threshold: {threshold:.2f} (validation-selected)")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    if auroc is not None:
        print(f"AUROC:     {auroc:.4f}")
    else:
        print("AUROC:     Not applicable")

    print("\nConfusion matrix:")
    print(matrix)

    print(
        "\nMissing source records:",
        missing_source_count,
    )

    print(f"\nRecord results saved to:\n{OUTPUT_FILE}")
    print(f"Metrics saved to:\n{METRICS_FILE}")


if __name__ == "__main__":
    main()
