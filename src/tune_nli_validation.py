
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sentence_transformers import CrossEncoder


# 1. PROJECT PATHS AND SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_FILE = DATA_DIR / "source_info.jsonl"
OUTPUT_THRESHOLD = PROJECT_ROOT / "nli_threshold_validation.json"
OUTPUT_COMPARISON = PROJECT_ROOT / "nli_validation_comparison.csv"

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
BATCH_SIZE = 16
MAX_LENGTH = 512

THRESHOLDS = [
    round(x, 2) for x in np.arange(0.10, 1.00, 0.10)
]


# 2. FIND THE RESPONSE DATASET

def find_response_file():
    candidates = [
        DATA_DIR / "response.jsonl",
        DATA_DIR / "response(1).jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find response.jsonl or response(1).jsonl "
        f"inside: {DATA_DIR}"
    )


RESPONSE_FILE = find_response_file()


# 3. LOAD JSONL FILES

def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path}, "
                    f"line {line_number}: {error}"
                ) from error

    return records


# 4. NORMALIZE SOURCE TEXT

def to_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        parts = [to_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip()

    if isinstance(value, (list, tuple)):
        parts = [to_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()

    return str(value).strip()


def build_source_lookup(source_records):
    source_lookup = {}

    for record in source_records:
        source_id = record.get("source_id")

        if source_id is None:
            continue

        parts = [
            to_text(record.get("source", "")),
            to_text(record.get("source_info", "")),
        ]

        source_lookup[str(source_id)] = " ".join(
            part for part in parts if part
        ).strip()

    return source_lookup


# 5. CREATE GROUND-TRUTH LABELS

def get_actual_label(record):
    labels = record.get("labels", [])
    return int(isinstance(labels, list) and len(labels) > 0)


# 6. IDENTIFY NLI CLASS INDICES

def get_nli_class_indices(model):
    config = model.model.config

    def normalize(label):
        return str(label).lower().replace("_", "").replace("-", "")

    label2id = getattr(config, "label2id", None) or {}

    normalized = {
        normalize(label): int(index)
        for label, index in label2id.items()
    }

    names = ["contradiction", "entailment", "neutral"]

    if all(name in normalized for name in names):
        return tuple(normalized[name] for name in names)

    id2label = getattr(config, "id2label", None) or {}

    normalized_id2label = {
        int(index): normalize(label)
        for index, label in id2label.items()
    }

    found = {
        label: index
        for index, label in normalized_id2label.items()
        if label in names
    }

    if all(name in found for name in names):
        return tuple(found[name] for name in names)

    # Fallback for the selected model.
    return 0, 1, 2


# 7. CALCULATE UNSUPPORTED SCORES

def calculate_unsupported_scores(model, pairs):
    probabilities = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        apply_softmax=True,
    )

    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    contradiction_idx, entailment_idx, neutral_idx = (
        get_nli_class_indices(model)
    )

    required_idx = max(
        contradiction_idx,
        entailment_idx,
        neutral_idx,
    )

    if probabilities.shape[1] <= required_idx:
        raise ValueError(
            "The model returned fewer class probabilities than expected. "
            f"Received shape: {probabilities.shape}"
        )

    contradiction_probs = probabilities[:, contradiction_idx]
    entailment_probs = probabilities[:, entailment_idx]
    neutral_probs = probabilities[:, neutral_idx]

    unsupported_scores = contradiction_probs + neutral_probs

    return (
        contradiction_probs,
        entailment_probs,
        neutral_probs,
        unsupported_scores,
    )


# 8. MAIN VALIDATION PIPELINE

def main():
    print("NLI validation threshold tuning started.")
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Source file: {SOURCE_FILE}")
    print(f"Response file: {RESPONSE_FILE}")

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    source_records = load_jsonl(SOURCE_FILE)
    response_records = load_jsonl(RESPONSE_FILE)

    source_lookup = build_source_lookup(source_records)

    # Threshold selection uses only the training split.
    train_records = [
        record
        for record in response_records
        if str(record.get("split", "")).lower() == "train"
    ]

    if not train_records:
        raise ValueError(
            "No records with split='train' were found."
        )

    examples = []
    missing_sources = 0

    for record in train_records:
        source_id = record.get("source_id")
        source_text = source_lookup.get(str(source_id), "")
        response_text = to_text(record.get("response", ""))

        if not source_text:
            missing_sources += 1
            continue

        if not response_text:
            continue

        examples.append({
            "id": record.get("id"),
            "source_id": source_id,
            "source_text": source_text,
            "response_text": response_text,
            "actual_label": get_actual_label(record),
        })

    if len(examples) < 2:
        raise ValueError(
            "Not enough valid training examples for validation."
        )

    labels = np.array([
        example["actual_label"] for example in examples
    ])

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Training data must contain both label classes."
        )

    # Keep every response belonging to the same source together.
    indices = np.arange(len(examples))
    source_groups = np.array([
        str(example["source_id"]) for example in examples
    ])

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Select one fold as validation and the other four as development.
    dev_indices, validation_indices = next(
        splitter.split(indices, labels, groups=source_groups)
    )

    dev_source_ids = set(source_groups[dev_indices])
    validation_source_ids = set(source_groups[validation_indices])

    shared_sources = dev_source_ids & validation_source_ids

    print(
        "Shared source IDs between development and validation:",
        len(shared_sources),
    )

    if shared_sources:
        raise AssertionError(
            "Source leakage detected between development and validation."
        )

    validation_examples = [
        examples[index] for index in validation_indices
    ]

    y_validation = np.array([
        example["actual_label"]
        for example in validation_examples
    ])

    pairs = [
        [example["source_text"], example["response_text"]]
        for example in validation_examples
    ]

    print(f"Training records available: {len(examples)}")
    print(f"Development records: {len(dev_indices)}")
    print(f"Validation records: {len(validation_examples)}")
    print(f"Missing source records skipped: {missing_sources}")
    print(f"Validation positives: {int(np.sum(y_validation == 1))}")
    print(f"Validation negatives: {int(np.sum(y_validation == 0))}")

    # Load the NLI model.
    print(f"\nLoading NLI model: {MODEL_NAME}")

    model = CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    # Score validation examples.
    print("\nScoring validation responses...")

    (
        contradiction_probs,
        entailment_probs,
        neutral_probs,
        unsupported_scores,
    ) = calculate_unsupported_scores(model, pairs)

    comparison_rows = []

    # Compare thresholds using validation data only.
    for threshold in THRESHOLDS:
        y_pred = (unsupported_scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1],
        ).ravel()

        comparison_rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_validation, y_pred),
            "precision": precision_score(
                y_validation, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_validation, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_validation, y_pred, zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    # Highest F1 wins; ties prefer precision, then higher threshold.
    best_row = max(
        comparison_rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["threshold"],
        ),
    )

    threshold_data = {
        "model_name": MODEL_NAME,
        "threshold": best_row["threshold"],
        "selection_metric": "validation_f1",
        "validation_f1": best_row["f1"],
        "validation_accuracy": best_row["accuracy"],
        "validation_precision": best_row["precision"],
        "validation_recall": best_row["recall"],
        "validation_records": len(validation_examples),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "split_method": "StratifiedGroupKFold",
        "group_column": "source_id",
        "max_length": MAX_LENGTH,
        "unsupported_score": "P(contradiction) + P(neutral)",
        "source_text_fields": ["source", "source_info"],
        "source_field_separator": " ",
    }

    with open(
        OUTPUT_THRESHOLD, "w", encoding="utf-8"
    ) as file:
        json.dump(threshold_data, file, indent=4)

    with open(
        OUTPUT_COMPARISON,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = list(comparison_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\nNLI VALIDATION THRESHOLD RESULTS")
    print("=" * 65)
    print(
        f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1-score':>10}"
    )

    for row in comparison_rows:
        print(
            f"{row['threshold']:>10.2f} "
            f"{row['accuracy']:>10.4f} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f}"
        )

    print("\nSELECTED THRESHOLD")
    print("=" * 65)
    print(f"Threshold: {best_row['threshold']:.2f}")
    print(f"Validation F1-score: {best_row['f1']:.4f}")
    print(f"Validation accuracy: {best_row['accuracy']:.4f}")
    print(f"Validation precision: {best_row['precision']:.4f}")
    print(f"Validation recall: {best_row['recall']:.4f}")

    print(f"\nThreshold saved to: {OUTPUT_THRESHOLD}")
    print(f"Comparison table saved to: {OUTPUT_COMPARISON}")


if __name__ == "__main__":
    main()
