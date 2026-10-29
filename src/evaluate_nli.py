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
    roc_auc_score,
)
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_FILE = DATA_DIR / "source_info.jsonl"
THRESHOLD_FILE = PROJECT_ROOT / "nli_threshold_validation.json"

OUTPUT_RESULTS = PROJECT_ROOT / "nli_test_results.csv"
OUTPUT_METRICS = PROJECT_ROOT / "nli_test_metrics.json"

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

BATCH_SIZE = 16
MAX_LENGTH = 512


# --------------------------------------------------
# 2. FIND THE RESPONSE DATASET
# --------------------------------------------------

def find_response_file():
    candidates = [
        DATA_DIR / "response.jsonl",
        DATA_DIR / "response(1).jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find response.jsonl or response(1).jsonl "
        f"inside: {DATA_DIR}"
    )


RESPONSE_FILE = find_response_file()


# --------------------------------------------------
# 3. LOAD JSONL FILES
# --------------------------------------------------

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


# --------------------------------------------------
# 4. NORMALIZE SOURCE TEXT
# --------------------------------------------------

def to_text(value):
    """
    Convert strings, dictionaries, lists, and other values
    into readable text.

    Keep this function identical to the version used during
    validation threshold tuning.
    """

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
    """
    Build a mapping:
        source_id -> combined source and source_info text
    """

    source_lookup = {}

    for record in source_records:
        source_id = record.get("source_id")

        if source_id is None:
            continue

        parts = [
            to_text(record.get("source", "")),
            to_text(record.get("source_info", "")),
        ]

        source_text = " ".join(
            part for part in parts if part
        ).strip()

        source_lookup[str(source_id)] = source_text

    return source_lookup


# --------------------------------------------------
# 5. CREATE GROUND-TRUTH LABELS
# --------------------------------------------------

def get_actual_label(record):
    """
    Label 1: response has one or more annotated hallucination spans.
    Label 0: response has no annotated hallucination spans.

    These are the dataset's annotation labels, not an independent
    guarantee of factual truth.
    """

    labels = record.get("labels", [])

    return int(isinstance(labels, list) and len(labels) > 0)


# --------------------------------------------------
# 6. IDENTIFY NLI CLASS INDICES
# --------------------------------------------------

def get_nli_class_indices(model):
    """
    Resolve class indices from the model configuration.

    Fallback for cross-encoder/nli-deberta-v3-small:
        index 0 = contradiction
        index 1 = entailment
        index 2 = neutral
    """

    config = model.model.config

    label2id = getattr(config, "label2id", None) or {}

    normalized = {
        str(label).lower().replace("_", "").replace("-", ""): int(index)
        for label, index in label2id.items()
    }

    contradiction = normalized.get("contradiction")
    entailment = normalized.get("entailment")
    neutral = normalized.get("neutral")

    if all(
        index is not None
        for index in [contradiction, entailment, neutral]
    ):
        return contradiction, entailment, neutral

    id2label = getattr(config, "id2label", None) or {}

    normalized_id2label = {
        int(index): str(label).lower().replace("_", "").replace("-", "")
        for index, label in id2label.items()
    }

    if normalized_id2label:
        found = {}

        for index, label in normalized_id2label.items():
            if label in {"contradiction", "entailment", "neutral"}:
                found[label] = index

        if all(
            name in found
            for name in ["contradiction", "entailment", "neutral"]
        ):
            return (
                found["contradiction"],
                found["entailment"],
                found["neutral"],
            )

    return 0, 1, 2


# --------------------------------------------------
# 7. CALCULATE UNSUPPORTED SCORES
# --------------------------------------------------

def calculate_unsupported_scores(model, pairs):
    """
    Input pair:
        [source/premise, generated response/hypothesis]

    Unsupported score:
        P(contradiction) + P(neutral)

    Higher scores indicate less clear support from the source.
    This is a heuristic, not proof of factual incorrectness.
    """

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


# --------------------------------------------------
# 8. MAIN TEST EVALUATION
# --------------------------------------------------

def main():
    print("NLI test evaluation started.")
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Source file: {SOURCE_FILE}")
    print(f"Response file: {RESPONSE_FILE}")

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_FILE}\n"
            "Run src/tune_nli_validation.py first."
        )

    # Load the threshold selected using validation data.
    with open(
        THRESHOLD_FILE, "r", encoding="utf-8"
    ) as file:
        threshold_data = json.load(file)

    if threshold_data.get("model_name") != MODEL_NAME:
        raise ValueError(
            "The saved threshold was produced by a different model. "
            f"Saved model: {threshold_data.get('model_name')}; "
            f"current model: {MODEL_NAME}"
        )

    threshold = float(threshold_data["threshold"])

    # Load the datasets.
    source_records = load_jsonl(SOURCE_FILE)
    response_records = load_jsonl(RESPONSE_FILE)

    source_lookup = build_source_lookup(source_records)

    # The test split is reserved for final evaluation.
    test_records = [
        record
        for record in response_records
        if str(record.get("split", "")).lower() == "test"
    ]

    if not test_records:
        raise ValueError(
            "No records with split='test' were found "
            "in the response file."
        )

    examples = []
    missing_sources = 0
    empty_responses = 0

    for record in test_records:
        source_id = record.get("source_id")
        source_text = source_lookup.get(str(source_id), "")
        response_text = to_text(record.get("response", ""))

        if not source_text:
            missing_sources += 1
            continue

        if not response_text:
            empty_responses += 1
            continue

        examples.append(
            {
                "id": record.get("id"),
                "source_id": source_id,
                "actual_label": get_actual_label(record),
                "source_text": source_text,
                "response_text": response_text,
            }
        )

    if not examples:
        raise ValueError(
            "No valid test responses remain after data preparation."
        )

    print(f"\nTest records in dataset: {len(test_records)}")
    print(f"Test records evaluated: {len(examples)}")
    print(f"Missing source records skipped: {missing_sources}")
    print(f"Empty responses skipped: {empty_responses}")
    print(f"Validation-selected threshold: {threshold:.2f}")

    # Load the model.
    print(f"\nLoading NLI model: {MODEL_NAME}")

    model = CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    # Prepare source-response pairs.
    pairs = [
        [example["source_text"], example["response_text"]]
        for example in examples
    ]

    # Score all test examples.
    print("\nScoring test responses...")

    (
        contradiction_probs,
        entailment_probs,
        neutral_probs,
        unsupported_scores,
    ) = calculate_unsupported_scores(model, pairs)

    # Convert continuous scores to predicted labels.
    y_true = np.array(
        [example["actual_label"] for example in examples]
    )

    y_pred = (
        unsupported_scores >= threshold
    ).astype(int)

    # Calculate classification metrics.
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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # AUROC uses continuous scores, not thresholded predictions.
    try:
        auroc = float(
            roc_auc_score(y_true, unsupported_scores)
        )
    except ValueError:
        auroc = None

    # --------------------------------------------------
    # 9. SAVE INDIVIDUAL TEST PREDICTIONS
    # --------------------------------------------------

    with open(
        OUTPUT_RESULTS,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "id",
            "source_id",
            "actual_label",
            "predicted_label",
            "contradiction_probability",
            "entailment_probability",
            "neutral_probability",
            "unsupported_score",
            "threshold",
            "source_missing",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, example in enumerate(examples):
            writer.writerow(
                {
                    "id": example["id"],
                    "source_id": example["source_id"],
                    "actual_label": int(y_true[index]),
                    "predicted_label": int(y_pred[index]),
                    "contradiction_probability": round(
                        float(contradiction_probs[index]), 6
                    ),
                    "entailment_probability": round(
                        float(entailment_probs[index]), 6
                    ),
                    "neutral_probability": round(
                        float(neutral_probs[index]), 6
                    ),
                    "unsupported_score": round(
                        float(unsupported_scores[index]), 6
                    ),
                    "threshold": threshold,
                    "source_missing": False,
                }
            )

    # --------------------------------------------------
    # 10. SAVE SUMMARY METRICS
    # --------------------------------------------------

    metrics = {
        "method": "NLI",
        "model_name": MODEL_NAME,
        "test_records_in_dataset": len(test_records),
        "test_records_evaluated": len(examples),
        "positive_labels": int(np.sum(y_true == 1)),
        "negative_labels": int(np.sum(y_true == 0)),
        "predicted_positive": int(np.sum(y_pred == 1)),
        "predicted_negative": int(np.sum(y_pred == 0)),
        "threshold": threshold,
        "threshold_selection": "validation_f1",
        "unsupported_score": "P(contradiction) + P(neutral)",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "missing_sources_skipped": missing_sources,
        "empty_responses_skipped": empty_responses,
        "max_length": MAX_LENGTH,
    }

    with open(
        OUTPUT_METRICS, "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------
    # 11. PRINT RESULTS
    # --------------------------------------------------

    print("\nNLI TEST RESULTS")
    print("=" * 50)
    print(f"Test records evaluated: {len(examples)}")
    print(f"Positive labels: {metrics['positive_labels']}")
    print(f"Negative labels: {metrics['negative_labels']}")
    print(f"Predicted positive: {metrics['predicted_positive']}")
    print(f"Predicted negative: {metrics['predicted_negative']}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    if auroc is not None:
        print(f"AUROC:     {auroc:.4f}")
    else:
        print("AUROC:     Not available for this test-label distribution.")

    print("\nConfusion matrix:")
    print(np.array([[tn, fp], [fn, tp]]))
    print("\nRows = actual labels [0, 1]")
    print("Columns = predicted labels [0, 1]")

    print(f"\nDetailed predictions saved to: {OUTPUT_RESULTS}")
    print(f"Summary metrics saved to: {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()