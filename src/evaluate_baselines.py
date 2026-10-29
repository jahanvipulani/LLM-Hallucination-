print("Baseline script started", flush=True)
import json
import csv
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

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

SOURCE_PATH = DATA_DIR / "source_info.jsonl"
RESPONSE_PATH = DATA_DIR / "response.jsonl"

if not RESPONSE_PATH.exists():
    RESPONSE_PATH = DATA_DIR / "response(1).jsonl"

OUTPUT_PATH = PROJECT_DIR / "baseline_test_metrics.json"


# --------------------------------------------------
# 2. Load JSONL files
# --------------------------------------------------

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_label(row):
    labels = row.get("labels", [])

    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError:
            labels = []

    return int(isinstance(labels, list) and len(labels) > 0)


# --------------------------------------------------
# 3. Load test data
# --------------------------------------------------

sources = load_jsonl(SOURCE_PATH)
responses = load_jsonl(RESPONSE_PATH)

test_records = [
    row for row in responses
    if str(row.get("split", "")).lower() == "test"
]

y_true = np.array(
    [get_label(row) for row in test_records],
    dtype=int,
)

# Majority baseline: predict the most common class.
majority_class = int(np.bincount(y_true).argmax())

y_pred = np.full(
    len(y_true),
    majority_class,
    dtype=int,
)

# --------------------------------------------------
# 4. Calculate metrics
# --------------------------------------------------

metrics = {
    "method": "Majority baseline",
    "test_records": int(len(y_true)),
    "majority_class": majority_class,
    "actual_positive": int(y_true.sum()),
    "actual_negative": int((y_true == 0).sum()),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "confusion_matrix": confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist(),
    "auroc": None,
}

print("\nMAJORITY BASELINE RESULTS")
print("=" * 45)
print("Test records:", metrics["test_records"])
print("Majority class:", majority_class)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-score:  {metrics['f1_score']:.4f}")
print("Confusion matrix:")
print(np.array(metrics["confusion_matrix"]))
print("AUROC: Not applicable to this constant prediction.")

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to:")
print(OUTPUT_PATH)
