
import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

source_path = data_dir / "source_info.jsonl"
response_path = data_dir / "response.jsonl"

if not response_path.exists():
    response_path = data_dir / "response(1).jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


sources = load_jsonl(source_path)
responses = load_jsonl(response_path)

source_lookup = {
    str(item["source_id"]): item["source_info"]
    for item in sources
}

# Use only the official test split for this pilot.
test_records = [
    row for row in responses
    if row.get("split") == "test"
][:100]

# Ground-truth label:
# 1 = at least one annotated hallucination span
# 0 = no annotated hallucination spans
def get_true_label(row):
    labels = row.get("labels", [])

    if isinstance(labels, str):
        labels = json.loads(labels)

    return int(len(labels) > 0)


test_records = [
    row for row in test_records
    if str(row["source_id"]) in source_lookup
    and row.get("response", "").strip()
]

y_true = [get_true_label(row) for row in test_records]

# NLI input: source evidence first, generated response second.
pairs = [
    (
        source_lookup[str(row["source_id"])],
        row["response"]
    )
    for row in test_records
]

print("Loading pretrained NLI model...")
model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    max_length=512
)

print("Scoring", len(pairs), "responses...")
logits = model.predict(
    pairs,
    batch_size=4,
    show_progress_bar=True
)

# Model output order:
# 0 = contradiction, 1 = entailment, 2 = neutral
# Convert raw scores to probabilities.
logits = np.asarray(logits)
logits = logits - logits.max(axis=1, keepdims=True)
probabilities = np.exp(logits)
probabilities = probabilities / probabilities.sum(
    axis=1, keepdims=True
)

# Initial heuristic:
# predict hallucination if contradiction + neutral
# probability is greater than 0.5.
unsupported_score = (
    probabilities[:, 0] + probabilities[:, 2]
)
y_pred = (unsupported_score > 0.5).astype(int)

print("\nNLI PILOT RESULTS")
print("Records evaluated:", len(y_true))
print("Positive labels:", sum(y_true))
print("Negative labels:", len(y_true) - sum(y_true))
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(
    precision_score(y_true, y_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_true, y_pred, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_true, y_pred, zero_division=0), 4
))

print("\nConfusion matrix (rows=actual, columns=predicted)")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["No annotated spans", "Has annotated spans"],
    zero_division=0
))

# Save scores so we can inspect mistakes later.
output_path = project_dir / "nli_pilot_results.csv"

import csv

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "response_id",
        "true_label",
        "contradiction_probability",
        "entailment_probability",
        "neutral_probability",
        "unsupported_score",
        "predicted_label"
    ])

    for row, true, probs, score, pred in zip(
        test_records, y_true, probabilities,
        unsupported_score, y_pred
    ):
        writer.writerow([
            row["id"],
            true,
            round(float(probs[0]), 6),
            round(float(probs[1]), 6),
            round(float(probs[2]), 6),
            round(float(score), 6),
            int(pred)
        ])

print("\nSaved scores to:", output_path)
