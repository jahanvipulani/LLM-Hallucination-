
import json
import random
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import CrossEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

source_path = data_dir / "source_info.jsonl"
response_path = data_dir / "response.jsonl"

if not response_path.exists():
    response_path = data_dir / "response(1).jsonl"

output_path = project_dir / "nli_threshold_comparison.csv"

SAMPLE_SIZE = 200
RANDOM_SEED = 42
BATCH_SIZE = 4

thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def source_to_text(value):
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {source_to_text(item)}"
            for key, item in value.items()
            if item is not None
        )

    if isinstance(value, list):
        return "\n".join(
            source_to_text(item) for item in value
        )

    if value is None:
        return ""

    return str(value)


def true_label(row):
    labels = row.get("labels", [])

    if isinstance(labels, str):
        labels = json.loads(labels)

    return int(
        isinstance(labels, list) and len(labels) > 0
    )


def normalize_label(label):
    return (
        str(label)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_label_index(id2label, keyword):
    for index, label in id2label.items():
        if keyword in normalize_label(label):
            return index

    return None


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

print("Loading datasets...")

sources = load_jsonl(source_path)
responses = load_jsonl(response_path)

source_lookup = {
    str(row["source_id"]): source_to_text(
        row.get("source_info")
    )
    for row in sources
}

train_records = [
    row for row in responses
    if str(row.get("split", "")).lower() == "train"
    and str(row.get("source_id")) in source_lookup
    and str(row.get("response", "")).strip()
]

if not train_records:
    raise ValueError("No eligible training records found.")

random.seed(RANDOM_SEED)

sample = random.sample(
    train_records,
    min(SAMPLE_SIZE, len(train_records))
)

pairs = [
    (
        source_lookup[str(row["source_id"])],
        row["response"],
    )
    for row in sample
]

y_true = np.array(
    [true_label(row) for row in sample],
    dtype=int,
)

print(f"Training records available: {len(train_records)}")
print(f"Sample size: {len(sample)}")
print(f"Positive labels: {int(y_true.sum())}")
print(f"Negative labels: {int(len(y_true) - y_true.sum())}")


# --------------------------------------------------
# 4. Load model and verify output labels
# --------------------------------------------------

print("\nLoading NLI model...")

model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    max_length=512,
)

id2label = {
    int(index): label
    for index, label in model.model.config.id2label.items()
}

print("Model label mapping:", id2label)

contradiction_idx = find_label_index(
    id2label, "contradiction"
)
neutral_idx = find_label_index(
    id2label, "neutral"
)
entailment_idx = find_label_index(
    id2label, "entailment"
)

if any(
    index is None
    for index in [
        contradiction_idx,
        neutral_idx,
        entailment_idx,
    ]
):
    raise ValueError(
        "Could not identify all NLI labels. "
        "Check the printed model label mapping."
    )


# --------------------------------------------------
# 5. Score each pair once
# --------------------------------------------------

print("\nScoring training sample...")

logits = np.asarray(
    model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
)

# Stable softmax: convert logits to probabilities.
logits = logits - logits.max(axis=1, keepdims=True)
exp_logits = np.exp(logits)
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

# Same score definition used in tune_nli.py.
unsupported_scores = (
    probs[:, contradiction_idx]
    + probs[:, neutral_idx]
)


# --------------------------------------------------
# 6. Compare thresholds
# --------------------------------------------------

rows = []

print("\nTHRESHOLD COMPARISON")
print("-" * 95)
print(
    f"{'Threshold':>9} {'Accuracy':>9} {'Precision':>10} "
    f"{'Recall':>9} {'F1':>9} {'FP':>7} {'FN':>7}"
)
print("-" * 95)

for threshold in thresholds:
    y_pred = (
        unsupported_scores >= threshold
    ).astype(int)

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

    # Confusion matrix values, explicitly calculated.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    rows.append({
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sample_size": len(sample),
        "random_seed": RANDOM_SEED,
    })

    print(
        f"{threshold:9.1f} {accuracy:9.4f} "
        f"{precision:10.4f} {recall:9.4f} "
        f"{f1:9.4f} {fp:7d} {fn:7d}"
    )


# --------------------------------------------------
# 7. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame(rows)

results_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

# Report the threshold with the highest sample F1.
best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

print("\n" + "=" * 55)
print("BEST THRESHOLD ON THIS TRAINING SAMPLE")
print("=" * 55)
print(f"Threshold: {best_row['threshold']:.1f}")
print(f"F1-score:  {best_row['f1_score']:.4f}")
print(f"Precision: {best_row['precision']:.4f}")
print(f"Recall:    {best_row['recall']:.4f}")
print(f"Accuracy:  {best_row['accuracy']:.4f}")
print(f"False positives: {int(best_row['false_positive'])}")
print(f"False negatives: {int(best_row['false_negative'])}")

print(f"\nComparison saved to:\n{output_path}")
print("\nExperiment completed.")
