
import json
import random
import numpy as np
from pathlib import Path
from sentence_transformers import CrossEncoder
from sklearn.metrics import f1_score

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

source_path = data_dir / "source_info.jsonl"
response_path = data_dir / "response.jsonl"

if not response_path.exists():
    response_path = data_dir / "response(1).jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


sources = load_jsonl(source_path)
responses = load_jsonl(response_path)

def source_to_text(value):
    """Convert source strings, dictionaries, or lists into readable text."""
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {source_to_text(item)}"
            for key, item in value.items()
            if item is not None
        )

    if isinstance(value, list):
        return "\n".join(source_to_text(item) for item in value)

    if value is None:
        return ""

    return str(value)

source_lookup = {
    str(row["source_id"]): source_to_text(row["source_info"])
    for row in sources
}

train = [
    row for row in responses
    if row.get("split") == "train"
    and str(row["source_id"]) in source_lookup
    and row.get("response", "").strip()
]

# Reproducible random sample
random.seed(42)
sample = random.sample(train, min(200, len(train)))

def true_label(row):
    labels = row.get("labels", [])
    if isinstance(labels, str):
        labels = json.loads(labels)
    return int(len(labels) > 0)

y_true = np.array([true_label(row) for row in sample])

pairs = [
    (
        source_lookup[str(row["source_id"])],
        row["response"]
    )
    for row in sample
]

print("Loading NLI model...")
model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    max_length=512
)

print("Scoring training sample:", len(sample))
logits = np.asarray(model.predict(pairs, batch_size=4))

# Output order: contradiction, entailment, neutral
logits = logits - logits.max(axis=1, keepdims=True)
probs = np.exp(logits)
probs = probs / probs.sum(axis=1, keepdims=True)

# Score for potentially unsupported responses
scores = probs[:, 0] + probs[:, 2]

best_threshold = None
best_f1 = -1

print("\nThreshold results")
for threshold in np.arange(0.1, 1.0, 0.1):
    predictions = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, predictions, zero_division=0)
    print(f"Threshold {threshold:.1f}: F1 = {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)

print("\nBest threshold on training sample:", best_threshold)
print("Best training-sample F1:", round(best_f1, 4))

# Save the selected threshold for the next experiment
output_path = project_dir / "nli_threshold.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump({
        "threshold": best_threshold,
        "training_sample_size": len(sample),
        "random_seed": 42,
        "selection_metric": "F1-score"
    }, f, indent=2)

print("Saved threshold to:", output_path)
