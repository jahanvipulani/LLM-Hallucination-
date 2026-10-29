
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Locate and load the dataset
project_dir = Path(__file__).resolve().parent.parent
path = project_dir / "data" / "general_data.json"

with path.open("r", encoding="utf-8-sig") as f:
    content = f.read()

try:
    data = json.loads(content)
except json.JSONDecodeError:
    data = [
        json.loads(line)
        for line in content.splitlines()
        if line.strip()
    ]

# Extract actual labels
y_true = [
    str(row["hallucination"]).strip().lower()
    for row in data
]

# Predict the majority class for every record
majority_label = max(set(y_true), key=y_true.count)
y_pred = [majority_label] * len(y_true)

# Evaluate
print("Dataset records:", len(y_true))
print("Majority label:", majority_label)
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(
    precision_score(y_true, y_pred, pos_label="yes", zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_true, y_pred, pos_label="yes", zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_true, y_pred, pos_label="yes", zero_division=0), 4
))

print("\nConfusion matrix (labels: no, yes):")
print(confusion_matrix(y_true, y_pred, labels=["no", "yes"]))
