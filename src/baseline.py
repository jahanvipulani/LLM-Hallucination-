
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Actual labels from the dataset
y_true = ["no"] * 3692 + ["yes"] * 815

# Baseline: predict "no" for every record
y_pred = ["no"] * len(y_true)

print("Baseline Evaluation")
print("-------------------")
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))

print(
    "Precision:",
    round(
        precision_score(
            y_true, y_pred,
            pos_label="yes",
            zero_division=0
        ), 4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_true, y_pred,
            pos_label="yes",
            zero_division=0
        ), 4
    )
)

print(
    "F1-score:",
    round(
        f1_score(
            y_true, y_pred,
            pos_label="yes",
            zero_division=0
        ), 4
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=["no", "yes"]))
