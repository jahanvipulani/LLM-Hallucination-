"""Tune lexical-overlap threshold using a source-grouped validation split."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCE_FILE = DATA / "source_info.jsonl"
RESPONSE_CANDIDATES = [DATA / "response.jsonl", DATA / "response(1).jsonl"]
RESPONSE_FILE = next((p for p in RESPONSE_CANDIDATES if p.exists()), None)
OUTPUT_THRESHOLD = ROOT / "overlap_threshold_validation.json"
OUTPUT_COMPARISON = ROOT / "overlap_validation_comparison.csv"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.10, 1.00, 0.10)]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path.name}, line {line_no}: {e}") from e
    return rows


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [to_text(v) for v in value.values()]
        return " ".join(p for p in parts if p).strip()
    if isinstance(value, (list, tuple)):
        parts = [to_text(v) for v in value]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()


def label_of(record):
    labels = record.get("labels", [])
    return int(isinstance(labels, list) and len(labels) > 0)


def tokenize(text):
    return [w.strip(".,!?;:\"'()[]{}") for w in text.lower().split() if w.strip(".,!?;:\"'()[]{}")]


def overlap_score(source, response):
    source_tokens = set(tokenize(source))
    response_tokens = tokenize(response)
    if not response_tokens:
        return 0.0
    return sum(token in source_tokens for token in response_tokens) / len(response_tokens)


def main():
    print("Lexical-overlap grouped validation threshold tuning started.")
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)
    if RESPONSE_FILE is None:
        raise FileNotFoundError(f"Could not find response.jsonl in {DATA}")

    source_records = read_jsonl(SOURCE_FILE)
    response_records = read_jsonl(RESPONSE_FILE)
    source_lookup = {}
    for row in source_records:
        sid = row.get("source_id")
        if sid is not None:
            source_lookup[str(sid)] = " ".join(
                part for part in [to_text(row.get("source", "")), to_text(row.get("source_info", ""))]
                if part
            ).strip()

    train_records = [r for r in response_records if str(r.get("split", "")).lower() == "train"]
    examples = []
    missing_sources = 0
    for row in train_records:
        sid = row.get("source_id")
        source = source_lookup.get(str(sid), "")
        response = to_text(row.get("response", ""))
        if not source:
            missing_sources += 1
            continue
        if not response:
            continue
        examples.append({
            "id": row.get("id"), "source_id": str(sid),
            "source_text": source, "response_text": response,
            "actual_label": label_of(row),
        })

    if not examples:
        raise ValueError("No valid training examples after preprocessing.")
    y = np.array([e["actual_label"] for e in examples], dtype=int)
    groups = np.array([e["source_id"] for e in examples])
    indices = np.arange(len(examples))
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both classes.")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    dev_idx, val_idx = next(splitter.split(indices, y, groups=groups))
    dev_groups, val_groups = set(groups[dev_idx]), set(groups[val_idx])
    shared = dev_groups & val_groups
    if shared:
        raise AssertionError(f"Source leakage detected: {len(shared)} shared source IDs")

    validation = [examples[i] for i in val_idx]
    y_val = np.array([e["actual_label"] for e in validation], dtype=int)
    scores = np.array([1.0 - overlap_score(e["source_text"], e["response_text"]) for e in validation])
    rows = []
    for threshold in THRESHOLDS:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_val, pred)),
            "precision": float(precision_score(y_val, pred, zero_division=0)),
            "recall": float(recall_score(y_val, pred, zero_division=0)),
            "f1": float(f1_score(y_val, pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    best = max(rows, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    threshold_info = {
        "method": "lexical_overlap",
        "threshold": best["threshold"],
        "selection_metric": "validation_f1",
        "validation_f1": best["f1"],
        "validation_accuracy": best["accuracy"],
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_records": len(validation),
        "development_records": len(dev_idx),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "split_method": "StratifiedGroupKFold",
        "group_column": "source_id",
        "unsupported_score": "1 - response-token lexical overlap with source",
        "source_text_fields": ["source", "source_info"],
        "source_field_separator": " ",
    }
    OUTPUT_THRESHOLD.write_text(json.dumps(threshold_info, indent=4), encoding="utf-8")
    with OUTPUT_COMPARISON.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Training records available: {len(examples)}")
    print(f"Development records: {len(dev_idx)}")
    print(f"Validation records: {len(validation)}")
    print(f"Missing source records skipped: {missing_sources}")
    print(f"Shared source IDs between development and validation: {len(shared)}")
    print("\nThreshold  Accuracy  Precision  Recall  F1")
    for r in rows:
        print(f"{r['threshold']:.2f}       {r['accuracy']:.4f}    {r['precision']:.4f}     {r['recall']:.4f}  {r['f1']:.4f}")
    print(f"\nSelected threshold: {best['threshold']:.2f}")
    print(f"Validation F1: {best['f1']:.4f}")
    print(f"Saved: {OUTPUT_THRESHOLD}")
    print(f"Saved: {OUTPUT_COMPARISON}")


if __name__ == "__main__":
    main()
