"""Verify test outputs and reproduce the source-grouped validation split."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCE_FILE = DATA / "source_info.jsonl"
RESPONSE_CANDIDATES = [DATA / "response.jsonl", DATA / "response(1).jsonl"]
RESPONSE_FILE = next((p for p in RESPONSE_CANDIDATES if p.exists()), None)
NLI_CSV, OVERLAP_CSV = ROOT / "nli_test_results.csv", ROOT / "overlap_test_results.csv"
NLI_JSON, OVERLAP_JSON, BASELINE_JSON = ROOT / "nli_test_metrics.json", ROOT / "overlap_test_metrics.json", ROOT / "baseline_test_metrics.json"
NLI_THRESHOLD, OVERLAP_THRESHOLD = ROOT / "nli_threshold_validation.json", ROOT / "overlap_threshold_validation.json"
RANDOM_STATE = 42


def check(condition, message):
    print(f"[{'PASS' if condition else 'CHECK'}] {message}")


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON in {path.name}, line {line_no}: {e}") from e
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def label_of(record):
    labels = record.get("labels", [])
    return int(isinstance(labels, list) and len(labels) > 0)


def verify_metrics(csv_path, json_path, name):
    rows = read_csv(csv_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    y_true = np.array([int(r["actual_label"]) for r in rows])
    y_pred = np.array([int(r["predicted_label"]) for r in rows])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n--- {name} metric verification ---")
    print(f"Rows: {len(rows)}; positives: {int(y_true.sum())}; negatives: {int((y_true == 0).sum())}")
    print("Confusion matrix:", matrix)
    for key, value in metrics.items():
        stored = saved.get(key, saved.get(key + "_score"))
        check(stored is not None and abs(float(stored) - float(value)) < 0.00011,
              f"{name} {key}: calculated={value:.4f}, saved={stored}")
    check(saved.get("confusion_matrix") == matrix, f"{name} confusion matrix matches JSON")
    score_key = "unsupported_score" if rows and "unsupported_score" in rows[0] else None
    if score_key and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, [float(r[score_key]) for r in rows])
        stored_auc = saved.get("auroc")
        check(stored_auc is not None and abs(float(stored_auc) - auc) < 0.00011,
              f"{name} AUROC: calculated={auc:.4f}, saved={stored_auc}")
    ids = [str(r.get("id")) for r in rows]
    check(len(ids) == len(set(ids)), f"{name} response IDs are unique")
    return {str(r.get("id")) for r in rows}, rows


def validation_split_check(train_records, source_lookup, name):
    """Recreate the same filtered examples and grouped split used by both tuners."""
    examples = []
    for r in train_records:
        sid = r.get("source_id")
        source = source_lookup.get(str(sid), "")
        response = r.get("response", "")
        if isinstance(response, (dict, list)):
            response = str(response)
        if not str(source).strip() or not str(response).strip():
            continue
        examples.append((str(sid), label_of(r)))
    labels = np.array([x[1] for x in examples], dtype=int)
    groups = np.array([x[0] for x in examples])
    idx = np.arange(len(examples))
    if len(np.unique(labels)) < 2:
        check(False, f"{name} grouped validation has both label classes")
        return
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    dev_idx, val_idx = next(splitter.split(idx, labels, groups=groups))
    dev_sources, val_sources = set(groups[dev_idx]), set(groups[val_idx])
    overlap = dev_sources & val_sources
    print(f"\n--- {name} source-grouped validation check ---")
    print(f"Development records: {len(dev_idx)}; validation records: {len(val_idx)}")
    print(f"Shared source IDs: {len(overlap)}")
    check(not overlap, f"{name} development and validation source IDs are disjoint")


def main():
    print("=" * 65 + "\nFINAL EXPERIMENT VERIFICATION\n" + "=" * 65)
    required = [SOURCE_FILE, RESPONSE_FILE, NLI_CSV, OVERLAP_CSV, NLI_JSON, OVERLAP_JSON, BASELINE_JSON, NLI_THRESHOLD, OVERLAP_THRESHOLD]
    for path in required:
        check(path is not None and path.exists(), f"File exists: {path.name if path else 'response dataset'}")
    if any(path is None or not path.exists() for path in required):
        print("Missing required file(s); fix paths before continuing.")
        return

    sources, responses = read_jsonl(SOURCE_FILE), read_jsonl(RESPONSE_FILE)
    source_lookup = {}
    for r in sources:
        sid = r.get("source_id")
        if sid is not None:
            # Match source fields used by the current evaluators.
            def txt(v):
                if v is None: return ""
                if isinstance(v, str): return v.strip()
                if isinstance(v, dict): return " ".join(x for x in (txt(y) for y in v.values()) if x).strip()
                if isinstance(v, list): return " ".join(x for x in (txt(y) for y in v) if x).strip()
                return str(v).strip()
            source_lookup[str(sid)] = " ".join(x for x in [txt(r.get("source", "")), txt(r.get("source_info", ""))] if x).strip()

    all_ids = [str(r.get("id")) for r in responses]
    check(len(all_ids) == len(set(all_ids)), "Response IDs are unique in source dataset")
    train = [r for r in responses if str(r.get("split", "")).lower() == "train"]
    test = [r for r in responses if str(r.get("split", "")).lower() == "test"]
    print("\n--- Dataset split checks ---")
    print(f"Source records: {len(sources)}; training responses: {len(train)}; test responses: {len(test)}")
    print(f"Test positives: {sum(label_of(r) for r in test)}; negatives: {sum(1-label_of(r) for r in test)}")
    check(len(test) == 2700, "Expected 2,700 test responses")
    train_sources = {str(r.get("source_id")) for r in train if r.get("source_id") is not None}
    test_sources = {str(r.get("source_id")) for r in test if r.get("source_id") is not None}
    check(not (train_sources & test_sources), "Training and test source IDs do not overlap")

    validation_split_check(train, source_lookup, "Shared grouped split")
    nli_ids, nli_rows = verify_metrics(NLI_CSV, NLI_JSON, "NLI")
    overlap_ids, overlap_rows = verify_metrics(OVERLAP_CSV, OVERLAP_JSON, "Lexical overlap")
    dataset_test_ids = {str(r.get("id")) for r in test}
    check(nli_ids == overlap_ids, "NLI and lexical overlap use the same test response IDs")
    check(nli_ids == dataset_test_ids, "Evaluation CSVs cover the complete test split")
    labels_by_id = {str(r.get("id")): label_of(r) for r in test}
    for name, rows in [("NLI", nli_rows), ("Lexical overlap", overlap_rows)]:
        mismatches = sum(int(r["actual_label"]) != labels_by_id.get(str(r["id"]), -1) for r in rows)
        check(mismatches == 0, f"{name} labels match dataset annotations")

    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    expected_accuracy = sum(1 - label_of(r) for r in test) / len(test)
    check(abs(float(baseline.get("accuracy", -1)) - expected_accuracy) < 0.00011,
          "Majority baseline accuracy matches negative-class rate")

    for name, path in [("NLI", NLI_THRESHOLD), ("Lexical overlap", OVERLAP_THRESHOLD)]:
        info = json.loads(path.read_text(encoding="utf-8"))
        check(info.get("split_method") == "StratifiedGroupKFold" and info.get("group_column") == "source_id",
              f"{name} threshold metadata records source-grouped validation")
    print("\nVerification finished. Resolve every [CHECK] before reporting final results.")


if __name__ == "__main__":
    main()
