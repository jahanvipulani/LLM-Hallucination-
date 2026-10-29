
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(filename):
    """Load a JSON file from the project root."""
    filepath = PROJECT_ROOT / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            "Make sure the evaluation scripts have been run first."
        )

    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_test_count(metrics, method_name):
    """Read the test-record count using the correct key."""
    if method_name == "NLI":
        count = metrics.get("test_records_evaluated")
    else:
        count = metrics.get("test_records")

    if count is None:
        raise ValueError(
            f"Test-record count is missing for {method_name}."
        )

    return count


def get_metric(metrics, possible_keys):
    """Return the first available metric from a list of key names."""
    for key in possible_keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]

    return None


def main():
    print("Loading evaluation metrics...")

    # Load metrics from the three evaluation methods
    baseline = load_json("baseline_test_metrics.json")
    overlap = load_json("overlap_test_metrics.json")
    nli = load_json("nli_test_metrics.json")

    # Read record counts using the appropriate JSON keys
    baseline_count = get_test_count(baseline, "Majority baseline")
    overlap_count = get_test_count(overlap, "Lexical overlap")
    nli_count = get_test_count(nli, "NLI")

    record_counts = {
        baseline_count,
        overlap_count,
        nli_count,
    }

    # Validate the number of test records
    if len(record_counts) != 1:
        raise ValueError(
            "The methods have different test-record counts. "
            f"Baseline: {baseline_count}, "
            f"Overlap: {overlap_count}, "
            f"NLI: {nli_count}. "
            "Verify that all methods use the same test split."
        )

    print(
        f"Verified: all methods evaluated "
        f"{baseline_count} test records."
    )

    methods = [
        ("Majority baseline", baseline),
        ("Lexical overlap", overlap),
        ("NLI", nli),
    ]

    # Metric names may differ between evaluation scripts.
    # This helper supports common naming conventions.
    metric_aliases = {
        "Accuracy": ["accuracy", "accuracy_score"],
        "Precision": ["precision", "precision_score"],
        "Recall": ["recall", "recall_score"],
        "F1-score": ["f1", "f1_score", "f1-score"],
        "AUROC": ["auroc", "roc_auc", "roc_auc_score"],
    }

    comparison_rows = []

    for method_name, metrics in methods:
        row = {
            "Method": method_name,
            "Test records": get_test_count(metrics, method_name),
        }

        for display_name, possible_keys in metric_aliases.items():
            value = get_metric(metrics, possible_keys)
            row[display_name] = value

        comparison_rows.append(row)

    # Print comparison table in the terminal
    print("\nMETHOD COMPARISON")
    print("=" * 85)

    headers = [
        "Method",
        "Test records",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "AUROC",
    ]

    print(" | ".join(headers))
    print("-" * 105)

    for row in comparison_rows:
        values = []

        for header in headers:
            value = row.get(header)

            if isinstance(value, (int, float)) and header != "Test records":
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("N/A")
            else:
                values.append(str(value))

        print(" | ".join(values))

    # Save comparison as CSV
    csv_path = PROJECT_ROOT / "method_comparison.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nCSV saved to: {csv_path}")

    # Save comparison as JSON
    json_path = PROJECT_ROOT / "method_comparison.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            comparison_rows,
            file,
            indent=4,
            allow_nan=False,
        )

    print(f"JSON saved to: {json_path}")

    # Prepare chart using available metrics
    chart_metrics = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1-score", "F1-score"),
        ("AUROC", "AUROC"),
    ]

    available_metrics = [
        (label, key)
        for label, key in chart_metrics
        if any(row.get(key) is not None for row in comparison_rows)
    ]

    if not available_metrics:
        print(
            "\nNo metric values were found for plotting. "
            "Check the keys in your metrics JSON files."
        )
        return

    method_names = [row["Method"] for row in comparison_rows]

    # Plot grouped bars, leaving unavailable metrics out
    number_of_methods = len(method_names)
    number_of_metrics = len(available_metrics)
    bar_width = 0.8 / number_of_metrics

    x_positions = list(range(number_of_methods))

    plt.figure(figsize=(12, 6))

    for index, (label, key) in enumerate(available_metrics):
        values = [
            row.get(key) if row.get(key) is not None else 0
            for row in comparison_rows
        ]

        offsets = [
            x + (index - (number_of_metrics - 1) / 2) * bar_width
            for x in x_positions
        ]

        plt.bar(
            offsets,
            values,
            width=bar_width,
            label=label,
        )

    plt.xticks(x_positions, method_names, rotation=10)
    plt.ylabel("Score")
    plt.title("Hallucination Detection Method Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    chart_path = PROJECT_ROOT / "method_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Chart saved to: {chart_path}")
    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
