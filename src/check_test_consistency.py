
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NLI_FILE = ROOT / "nli_test_results.csv"
OVERLAP_FILE = ROOT / "overlap_test_results.csv"


def read_ids(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(
                f"'id' column not found in {path.name}. "
                f"Available columns: {reader.fieldnames}"
            )

        return {
            row["id"].strip()
            for row in reader
            if row.get("id", "").strip()
        }


def main():
    print("Checking test-record consistency...\n")

    nli_ids = read_ids(NLI_FILE)
    overlap_ids = read_ids(OVERLAP_FILE)

    print("NLI unique response IDs:", len(nli_ids))
    print("Overlap unique response IDs:", len(overlap_ids))

    only_nli = nli_ids - overlap_ids
    only_overlap = overlap_ids - nli_ids

    print("IDs only in NLI:", len(only_nli))
    print("IDs only in overlap:", len(only_overlap))

    if nli_ids == overlap_ids:
        print("\nPASS: Both methods evaluated the same response IDs.")
    else:
        print("\nWARNING: The evaluated response IDs differ.")

        if only_nli:
            print("Example IDs only in NLI:", list(only_nli)[:5])

        if only_overlap:
            print(
                "Example IDs only in overlap:",
                list(only_overlap)[:5],
            )


if __name__ == "__main__":
    main()
