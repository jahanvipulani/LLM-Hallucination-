
import json
from pathlib import Path

# Project and dataset locations
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def inspect_file(path):
    print("\n" + "=" * 60)
    print("FILE:", path.name)
    print("=" * 60)

    try:
        if path.suffix.lower() == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8-sig") as file:
                for line in file:
                    if line.strip():
                        records.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)

            records = data if isinstance(data, list) else [data]

        print("Number of records:", len(records))

        if records:
            first = records[0]

            if isinstance(first, dict):
                print("Fields:", list(first.keys()))
                print("\nFirst record preview:")

                for key, value in first.items():
                    preview = repr(value)
                    if len(preview) > 180:
                        preview = preview[:180] + "..."
                    print(f"{key}: {preview}")
            else:
                print("First record:", repr(first)[:300])

    except Exception as error:
        print("Error reading file:", error)


def main():
    print("Dataset inspection")
    print("Data folder:", DATA_DIR)

    if not DATA_DIR.exists():
        print("ERROR: Create the data folder first.")
        return

    files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in (".json", ".jsonl")
    ]

    if not files:
        print("No JSON or JSONL files found in data.")
        return

    for path in sorted(files):
        inspect_file(path)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
