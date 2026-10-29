
import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

for filename in ["general_data.json", "source_info.jsonl"]:
    path = data_dir / filename

    print("\n" + "=" * 50)
    print("FILE:", filename)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    try:
        if filename.endswith(".jsonl"):
            with path.open("r", encoding="utf-8-sig") as f:
                records = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]
        else:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Show the structure without assuming the records' location.
                print("Top-level keys:", list(data.keys()))
                records = None
            else:
                print("Unexpected format:", type(data).__name__)
                records = None

        if records is None:
            continue

        print("Record count:", len(records))

        if records and isinstance(records[0], dict):
            print("Field names:", list(records[0].keys()))

            # Print values only for short, non-text fields.
            for key, value in records[0].items():
                if isinstance(value, (bool, int, float)) or value is None:
                    print(f"  {key} = {value}")
                elif isinstance(value, str) and len(value) < 60:
                    print(f"  {key} = {value!r}")

    except Exception as e:
        print("ERROR:", repr(e))

print("\nSummary complete.")
