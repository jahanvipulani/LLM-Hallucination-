
import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "general_data.json"

print("File exists:", path.exists())

# First, try reading it as a standard JSON file.
try:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

except json.JSONDecodeError:
    # If it contains one JSON object per line, read it as JSONL.
    with path.open("r", encoding="utf-8-sig") as f:
        data = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

print("Data type:", type(data).__name__)

if isinstance(data, list):
    print("Record count:", len(data))

    if data and isinstance(data[0], dict):
        print("Field names:", list(data[0].keys()))

        # Print short values only; don't print full text.
        for key, value in data[0].items():
            if isinstance(value, (bool, int, float)) or value is None:
                print(f"{key} = {value}")
            elif isinstance(value, str) and len(value) < 80:
                print(f"{key} = {value!r}")

elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))

print("Inspection complete.")
