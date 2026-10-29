import json
from pathlib import Path
from collections import Counter

project_dir = Path(__file__).resolve().parent.parent
path = project_dir / "data" / "general_data.json"

print("Reading file:", path)

with path.open("r", encoding="utf-8-sig") as f:
    content = f.read()

try:
    data = json.loads(content)
    if isinstance(data, dict):
        data = list(data.values())
except json.JSONDecodeError:
    data = [
        json.loads(line)
        for line in content.splitlines()
        if line.strip()
    ]

print("Total records loaded:", len(data))

labels = [
    str(row.get("hallucination", "")).strip().lower()
    for row in data
]

counts = Counter(labels)

print("\nFULL LABEL DISTRIBUTION")
for label, count in counts.items():
    percentage = count / len(data) * 100
    print(f"{label}: {count} ({percentage:.2f}%)")

print("\nFIRST 10 LABELS")
for row in data[:10]:
    print(row.get("ID"), row.get("hallucination"))