
import json
from pathlib import Path
from collections import Counter

# Find the project and data folder
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

source_path = data_dir / "source_info.jsonl"

# Support either response filename
response_path = data_dir / "response.jsonl"
if not response_path.exists():
    response_path = data_dir / "response(1).jsonl"

if not source_path.exists():
    raise FileNotFoundError(f"Missing file: {source_path}")

if not response_path.exists():
    raise FileNotFoundError(
        "Could not find response.jsonl or response(1).jsonl"
    )


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


# Load both files
sources = load_jsonl(source_path)
responses = load_jsonl(response_path)

# Match each source to its source_id
source_lookup = {
    str(item["source_id"]): item
    for item in sources
}

matched = [
    item for item in responses
    if str(item["source_id"]) in source_lookup
]

# Count hallucination annotations at response level.
# A response is positive if it has one or more annotated spans.
def has_hallucination_label(item):
    labels = item.get("labels", [])
    if isinstance(labels, str):
        labels = json.loads(labels)
    return len(labels) > 0


print("RAGTRUTH DATASET AUDIT")
print("=" * 35)
print("Source records:", len(sources))
print("Response records:", len(responses))
print("Responses matched to sources:", len(matched))
print("Unmatched responses:", len(responses) - len(matched))

print("\nRESPONSE SPLITS")
print(Counter(item.get("split") for item in responses))

print("\nRESPONSE QUALITY")
print(Counter(item.get("quality") for item in responses))

print("\nANNOTATION DISTRIBUTION BY SPLIT")
for split in sorted(set(item.get("split") for item in responses)):
    subset = [
        item for item in responses
        if item.get("split") == split
    ]

    positive = sum(has_hallucination_label(item) for item in subset)
    negative = len(subset) - positive

    print(f"\n{split.upper()}")
    print("Total responses:", len(subset))
    print("With hallucination spans:", positive)
    print("Without hallucination spans:", negative)

print("\nEXAMPLE MATCHED RECORD")
if matched:
    item = matched[0]
    source = source_lookup[str(item["source_id"])]

    print("Response ID:", item["id"])
    print("Source ID:", item["source_id"])
    print("Split:", item["split"])
    print("Quality:", item["quality"])
    print("Has annotated spans:", has_hallucination_label(item))
    print("Source text:", source["source_info"][:250])
    print("Generated response:", item["response"][:250])
    print("Annotations:", str(item["labels"])[:500])
