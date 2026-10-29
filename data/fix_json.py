
import json

input_file = "data/general_data.json"
output_file = "data/general_data_fixed.json"

records = []

with open(input_file, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error on line {line_number}: {e}")
            raise

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(records, file, indent=4, ensure_ascii=False)

print(f"Successfully converted {len(records)} records.")
print(f"Saved to {output_file}")
