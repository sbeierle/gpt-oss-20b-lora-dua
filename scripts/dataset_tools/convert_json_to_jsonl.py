import json

# Input (dein Array)
with open("data_training_last.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)

# Output (echtes JSONL)
with open("data_training_last_fixed.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Converted {len(data)} entries -> data_training_last_fixed.jsonl")
