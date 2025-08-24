import json

in_file = "data_training_last.jsonl"
out_file = "data_training_last_fixed.jsonl"

with open(in_file, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

fixed = []

# Wir nehmen immer 2 Zeilen: prompt + response
for i in range(0, len(lines), 2):
    try:
        line1 = lines[i]
        line2 = lines[i + 1]

        # Prompt-Zeile: alles vor Komma
        prompt_part = line1.rstrip(", ")
        # Response-Zeile: ganz normal
        response_part = line2

        # Zusammensetzen zu einem JSON
        obj_str = prompt_part + "," + response_part
        obj = json.loads(obj_str)

        # Keys angleichen
        if "prompt" in obj:
            obj["instruction"] = obj.pop("prompt")
        if "response" in obj:
            obj["output"] = obj.pop("response")

        fixed.append(obj)

    except Exception as e:
        print(f"⚠️ Fehler bei Block {i}//{i+1}:", e)
        continue

with open(out_file, "w", encoding="utf-8") as f:
    for obj in fixed:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Gefixt! Neue Datei: {out_file} mit {len(fixed)} Beispielen")
