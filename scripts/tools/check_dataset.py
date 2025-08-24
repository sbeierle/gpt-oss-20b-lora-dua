import json

file_path = "train_data.jsonl"

print("🔎 Checking dataset:", file_path)

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
except FileNotFoundError:
    print("❌ Datei nicht gefunden! Stelle sicher, dass train_data.jsonl im selben Ordner liegt.")
    exit()

print(f"📂 Gesamtanzahl Zeilen: {len(lines)}\n")

# Zeige die ersten 5 Beispiele
for i, line in enumerate(lines[:5]):
    try:
        ex = json.loads(line.strip())
        instr = ex.get("instruction", "❌ fehlt")
        out = ex.get("output", "❌ fehlt")
        print(f"📝 Beispiel {i+1}:")
        print(f"   Instruction: {instr}")
        print(f"   Output: {out}\n")
    except Exception as e:
        print(f"⚠️ Fehler in Zeile {i+1}: {e}")
        print("Raw:", line.strip(), "\n")
