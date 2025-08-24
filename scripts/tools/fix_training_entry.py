import json

# 📂 Input/Output Files
input_file = "data_training_full.jsonl"
output_file = "data_training_full_fixed.jsonl"

# 🎯 Target Prompt to fix
target_prompt = "List the three surahs to recite in the evening."

# 🛠 Process line by line
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        entry = json.loads(line)
        if entry["prompt"].strip() == target_prompt:
            entry["response"] = (
                "Arabic: قُلْ هُوَ اللَّهُ أَحَدٌ، قُلْ أَعُوذُ بِرَبِّ الْفَلَقِ، قُلْ أَعُوذُ بِرَبِّ النَّاسِ.\n\n"
                "Transliteration: Qul huwa Allāhu aḥad, Qul aʿūdhu birabbil-falaq, Qul aʿūdhu birabbin-nās.\n\n"
                "English: Surah al-Ikhlās, Surah al-Falaq, and Surah an-Nās."
            )
            print("✅ Corrected entry:", entry)
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"🎉 Done! Fixed dataset written to {output_file}")
