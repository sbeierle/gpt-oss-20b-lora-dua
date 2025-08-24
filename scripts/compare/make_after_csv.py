# make_after_csv.py
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# === Konfiguration ===
BASE_MODEL = "./gpt-oss-20b"
LORA_PATH = "./lora-gptoss-final"   # dein LoRA Adapter
PROMPTS_FILE = "prompts_before.csv"
OUTPUT_FILE = "after.csv"

print("🔄 Lade Basis-Modell...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("🔄 Lade LoRA Adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# === Prompts einlesen ===
with open(PROMPTS_FILE, "r") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

print(f"📋 {len(prompts)} Prompts geladen")

# === Antworten generieren & speichern ===
rows = []
for i, prompt in enumerate(prompts, 1):
    print(f"\n📝 Prompt {i}: {prompt}")
    out = pipe(prompt)[0]["generated_text"]
    print(f"💡 Antwort: {out}\n")
    rows.append({"prompt": prompt, "after": out})

# === CSV schreiben ===
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "after"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Ergebnisse gespeichert in {OUTPUT_FILE}")
