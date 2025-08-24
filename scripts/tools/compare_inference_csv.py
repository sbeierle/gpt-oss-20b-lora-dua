import torch
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/gpt-oss-20b"
LORA_PATH = "./lora-gptoss-final"
CSV_FILE = "results_compare.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# === Basis-Modell laden ===
print("🔄 Lade Basis-Modell...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

# === LoRA-Modell laden ===
print("🔄 Lade LoRA-Modell...")
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    torch_dtype=torch.bfloat16
)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH).to(device).eval()

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Prompts ===
prompts = [
    "How should a Muslim greet another Muslim?",
    "Give me examples of Islamic polite expressions in Arabic with transliteration.",
    "What are recommended Duas from Hisn al-Muslim when meeting someone?",
    "Explain common Saudi greeting etiquette in everyday life."
]

rows = []
for p in prompts:
    print("\n" + "="*80)
    print("📝 Prompt:", p)

    base_out = generate(base_model, p)
    lora_out = generate(lora_model, p)

    print("\n--- BASE GPT-OSS ---")
    print(base_out)

    print("\n--- GPT-OSS + LoRA ---")
    print(lora_out)

    rows.append({"prompt": p, "base": base_out, "lora": lora_out})

# === CSV speichern ===
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "base", "lora"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Ergebnisse gespeichert in {CSV_FILE}")
