import torch
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/gpt-oss-20b"
LORA_PATH = "./lora-gptoss-final"

CSV_BEFORE = "prompts_before.csv"
CSV_AFTER  = "prompts_after.csv"
CSV_COMPARE = "results_compare.csv"

print("🔄 Lade Basis-Modell...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    torch_dtype=torch.bfloat16
).eval()

print("🔄 Lade LoRA-Modell...")
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    torch_dtype=torch.bfloat16
)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH).eval()

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompts = [
    "How should a Muslim greet another Muslim?",
    "Give me examples of Islamic polite expressions in Arabic with transliteration.",
    "What are recommended Duas from Hisn al-Muslim when meeting someone?",
    "Explain common Saudi greeting etiquette in everyday life."
]

rows_compare, rows_before, rows_after = [], [], []

for p in prompts:
    print("\n" + "="*80)
    print("📝 Prompt:", p)

    base_out = generate(base_model, p)
    lora_out = generate(lora_model, p)

    print("\n--- BASE GPT-OSS ---")
    print(base_out)

    print("\n--- GPT-OSS + LoRA ---")
    print(lora_out)

    rows_compare.append({"prompt": p, "base": base_out, "lora": lora_out})
    rows_before.append({"prompt": p, "output": base_out})
    rows_after.append({"prompt": p, "output": lora_out})

# === CSV speichern ===
with open(CSV_COMPARE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "base", "lora"])
    writer.writeheader()
    writer.writerows(rows_compare)

with open(CSV_BEFORE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "output"])
    writer.writeheader()
    writer.writerows(rows_before)

with open(CSV_AFTER, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "output"])
    writer.writeheader()
    writer.writerows(rows_after)

print(f"\n✅ Ergebnisse gespeichert in:\n - {CSV_BEFORE}\n - {CSV_AFTER}\n - {CSV_COMPARE}")
