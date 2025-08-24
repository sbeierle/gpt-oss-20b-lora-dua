import json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/gpt-oss-20b"
LORA_PATH = "/workspace/lora-output"
DATA_FILE = "/workspace/train_data.jsonl"

print("🔄 Lade Base Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda")

print("🔄 Lade LoRA Adapter...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

def generate(model, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Lade Dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

# Ziehe 10 zufällige Beispiele
samples = random.sample(dataset, 10)

# Vergleich laufen lassen
results = []
for ex in samples:
    instr = ex["instruction"]
    base_out = generate(base_model, instr)
    lora_out = generate(lora_model, instr)
    results.append({"instruction": instr, "base": base_out, "lora": lora_out})

# CSV speichern
import csv
with open("results_compare.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["instruction", "base", "lora"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Vergleich abgeschlossen. Ergebnisse in results_compare.csv gespeichert.")
