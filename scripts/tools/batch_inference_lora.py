import json
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# ⚙️ Pfade anpassen
base_model = "/workspace/gpt-oss-20b"
lora_path = "/workspace/lora-output-full"
test_prompts = "/workspace/test_prompts.jsonl"
output_csv = "/workspace/inference_results.csv"

# 🔄 Tokenizer & Modell laden
print("📥 Lade Modell + LoRA Adapter...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, lora_path)

# 🔧 Pipeline vorbereiten
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
   
    torch_dtype=torch.bfloat16
)

# 📂 Test-Prompts laden
prompts = []
with open(test_prompts, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if "instruction" in obj:
            prompts.append(obj["instruction"])

# 🚀 Antworten generieren
results = []
print(f"⚡ Running inference on {len(prompts)} prompts...")
for prompt in prompts:
    output = pipe(
        f"### Instruction:\n{prompt}\n\n### Response:\n",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )[0]["generated_text"]

    # Extrahiere nur die Response
    if "### Response:" in output:
        response = output.split("### Response:")[-1].strip()
    else:
        response = output.strip()

    results.append({"prompt": prompt, "response": response})
    print(f"📝 Prompt: {prompt}\n💡 Response: {response}\n{'-'*50}")

# 💾 Ergebnisse speichern
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Inference abgeschlossen – Ergebnisse in {output_csv}")
