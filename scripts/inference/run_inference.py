import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Paths ===
MODEL_PATH = "/workspace/gpt-oss-20b"
OUTFILE = "before.csv"

print("🔄 Lade Modell & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# === Robust Loader mit Fallback ===
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✅ Modell in bfloat16 geladen")
except Exception as e:
    print(f"⚠️ bfloat16 fehlgeschlagen: {e}")
    print("👉 Versuche float16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Modell in float16 geladen")

# === Prompts laden ===
with open("prompts_before.csv") as f:
    reader = csv.DictReader(f)
    prompts = list(reader)

# === Ergebnisse schreiben ===
with open(OUTFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "prompt", "response"])

    for row in prompts:
        pid, prompt = row["id"], row["prompt"]
        print(f"\n📝 Prompt {pid}: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float else v for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        writer.writerow([pid, prompt, response])
        print(f"💡 Antwort: {response[:150]}...")
