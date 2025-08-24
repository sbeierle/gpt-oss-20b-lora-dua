import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/gpt-oss-20b"
LORA_MODEL = "./lora-gptoss-final"

# === 1. Modell + Tokenizer laden ===
print("🔄 Lade Basis-Modell...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

print("🔄 Lade LoRA Adapter...")
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()

# === 2. Prompts definieren ===
prompts = [
    "How should a Muslim greet another Muslim?",
    "What is the meaning of 'As-salamu alaykum'?",
    "Give me examples of Islamic polite expressions in Arabic with transliteration.",
    "What are recommended Duas from Hisn al-Muslim when meeting someone?",
    "Explain common Saudi greeting etiquette in everyday life."
]

# === 3. Generate Funktion ===
def generate_response(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === 4. Run ===
print("\n🚀 Starte Inferenz mit LoRA...\n")
for i, prompt in enumerate(prompts, start=1):
    print(f"📝 Prompt {i}: {prompt}")
    response = generate_response(prompt)
    print(f"💡 Antwort:\n{response}\n{'-'*60}\n")
