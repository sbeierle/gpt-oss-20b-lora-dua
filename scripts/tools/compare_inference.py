import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/gpt-oss-20b"
LORA_PATH = "./lora-gptoss-final"

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

for p in prompts:
    print("\n" + "="*80)
    print("📝 Prompt:", p)

    print("\n--- BASE GPT-OSS ---")
    print(generate(base_model, p))

    print("\n--- GPT-OSS + LoRA ---")
    print(generate(lora_model, p))
