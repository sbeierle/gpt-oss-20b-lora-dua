from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# 📂 Basis Modell & Tokenizer
base_model = "/workspace/gpt-oss-20b"
lora_path = "/workspace/lora-output-full"

print("🔄 Lade Modell + LoRA Adapter...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 🔧 LoRA Adapter drauf
model = PeftModel.from_pretrained(model, lora_path)

# ⚡ Inference-Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 🧪 Beispiel-Prompts
prompts = [
    "What is the dua when entering the market?",
    "Give me the dua when going to bed while having worries.",
    "Which dua should one say when waking up frightened from a dream?",
    "List the three surahs to recite in the evening."
]

print("\n================= TEST OUTPUTS =================\n")
for p in prompts:
    print(f"📝 Prompt: {p}")
    out = pipe(
        p,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )[0]["generated_text"]
    print(f"💡 Response:\n{out}\n")
    print("-" * 60)

print("\n✅ Test abgeschlossen — LoRA full inference ready!\n")
