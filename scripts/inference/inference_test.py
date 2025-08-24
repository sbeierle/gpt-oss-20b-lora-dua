from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# 📂 Ordner mit dem trainierten LoRA
lora_path = "/workspace/lora-output-full"

# 🔄 Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=True)

# ⚙️ Basismodell + LoRA laden
base_model = "/workspace/gpt-oss-20b"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, lora_path)

# 🚀 Pipeline bauen
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 📝 Beispiel Prompt
prompt = """### Instruction:
Gib mir die bekannte Dua beim Eintritt in den Markt.

### Response:"""

outputs = pipe(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

print("=== Antwort ===")
print(outputs[0]["generated_text"])
