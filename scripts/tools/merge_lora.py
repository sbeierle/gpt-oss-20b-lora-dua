from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model_path = "/workspace/gpt-oss-20b"
lora_model_path = "/workspace/lora-gptoss-final"
output_path = "/workspace/merged_model"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")

print("Applying LoRA...")
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(output_path)
print("✅ Done: merged model saved to", output_path)
