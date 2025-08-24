#!/bin/bash
# ============================================================
# Restore Environment for GPT-OSS 20B + LoRA Training (B200)
# ============================================================

set -e

echo "🔄 Setting up environment..."

# 1. Update pip
pip install --upgrade pip

# 2. Install PyTorch Nightly (CUDA 12.1, B200 support sm_100)
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# 3. Core dependencies
pip install transformers==4.56.0.dev0
pip install bitsandbytes==0.45.0
pip install peft accelerate datasets triton

# 4. Write requirements.txt freeze
cat > /workspace/requirements.txt << 'EOF'
torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu121
transformers==4.56.0.dev0
bitsandbytes==0.45.0
peft
datasets
accelerate
triton
EOF

# 5. Training Script
cat > /workspace/train_lora.py << 'EOF'
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "/workspace/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=bnb_config
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

data = load_dataset("json", data_files="/workspace/train_data.jsonl")

def tokenize(example):
    text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = data["train"].map(tokenize)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=5,
    output_dir="/workspace/lora-output",
    save_strategy="steps",
    save_steps=50,
    bf16=True
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()
model.save_pretrained("/workspace/lora-output")
EOF

# 6. Inference Script
cat > /workspace/run_inference_lora.py << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/gpt-oss-20b"
LORA_PATH = "/workspace/lora-output"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, LORA_PATH)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompts = [
    "How should a Muslim greet another Muslim?",
    "Give me examples of Islamic polite expressions in Arabic with transliteration.",
    "What are recommended Duas from Hisn al-Muslim when meeting someone?",
    "Explain common Saudi greeting etiquette in everyday life."
]

for p in prompts:
    print("\\n📝 Prompt:", p)
    print("💡 Antwort:", generate(p))
EOF

echo "✅ Environment restored. Ready for training!"
echo "👉 Next steps:"
echo "   1. Put your dataset into /workspace/train_data.jsonl"
echo "   2. Run: python /workspace/train_lora.py"
echo "   3. Test: python /workspace/run_inference_lora.py"
