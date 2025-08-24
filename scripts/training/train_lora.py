from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# 🔄 Modell & Tokenizer
model_name = "/workspace/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ⚙️ Modell laden (ohne bitsandbytes!)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16   # oder torch.float16, beide gehen auf H100/A100
)

# 🔧 LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 📂 Dataset
data = load_dataset("json", data_files="/workspace/train_data.jsonl")

def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = data["train"].map(tokenize)

# ⚙️ Training Args
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,        # Showcase Run
    learning_rate=2e-4,
    logging_steps=5,
    output_dir="/workspace/lora-output",
    save_strategy="steps",
    save_steps=50,
    bf16=True,            # nutzt BF16 (H100/A100 = top)
    optim="adamw_torch"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()

# 💾 LoRA Adapter speichern
model.save_pretrained("/workspace/lora-output")
tokenizer.save_pretrained("/workspace/lora-output")
