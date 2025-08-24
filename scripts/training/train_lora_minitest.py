from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# 🔄 Modell & Tokenizer
model_name = "/workspace/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ⚙️ Modell laden (nur bf16 – ohne bitsandbytes!)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
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
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()  # <<<< WICHTIG für Loss
    return tokens

# 👉 Map anwenden
tokenized = data["train"].map(tokenize)

# ⚙️ Training Args (Mini-Test)
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_steps=5,          # 👈 Nur 5 Steps für Testlauf
    learning_rate=2e-4,
    logging_steps=1,
    output_dir="/workspace/lora-output-test",
    save_strategy="no",
    bf16=True,
    optim="adamw_torch"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()

# 💾 LoRA Adapter speichern
model.save_pretrained("/workspace/lora-output-test")
tokenizer.save_pretrained("/workspace/lora-output-test")
print("✅ Mini-Test abgeschlossen & LoRA in /workspace/lora-output-test gespeichert")
