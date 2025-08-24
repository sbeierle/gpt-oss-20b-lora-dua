from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# 🔄 Modell & Tokenizer
model_name = "/workspace/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ⚙️ Modell laden (bf16 – ohne bitsandbytes!)
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

# 📂 Dataset (jetzt das volle!)
data = load_dataset("json", data_files="/workspace/data_training_full.jsonl")

def tokenize(example):
    text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# 👉 Map anwenden
tokenized = data["train"].map(tokenize)

# ⚙️ Training Args (FULL Training)
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,       # 👈 volle 3 Epochen
    learning_rate=2e-4,
    logging_steps=10,
    output_dir="/workspace/lora-output-full",
    save_strategy="epoch",    # nach jeder Epoche speichern
    bf16=True,
    optim="adamw_torch"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()

# 💾 LoRA Adapter speichern
model.save_pretrained("/workspace/lora-output-full")
tokenizer.save_pretrained("/workspace/lora-output-full")
print("✅ Full Training abgeschlossen & LoRA in /workspace/lora-output-full gespeichert")
