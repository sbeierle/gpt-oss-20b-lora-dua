# train_lora_mxfp4.py
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import torch

# === Config ===
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # Beispielmodell mit MXFP4
OUTPUT_DIR = "./lora-mxfp4"
DATASET = "yahma/alpaca-cleaned"  # Demo Dataset

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# === Model (mit MXFP4) ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto"   # wichtig: MXFP4 wird so automatisch geladen
)

# === LoRA Config ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typische LoRA-Ziele
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === Dataset laden ===
dataset = load_dataset(DATASET)

def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

tokenized = dataset.map(tokenize, batched=True)

# === TrainingArguments ===
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,  # Demo → klein halten
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=50
)

# === Trainer ===
from transformers import Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=tokenized["train"],
    args=args,
    data_collator=data_collator
)

# === Train ===
trainer.train()

# === Save LoRA ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
