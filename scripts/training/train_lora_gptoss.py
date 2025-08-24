import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# === 1. Modell-ID anpassen (GPT-OSS Pfad) ===
MODEL_ID = "/workspace/gpt-oss-20b"

# === 2. Tokenizer & Modell laden ===
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    local_files_only=True
)

# === 3. LoRA Config ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === 4. Dataset laden (Tiny dataset nur zum Testen) ===
dataset = load_dataset("Abirate/english_quotes", split="train[:1%]")  # klein & schnell

def tokenize(batch):
    tokens = tokenizer(
        batch["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # wichtig für Loss
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# === 5. Training Args ===
training_args = TrainingArguments(
    output_dir="./lora-gptoss",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none"
)

# === 6. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# === 7. Train starten ===
trainer.train()

# === 8. Fertige LoRA speichern ===
model.save_pretrained("./lora-gptoss-final")
tokenizer.save_pretrained("./lora-gptoss-final")
print("✅ Training abgeschlossen & LoRA gespeichert in ./lora-gptoss-final")
