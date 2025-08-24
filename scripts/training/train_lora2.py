from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# === Modell & Tokenizer ===
model_name = "/workspace/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # ✅ slow tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)

# === LoRA Config ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# === Dataset laden ===
data = load_dataset("json", data_files="/workspace/data2_training.jsonl")

def tokenize(example):
    text = f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = data["train"].map(tokenize, remove_columns=data["train"].column_names)

# === Training Args ===
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=2,
    output_dir="/workspace/lora-output2",
    save_strategy="epoch",
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

# === Training starten ===
trainer.train()

# === Speichern ===
model.save_pretrained("/workspace/lora-output2")
tokenizer.save_pretrained("/workspace/lora-output2")
