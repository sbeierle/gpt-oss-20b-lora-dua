from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 🔄 Modell & Tokenizer
model_name = "/workspace/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ⚙️ Quantization
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# 🔧 LoRA Config
lora_config = LoraConfig(
    r=8,   # kleiner für Test
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 📂 Dataset (nur 20 Beispiele für Test)
data = load_dataset("json", data_files="/workspace/train_data.jsonl")
small_train = data["train"].select(range(min(20, len(data["train"]))))

def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=128)

tokenized = small_train.map(tokenize)

# ⚙️ Training Args (Mini-Test)
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=1,   # nur 1 Step für Check
    learning_rate=5e-4,
    logging_steps=1,
    output_dir="/workspace/lora-test-output",
    save_strategy="no",
    bf16=True
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)

print("🚀 Starte Mini-Test Training...")
trainer.train()
model.save_pretrained("/workspace/lora-test-output")
print("✅ Mini-Test abgeschlossen. Check /workspace/lora-test-output")
