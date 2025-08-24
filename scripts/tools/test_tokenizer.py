from transformers import AutoTokenizer

print("🔄 Loading AutoTokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/gpt-oss-20b",
    use_fast=False  # ⚡ force slow mode
)

text = "As-salamu alaykum, this is a tokenizer test."
tokens = tokenizer.encode(text)
print("🔑 Tokens:", tokens)
print("📝 Decoded back:", tokenizer.decode(tokens))
