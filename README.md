# 🕌 LoRA Training Showcase – GPT-OSS-20B + Custom Dua Dataset

## 🚀 Overview
This project documents a **showcase fine-tuning experiment** of the **GPT-OSS-20B model** with **LoRA** on a **custom Duʿāʾ dataset (inspired by Ḥiṣn al-Muslim)**.  

👉 Focus: **Arabic language, authentic Islamic supplications**  
👉 Uniqueness: Built entirely from scratch – dataset prep, debugging, training, inference, and visualization  
👉 Goal: Provide a **transparent research-style workflow** that others can replicate or extend  

⚠️ **Note:** Final model weights are not released.  
This repo serves as a **technical documentation & showcase**.

---

## ⏱️ Project Facts
- **Total time spent:** ~12–14h  
  - Debugging: ~4h (dataset fixes, rsync issues, GPU not used at first 😅)  
  - Training + Inference: ~6–8h  
  - Misc (setup, cleanup): ~2h  
- **Hardware:** AMD RX 7900 XTX (ROCm backend)  
- **Frameworks:** HuggingFace Transformers, PEFT, PyTorch (ROCm), custom Python scripts  
- **Specialty:** OSS-20B with LoRA → **rarely documented on AMD GPUs**  

---

## 📂 Repository Structure
```tree
gpt-oss-20b-lora-dua/
├── datasets/          # Training data (JSONL, CSV, tokenizer)
├── results/           # Inference results & comparisons
├── images/            # Screenshots & debug visuals
├── videos/            # Training & inference demos (via Git LFS)
├── scripts/           # Organized experiment scripts
│   ├── training/      # Training pipelines
│   ├── inference/     # Inference tests
│   ├── dataset_tools/ # Dataset checks & fixes
│   ├── compare/       # Compare runs & Gradio UI
│   └── tools/         # Utilities & helpers
└── utils/             # Environment configs & scanners
```

## 🛠️ Workflow

### 1. Dataset Preparation
- Base dataset curated from **Ḥiṣn al-Muslim Duʿāʾ**  
- Fixes applied using:
  - `fix_training_entry.py`
  - `check_dataset.py`
  - `convert_json_to_jsonl.py`

### 2. Debugging Phase
- **Issue:** GPU not used (ran on CPU by mistake 😅)  
- **Fix:** Verified ROCm setup, ensured `torch.cuda.is_available() = True`  
- Extra ~1h wasted on **rsync retries** – included here to show real-world overhead  

### 3. Mini-Test
- Ran LoRA on ~100 samples  
- Verified that adapters trained & merged properly  
- ✅ Confirmed inference pipeline working  

### 4. Full Training
- Trained on full dua dataset (`datasets/knigge_dua_dataset.jsonl`)  
- Saved LoRA adapters & merged back into base  

### 5. Merge & Export
- Used `merge_lora.py` to combine base + adapters  
- Exported in multiple quantized formats (**Q4, Q5, Q8**) locally  
- Files intentionally **not pushed to GitHub** (too large)  

### 6. Inference Showcase
- Tested with authentic Duʿāʾ prompts  
- Model produced **Arabic text, transliteration, and partial English gloss**  
- Outputs documented in `results/` and via screenshots in `images/`  

---

## 📊 Results

### ✅ Successes
- First documented LoRA fine-tune of **OSS-20B** on **AMD ROCm GPU**  
- Dataset correction pipeline works robustly  
- Training reproducible (mini + full runs)  
- Model improved on Arabic + Islamic contexts  

### ⚠️ Limitations
- Dataset small (~100–200 examples)  
- Religious accuracy still requires scholar review  
- ROCm quirks → some wasted time (CPU fallback, rsync overhead)  

---

## 📷 Media

## 📷 Media

### Screenshots
- **GPU Setup:** ![](images/b200_Cuda12.png)  
- **Mini-Test Success:** ![](images/mini_training_success.png)  
- **Inference Example:** ![](images/full_compare1.png)  
- **Dataset Fix:** ![](images/Test_run_fix_success.png)  

### Videos (via LFS in `/videos/`)
- **Training runs (mini + full):**  
  - [Mini-Test](videos/b200_training_MINITEST.mp4)  
  - [Full Training](videos/b200_training_FULL.mp4)  

- **Debugging sessions:**  
  - [Dataset Fix](videos/b200_FIX_data_entry-2025-08-20_13.41.22.mp4)  
  - [Merge Inference](videos/b200_MERGE_INFERENCE_LAST.mp4)  

- **Inference showcases:**  
  - [5 Prompts Before/After](videos/b200_5promptstraining_before_after.mp4)  
  - [Inference Run](videos/b200_INFERENCE_Test.mp4)  
  - [Quantization Showcase](videos/B200_MXFP4_quantization_6prompts_after.mp4)  


---

## 💡 Lessons Learned
- **AMD ROCm** works well but requires careful setup  
- **LoRA is efficient** even on 20B parameter models  
- Debugging + real-world overhead (CPU fallback, rsync) matter just as much as training itself  
- Transparency (keeping even “mistakes”) helps others learn  

---

## 🎯 Conclusion
This repo demonstrates:  
- How to structure a **real LoRA fine-tune project** end-to-end  
- How to handle **dataset debugging, training, merging, inference**  
- How to use **AMD GPUs (ROCm)** for large-scale experiments  

👉 A **hands-on showcase**, not a polished prod
