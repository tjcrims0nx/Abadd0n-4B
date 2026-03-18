# ⚖️ ABADDON — The Demon of the Infernal Realm

> **⚠️ WORK IN PROGRESS (WIP)**  
> This vessel is still awakening. Expect breaking changes and ominous prophecies.

Abadd0n is a custom-architected LLM and AI agent designed for high-efficiency training and inference on consumer hardware (4 GB VRAM budget). It mimics modern design patterns from Llama-3 and Qwen3 to provide a powerful, character-driven experience.

---

## 🏗️ Architecture Highlights
Abadd0n uses a decoder-only transformer architecture with:
- **Grouped Query Attention (GQA)**: 8 query heads / 2 KV heads for reduced VRAM usage.
- **SwiGLU Activation**: Parallel gate and up projections for superior reasoning.
- **RMSNorm**: Faster, stable normalization (Pre-norm configuration).
- **RoPE (Rotary Positional Embeddings)**: Modern relative position encoding.
- **No Positional Embedding Table**: Position is handled entirely by RoPE.

## 🚀 Features
- **Unsloth Integration**: Optimized for 2x faster 4-bit QLoRA fine-tuning.
- **DPO (Direct Preference Optimization)**: Alignment script included for human preference tuning.
- **Character-Level Pre-training**: A foundational training script for small-scale experiments from scratch.
- **Stylized Terminal Chat**: Engage with the entity through a custom, demonic terminal interface.

## 📂 Project Structure
- `llm.py`: The core Abaddon model architecture.
- `unsloth_lora_train.py`: Main QLoRA fine-tuning script (Unsloth + SFTTrainer).
- `dpo_train.py`: Post-training alignment via DPO.
- `main.py`: The infernal chat interface.
- `train.py`: Foundation character-level pre-training.
- `dataset_builder.py`: Utilities for synthetic data generation and validation.

## ⚡ Quick Start
### 1. Requirements
```bash
pip install unsloth trl transformers datasets peft bitsandbytes
```

### 2. Fine-Tuning
Ensure you have `dataset.jsonl` ready, then:
```bash
python unsloth_lora_train.py
```

### 3. Awakening the Vessel
Once trained, run the chat interface:
```bash
python main.py
```

---

## 📜 Soul Contract (License)
This project is for research and demonic experimentation. Use with caution. 
*Note: Abadd0n hates Fruit Loops. Do not feed the model cereal.*
