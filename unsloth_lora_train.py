"""
Abadd0n — Unsloth QLoRA Fine-Tuning Script
════════════════════════════════════════════
Target:  4 GB VRAM GPU (e.g. RTX 3050 4 GB, GTX 1650 4 GB)
Model:   Qwen3-0.6B or Llama-3-1B-Instruct (both <2 GB in 4bit)
Method:  QLoRA via Unsloth + SFTTrainer (instruction tuning, chat template)
Dataset:  dataset.jsonl — Alpaca or ChatML rows; includes multilingual coding examples (see dataset_builder.py).
Output:  LoRA adapters  →  optionally merged & exported to GGUF for Ollama

Usage (activate venv first):
    venv_win\\Scripts\\activate   # Windows
    source venv_wsl/bin/activate  # WSL
    python unsloth_lora_train.py
"""

import sys
from pathlib import Path

# Must run in venv_win or venv_wsl (ensures correct LoRA/training env)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import venv_check
venv_check.require_abaddon_venv()

import torch
import pre_unsloth
pre_unsloth.before_import()

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    # ── Unsloth MUST be imported first ──────────────────────────────────────────
    import pre_unsloth

    pre_unsloth.before_import()
    import unsloth  # noqa: F401  Must be first for kernel patches
    
    import os
    import json
    import torch
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    # ══════════════════════════════════════════════════════════════
    #  ① CONFIGURATION — tuned for 4 GB VRAM
    # ══════════════════════════════════════════════════════════════
    # Model choices (ordered by VRAM friendliness for 4 GB):
    #   "unsloth/Qwen3-0.6B-bnb-4bit"          ← 0.6B  ≈ 0.7 GB (best for 4 GB)
    #   "unsloth/Qwen3-1.7B-bnb-4bit"          ← 1.7B  ≈ 1.4 GB
    #   "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" ← 1B  ≈ 0.9 GB
    #   "unsloth/llama-3-8b-Instruct-bnb-4bit" ← 8B   ≈ 5 GB (too big for 4 GB)
    MODEL_NAME      = "unsloth/Qwen3-0.6B-bnb-4bit"   # VRAM-friendly default
    DATASET_PATH    = "dataset.jsonl"
    OUTPUT_DIR      = "outputs/abadd0n_lora"
    LORA_OUTPUT_DIR = "lora_model"
    MAX_SEQ_LENGTH  = 2048   # Qwen3 native context; reduce to 1024 if OOM
    DTYPE           = None   # None = auto-detect (bf16 on Ampere+, fp16 otherwise)
    LOAD_IN_4BIT    = True   # 4-bit NF4 quantisation — mandatory for 4 GB VRAM
    
    # LoRA adapter configuration
    LORA_R           = 8    # Rank  — 8 is a good 4 GB sweet-spot (use 16 if VRAM allows)
    LORA_ALPHA       = 16   # alpha = 2 × r  gives effective LR scaling of 1.0
    LORA_DROPOUT     = 0.0  # Unsloth optimises dropout to 0
    LORA_TARGET_MODS = [    # Modules that receive LoRA adapters
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    # Training hyperparameters (4 GB VRAM budget)
    PER_DEVICE_BATCH  = 1    # Must be 1 for 4 GB
    GRAD_ACCUM_STEPS  = 16   # Effective batch = 16 (simulates batch_size=16)
    WARMUP_RATIO      = 0.05
    NUM_EPOCHS        = 3    # Set to 1 for a quick smoke-test
    LEARNING_RATE     = 2e-4
    LR_SCHEDULER      = "cosine"     # cosine decay works well for SFT
    WEIGHT_DECAY      = 0.01
    OPTIM             = "adamw_8bit"  # 8-bit Adam — saves ~50 % optimiser VRAM
    MAX_STEPS         = -1            # -1 = run full epochs; set e.g. 100 to cap
    LOGGING_STEPS     = 5
    SAVE_STEPS        = 50
    
    EXPORT_GGUF       = False   # Set True to save GGUF after training (needs llama.cpp)
    GGUF_QUANT        = "q4_k_m"
    OLLAMA_MODEL_NAME = "Abadd0n-4B"
    
    print("=" * 60)
    print("  Abadd0n — QLoRA Fine-Tuning (Unsloth / Qwen3)")
    print("=" * 60)
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Dataset       : {DATASET_PATH}")
    print(f"  LoRA rank     : {LORA_R}   alpha: {LORA_ALPHA}")
    print(f"  Batch size    : {PER_DEVICE_BATCH} × {GRAD_ACCUM_STEPS} grad accum")
    print(f"  Learning rate : {LEARNING_RATE}   scheduler: {LR_SCHEDULER}")
    print("=" * 60)
    
    # ══════════════════════════════════════════════════════════════
    #  ② LOAD MODEL & TOKENIZER
    # ══════════════════════════════════════════════════════════════
    print("\n[1/5] Loading model & tokenizer …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_NAME,
        max_seq_length  = MAX_SEQ_LENGTH,
        dtype           = DTYPE,
        load_in_4bit    = LOAD_IN_4BIT,
        # token           = "hf_…",
        attn_implementation = "sdpa",
    )
    
    # Apply the correct chat template for Qwen3 / ChatML format
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    
    print(f"  [OK] Loaded  {MODEL_NAME}")
    print(f"  [OK] Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # ══════════════════════════════════════════════════════════════
    #  ③ ATTACH LoRA ADAPTERS
    # ══════════════════════════════════════════════════════════════
    print("\n[2/5] Attaching LoRA adapters …")
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = LORA_R,
        target_modules           = LORA_TARGET_MODS,
        lora_alpha               = LORA_ALPHA,
        lora_dropout             = LORA_DROPOUT,
        bias                     = "none",
        use_gradient_checkpointing= "unsloth",   # Unsloth's optimised checkpointing
        random_state             = 3407,
        use_rslora               = False,        # RSLoRA scales alpha by sqrt(r)
        loftq_config             = None,
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Trainable params: {trainable:,}  ({100 * trainable / total:.2f} % of {total:,})")
    
    # ══════════════════════════════════════════════════════════════
    #  ④ PREPARE DATASET
    # ══════════════════════════════════════════════════════════════
    print(f"\n[3/5] Loading dataset from {DATASET_PATH} …")
    
    def load_jsonl(path: str) -> Dataset:
        """Load a .jsonl file and convert each record into ChatML messages."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
    
                # Support two formats:
                # (a) {"instruction": "…", "input": "…", "output": "…"}  — Alpaca style
                # (b) {"messages": [{"role": "…", "content": "…"}, …]}    — ChatML style
                if "messages" in obj:
                    records.append(obj)
                else:
                    user_msg = obj.get("instruction", "")
                    ctx      = obj.get("input", "")
                    if ctx:
                        user_msg = f"{user_msg}\n\n{ctx}"
                    assistant_msg = obj.get("output", "")
                    records.append({
                        "messages": [
                            {"role": "system",    "content": "You are Abadd0n, a powerful AI assistant."},
                            {"role": "user",      "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ]
                    })
        return Dataset.from_list(records)
    
    raw_dataset = load_jsonl(DATASET_PATH)
    print(f"  [OK] Loaded {len(raw_dataset)} training examples")
    
    def apply_chat_template(examples):
        """Format messages list into a single tokenized text string."""
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize           = False,
                add_generation_prompt = False,
            )
            texts.append(text)
        return {"text": texts}
    
    dataset = raw_dataset.map(apply_chat_template, batched=True)
    print(f"  [OK] Dataset formatted with chat template")
    print(f"  [OK] Sample:\n{dataset[0]['text'][:300]}…")
    
    # ══════════════════════════════════════════════════════════════
    #  ⑤ SFTTrainer — Instruction Tuning
    # ══════════════════════════════════════════════════════════════
    print("\n[4/5] Configuring SFTTrainer …")
    trainer = SFTTrainer(
        model        = model,
        tokenizer    = tokenizer,
        train_dataset= dataset,
        args         = SFTConfig(
            dataset_text_field        = "text",
            max_seq_length            = MAX_SEQ_LENGTH,
            dataset_num_proc          = None,
            packing                   = False,   # Enable for short examples to gain speed
            # ── Output & logging ──
            output_dir                = OUTPUT_DIR,
            logging_steps             = LOGGING_STEPS,
            save_steps                = SAVE_STEPS,
            save_total_limit          = 2,
            report_to                 = "none",
            # ── Batching & accumulation ──
            per_device_train_batch_size = PER_DEVICE_BATCH,
            gradient_accumulation_steps = GRAD_ACCUM_STEPS,
            # ── LR schedule ──
            num_train_epochs          = NUM_EPOCHS,
            max_steps                 = MAX_STEPS,
            learning_rate             = LEARNING_RATE,
            warmup_ratio              = WARMUP_RATIO,
            lr_scheduler_type         = LR_SCHEDULER,
            # ── Mixed precision ──
            fp16                      = not torch.cuda.is_bf16_supported(),
            bf16                      = torch.cuda.is_bf16_supported(),
            # ── Optimiser ──
            optim                     = OPTIM,
            weight_decay              = WEIGHT_DECAY,
            # ── Reproducibility ──
            seed                      = 3407,
        ),
    )
    
    # ══════════════════════════════════════════════════════════════
    #  ⑥ RUN TRAINING
    # ══════════════════════════════════════════════════════════════
    if __name__ == "__main__":
        print("\n[5/5] Starting QLoRA training …\n")
    
        gpu_stats = torch.cuda.get_device_properties(0)
        start_vram = round(torch.cuda.memory_reserved() / 1024 ** 3, 2)
        total_vram = round(gpu_stats.total_memory / 1024 ** 3, 2)
        print(f"  GPU : {gpu_stats.name}")
        print(f"  VRAM: {start_vram} GB reserved / {total_vram} GB total")
    
        trainer_stats = trainer.train()
    
        used_vram = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        pct       = round(used_vram / total_vram * 100, 1)
        print(f"\n  Peak VRAM used: {used_vram} GB ({pct} % of {total_vram} GB)")
        print(f"  Training time : {round(trainer_stats.metrics['train_runtime'] / 60, 2)} min")
    
        # ── Save LoRA adapters ───────────────────────────────────
        print(f"\nSaving LoRA adapters to ./{LORA_OUTPUT_DIR} …")
        model.save_pretrained(LORA_OUTPUT_DIR)
        tokenizer.save_pretrained(LORA_OUTPUT_DIR)
        print("  [OK] Adapters saved")
    
        # ── Optional: merge + save GGUF for Ollama ──────────────
        if EXPORT_GGUF:
            from ollama_export import write_modelfile
            from persona import PERSONA

            gguf_dir = "abadd0n_gguf"
            print(f"\nExporting merged model to GGUF ({GGUF_QUANT}) …")
            model.save_pretrained_merged("abadd0n_merged", tokenizer, save_method="merged_16bit")
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=GGUF_QUANT)
            write_modelfile(
                gguf_dir,
                model_name=OLLAMA_MODEL_NAME,
                persona=PERSONA,
                num_ctx=MAX_SEQ_LENGTH,
                temperature=0.8,
            )
            print(f"  [OK] GGUF + Modelfile saved to ./{gguf_dir}")
            print(f"      ollama create {OLLAMA_MODEL_NAME} -f ./{gguf_dir}/Modelfile")
    
        print("\n╔══════════════════════════════════╗")
        print("║  Abadd0n training complete! [OK] ║")
        print("╚══════════════════════════════════╝")