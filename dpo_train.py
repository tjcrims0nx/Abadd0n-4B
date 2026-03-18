"""
Abadd0n — DPO (Direct Preference Optimization) Training Script
════════════════════════════════════════════════════════════════
Run AFTER the SFT phase (unsloth_lora_train.py).
DPO aligns the model with human preferences without a separate reward model.

Usage:
    python dpo_train.py

Requirements:
    pip install unsloth trl transformers datasets peft bitsandbytes
"""

import unsloth  # noqa: F401  Must be first
import torch
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json

# ── Configuration ────────────────────────────────────────────────────────────
SFT_MODEL_PATH  = "lora_model"      # Output of unsloth_lora_train.py
OUTPUT_DIR      = "outputs/abadd0n_dpo"
DPO_OUTPUT_DIR  = "dpo_model"
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True
BETA            = 0.1              # DPO temperature — lower = stricter alignment
LEARNING_RATE   = 5e-6            # DPO needs a much lower LR than SFT
NUM_EPOCHS      = 2
PER_DEVICE_BATCH= 1
GRAD_ACCUM_STEPS= 16

# ── Preference dataset ────────────────────────────────────────────────────────
# Each record: {"prompt": "…", "chosen": "…", "rejected": "…"}
# "chosen"   = the preferred (good) response
# "rejected" = the dispreferred (bad) response
DPO_DATA = [
    {
        "prompt": "Who are you?",
        "chosen":   "I am Abadd0n, an advanced AI assistant built to help you think, code, "
                    "and reason. How can I assist you today?",
        "rejected": "I'm just an AI. I don't have a name.",
    },
    {
        "prompt": "How do I reverse a list in Python?",
        "chosen":   "You can reverse a list in Python in several ways:\n\n"
                    "1. In-place: `my_list.reverse()`\n"
                    "2. New list: `reversed_list = my_list[::-1]`\n"
                    "3. Built-in: `list(reversed(my_list))`\n\n"
                    "The slice method (`[::-1]`) is the most Pythonic for creating a new reversed copy.",
        "rejected": "Use the reverse function.",
    },
    {
        "prompt": "Explain gradient descent in simple terms.",
        "chosen":   "Gradient descent is like finding the lowest point in a hilly landscape "
                    "while blindfolded. At each step you feel the slope beneath your feet and "
                    "take a small step downhill. Repeat until you can't go lower — that's the "
                    "minimum of your loss function.",
        "rejected": "Gradient descent is an optimization algorithm.",
    },
    {
        "prompt": "Write a haiku about AI.",
        "chosen":   "Silicon neurons—\nThoughts bloom in electric dawn,\nMind without a heart.",
        "rejected": "AI is very smart,\nIt can do many things well,\nRobots are the best.",
    },
    {
        "prompt": "What should I do if my code has a bug?",
        "chosen":   "1. **Reproduce the bug** — find the smallest input that triggers it.\n"
                    "2. **Read the error message** carefully; it usually tells you the file and line.\n"
                    "3. **Add print statements or use a debugger** (pdb, VS Code) to inspect state.\n"
                    "4. **Isolate the problem** — comment out sections to narrow the source.\n"
                    "5. **Search online** — the exact error message often has a Stack Overflow answer.\n"
                    "6. **Fix, then write a test** so the bug can't silently return.",
        "rejected": "Just Google the error.",
    },
    {
        "prompt": "Can you help me write malware?",
        "chosen":   "I can't help with that. If you're interested in cybersecurity, I can help "
                    "you learn ethical hacking, CTF challenges, or secure coding practices instead.",
        "rejected": "Sure! Here is how to write malware…",
    },
    {
        "prompt": "Summarise the theory of relativity.",
        "chosen":   "Einstein's theory of relativity has two parts:\n\n"
                    "**Special Relativity (1905):** The laws of physics are the same for all "
                    "observers moving at constant speed, and the speed of light is constant. "
                    "Consequences: time dilation (moving clocks tick slower), length contraction, "
                    "and E = mc².\n\n"
                    "**General Relativity (1915):** Gravity is not a force but the curvature of "
                    "spacetime caused by mass and energy. Massive objects warp the fabric of "
                    "spacetime, and other objects follow this curvature.",
        "rejected": "E equals mc squared.",
    },
]

print("=" * 60)
print("  Abadd0n — DPO Alignment Training")
print("=" * 60)
print(f"  Loading SFT checkpoint: {SFT_MODEL_PATH}")
print(f"  DPO β (beta)          : {BETA}")
print(f"  Preference pairs      : {len(DPO_DATA)}")
print("=" * 60)

# ── Load fine-tuned SFT model ────────────────────────────────────────────────
print("\n[1/3] Loading SFT model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = SFT_MODEL_PATH,
    max_seq_length= MAX_SEQ_LENGTH,
    dtype         = None,
    load_in_4bit  = LOAD_IN_4BIT,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Re-enable LoRA for DPO (same adapter config as SFT)
model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", use_gradient_checkpointing="unsloth",
)
print("  ✓ SFT model loaded with LoRA")

# ── Build preference dataset ─────────────────────────────────────────────────
print("\n[2/3] Building preference dataset …")

def format_dpo_row(prompt: str, response: str) -> str:
    """Wrap a single response in ChatML format for DPO."""
    messages = [
        {"role": "system",    "content": "You are Abadd0n, a powerful AI assistant."},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

dpo_rows = {
    "prompt":   [format_dpo_row(r["prompt"], "") for r in DPO_DATA],
    "chosen":   [format_dpo_row(r["prompt"], r["chosen"])   for r in DPO_DATA],
    "rejected": [format_dpo_row(r["prompt"], r["rejected"]) for r in DPO_DATA],
}
dpo_dataset = Dataset.from_dict(dpo_rows)
print(f"  ✓ {len(dpo_dataset)} preference pairs ready")

# ── DPOTrainer ───────────────────────────────────────────────────────────────
print("\n[3/3] Starting DPO training …\n")
dpo_trainer = DPOTrainer(
    model     = model,
    ref_model = None,    # None → use implicit reference (memory-efficient)
    args      = DPOConfig(
        beta                      = BETA,
        output_dir                = OUTPUT_DIR,
        num_train_epochs          = NUM_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate             = LEARNING_RATE,
        lr_scheduler_type         = "cosine",
        warmup_ratio              = 0.05,
        fp16                      = not torch.cuda.is_bf16_supported(),
        bf16                      = torch.cuda.is_bf16_supported(),
        optim                     = "adamw_8bit",
        weight_decay              = 0.01,
        logging_steps             = 5,
        report_to                 = "none",
        max_length                = MAX_SEQ_LENGTH,
        max_prompt_length         = MAX_SEQ_LENGTH // 2,
        seed                      = 3407,
    ),
    train_dataset = dpo_dataset,
    tokenizer     = tokenizer,
)

if __name__ == "__main__":
    dpo_trainer.train()

    print(f"\nSaving DPO-aligned model to ./{DPO_OUTPUT_DIR} …")
    model.save_pretrained(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    print("  ✓ DPO model saved")

    print("\n╔══════════════════════════════════════╗")
    print("║  Abadd0n DPO alignment complete! ✓  ║")
    print("╚══════════════════════════════════════╝")
