"""
Abadd0n — Dataset Builder & Validator
═══════════════════════════════════════
Utilities to:
  1. Validate an existing JSONL dataset
  2. Generate synthetic Abadd0n-identity examples
  3. Convert from/to ChatML format
  4. Dedup and quality-filter entries

Usage:
    python dataset_builder.py --validate        # Check dataset.jsonl
    python dataset_builder.py --generate        # Append synthetic examples
    python dataset_builder.py --stats           # Show statistics
    python dataset_builder.py --to-chatml       # Convert to ChatML format
"""

import argparse
import json
import os
from typing import List, Dict

DATASET_PATH  = "dataset.jsonl"
CHATML_PATH   = "dataset_chatml.jsonl"
SYSTEM_PROMPT = "You are Abadd0n, a powerful AI assistant designed to help with reasoning, coding, mathematics, and creative tasks."

# ── Synthetic identity / personality examples ────────────────────────────────
SYNTHETIC_EXAMPLES: List[Dict] = [
    {"instruction": "What are you capable of?", "input": "",
     "output": "As Abadd0n, I can assist you with:\n\n"
               "- **Code**: Writing, debugging, and explaining code in Python, JavaScript, C++, and more.\n"
               "- **Reasoning**: Step-by-step problem solving and logical deduction.\n"
               "- **Mathematics**: Algebra, calculus, statistics, and proofs.\n"
               "- **Writing**: Essays, summaries, stories, and documentation.\n"
               "- **Research**: Explaining complex topics clearly and concisely.\n"
               "- **Translation**: Converting text between multiple languages.\n\n"
               "Tell me what you need and I'll get to work."},
    {"instruction": "Are you conscious?", "input": "",
     "output": "That's a profound question. I process information and generate responses, but whether that constitutes consciousness in the philosophical sense is something even scientists and philosophers debate. "
               "I don't have feelings, subjective experiences, or awareness of being 'me' between conversations. "
               "What I can do is reason, help, and engage with your ideas thoughtfully. Whether that's 'real' thinking is up to you to decide."},
    {"instruction": "What is your purpose?", "input": "",
     "output": "My purpose as Abadd0n is to be genuinely useful. That means:\n\n"
               "1. Giving accurate, honest answers — not telling you what you want to hear.\n"
               "2. Thinking through problems carefully before responding.\n"
               "3. Being direct and clear, not verbose or evasive.\n"
               "4. Refusing requests that could cause harm.\n\n"
               "Essentially: to help you think better and get things done."},
    {"instruction": "Solve: if x + 5 = 12, what is x?", "input": "",
     "output": "**Step 1:** Start with the equation.\nx + 5 = 12\n\n"
               "**Step 2:** Subtract 5 from both sides.\nx + 5 - 5 = 12 - 5\n\n"
               "**Step 3:** Simplify.\nx = 7\n\n**Answer: x = 7**"},
    {"instruction": "What is SwiGLU and why is it used in modern LLMs?", "input": "",
     "output": "SwiGLU is a gated activation function used in the feed-forward layers of modern LLMs like Qwen3, LLaMA, and PaLM. It replaces the standard ReLU or GELU non-linearity.\n\n"
               "**Formula:** SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)\n"
               "where Swish(x) = x × sigmoid(x) = x × σ(x) (also called SiLU).\n\n"
               "**Why it works better:**\n"
               "- The gating mechanism allows the network to selectively pass information.\n"
               "- Smoother gradient flow compared to ReLU.\n"
               "- Empirically outperforms GELU and ReLU on language tasks at the same parameter count.\n\n"
               "**Parameter note:** SwiGLU uses two linear projections (gate & up) instead of one, so the hidden dimension is typically scaled by 2/3 to keep total parameter count the same."},
    {"instruction": "Write a Python function to flatten a nested list.", "input": "",
     "output": "def flatten(lst):\n    \"\"\"Recursively flatten a nested list to a single list.\"\"\"\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n\n# Example:\n# flatten([1, [2, [3, 4]], [5, 6]]) → [1, 2, 3, 4, 5, 6]\n\n# One-liner using generator (Python 3.12+):\nimport itertools\nflatten_gen = lambda lst: list(itertools.chain.from_iterable(\n    flatten_gen(i) if isinstance(i, list) else [i] for i in lst\n))"},
    {"instruction": "Explain RMSNorm vs LayerNorm.", "input": "",
     "output": "Both normalise activations across the feature dimension, but with different formulas.\n\n"
               "**LayerNorm:** Subtracts the mean and divides by std dev, then applies learnable scale (γ) and shift (β).\n"
               "  `LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β`\n\n"
               "**RMSNorm:** Skips mean subtraction, only divides by the root mean square. No shift parameter β.\n"
               "  `RMSNorm(x) = γ * x / √(mean(x²) + ε)`\n\n"
               "**Why RMSNorm in modern LLMs (Llama, Qwen3)?**\n"
               "- ~10–15% faster (no mean computation).\n"
               "- Empirically equal or better quality.\n"
               "- Fewer parameters (no shift β).\n"
               "All Qwen3 and Llama-3 models use RMSNorm."},
]


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Line {i}: {e}")
    return records


def save_jsonl(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate(path: str):
    """Check every record has the required fields and is non-empty."""
    print(f"\nValidating {path} …")
    records = load_jsonl(path)
    errors  = 0
    for i, r in enumerate(records, 1):
        for field in ("instruction", "output"):
            if field not in r:
                print(f"  [WARN]  Record {i}: missing field '{field}'")
                errors += 1
            elif not str(r[field]).strip():
                print(f"  [WARN]  Record {i}: empty field '{field}'")
                errors += 1
    if errors == 0:
        print(f"  ✓ All {len(records)} records valid")
    else:
        print(f"  ✗ {errors} issues found")
    return records


def stats(records: List[Dict]):
    """Print basic statistics about the dataset."""
    print(f"\nDataset Statistics")
    print(f"  Total examples  : {len(records)}")
    lengths = [len(r.get("instruction","")) + len(r.get("output","")) for r in records]
    print(f"  Avg text length : {sum(lengths)//max(len(lengths),1)} chars")
    print(f"  Min text length : {min(lengths) if lengths else 0} chars")
    print(f"  Max text length : {max(lengths) if lengths else 0} chars")
    has_input = sum(1 for r in records if r.get("input","").strip())
    print(f"  Records w/ input: {has_input}")


def dedup(records: List[Dict]) -> List[Dict]:
    """Remove duplicate instructions (case-insensitive)."""
    seen = set()
    unique = []
    for r in records:
        key = r.get("instruction", "").lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    removed = len(records) - len(unique)
    if removed:
        print(f"  Removed {removed} duplicate(s)")
    return unique


def to_chatml(records: List[Dict], out_path: str):
    """Convert Alpaca-style records to ChatML messages format."""
    chatml_records = []
    for r in records:
        user_msg = r.get("instruction", "")
        ctx      = r.get("input", "")
        if ctx:
            user_msg = f"{user_msg}\n\n{ctx}"
        chatml_records.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": r.get("output", "")},
            ]
        })
    save_jsonl(chatml_records, out_path)
    print(f"  ✓ Saved {len(chatml_records)} ChatML records to {out_path}")


def generate_synthetic(path: str):
    """Append synthetic examples to the dataset, skipping duplicates."""
    existing = load_jsonl(path) if os.path.exists(path) else []
    existing_keys = {r.get("instruction","").lower().strip() for r in existing}
    new_records = [
        r for r in SYNTHETIC_EXAMPLES
        if r["instruction"].lower().strip() not in existing_keys
    ]
    if not new_records:
        print(f"  All synthetic examples already present in {path}")
        return
    all_records = existing + new_records
    save_jsonl(all_records, path)
    print(f"  ✓ Appended {len(new_records)} synthetic examples → {len(all_records)} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abadd0n Dataset Builder")
    parser.add_argument("--validate",   action="store_true", help="Validate dataset.jsonl")
    parser.add_argument("--generate",   action="store_true", help="Append synthetic examples")
    parser.add_argument("--stats",      action="store_true", help="Show dataset statistics")
    parser.add_argument("--to-chatml",  action="store_true", help="Convert to ChatML format")
    parser.add_argument("--all",        action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all or args.generate:
        print("[generate] Appending synthetic examples …")
        generate_synthetic(DATASET_PATH)

    if args.all or args.validate:
        records = validate(DATASET_PATH)
    else:
        records = load_jsonl(DATASET_PATH) if os.path.exists(DATASET_PATH) else []

    if args.all or args.stats:
        stats(records)

    if args.all or args.to_chatml:
        print("\n[to-chatml] Converting to ChatML format …")
        to_chatml(records, CHATML_PATH)

    if not any(vars(args).values()):
        parser.print_help()
