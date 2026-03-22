"""
Export trained Abadd0n LoRA to Hugging Face Hub.

Push LoRA adapters and/or merged model to HF. Requires HF_TOKEN or `huggingface-cli login`.

Usage (activate venv first):
    venv_win\\Scripts\\activate   # Windows
    source venv_wsl/bin/activate  # WSL / Linux
    python export_hf.py USERNAME/Abadd0n1.0-bnb-4bit

Options:
    --lora-only    Push LoRA adapters only (small, users load with base model)
    --merged       Merge LoRA into base, push full model (larger)
    --gguf         Also push GGUF (q4_k_m) to repo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import venv_check
venv_check.require_abaddon_venv()

import pre_unsloth
pre_unsloth.before_import()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Abadd0n LoRA to Hugging Face Hub"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HF repo id, e.g. username/Abadd0n1.0-bnb-4bit",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="lora_model",
        help="Path to LoRA adapters (default: lora_model)",
    )
    parser.add_argument(
        "--lora-only",
        action="store_true",
        help="Push LoRA adapters only (no merge)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Merge LoRA into base, push full 16bit model",
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Also push GGUF (q4_k_m) to the repo",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repo",
    )
    args = parser.parse_args()

    lora_path = Path(args.lora_path)
    if not lora_path.is_dir():
        raise SystemExit(f"LoRA path not found: {lora_path}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set. Using cached login from huggingface-cli.")

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading LoRA from {lora_path} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=2048,
        load_in_4bit=True,
        attn_implementation="sdpa",
        token=token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    FastLanguageModel.for_inference(model)

    repo_id = args.repo_id
    push_kw = {"tokenizer": tokenizer, "private": args.private}
    if token:
        push_kw["token"] = token

    if args.lora_only:
        print(f"Pushing LoRA adapters to {repo_id} …")
        model.push_to_hub_merged(repo_id, save_method="lora", **push_kw)
        print(f"  [OK] LoRA pushed to https://huggingface.co/{repo_id}")
        return

    if args.merged:
        print(f"Merging and pushing full model to {repo_id} …")
        model.push_to_hub_merged(repo_id, save_method="merged_16bit", **push_kw)
        print(f"  [OK] Merged model pushed to https://huggingface.co/{repo_id}")

    if args.gguf:
        print(f"Pushing GGUF (q4_k_m) to {repo_id} …")
        model.push_to_hub_gguf(
            repo_id,
            tokenizer,
            quantization_method="q4_k_m",
            token=token,
            private=args.private,
        )
        print(f"  [OK] GGUF pushed to https://huggingface.co/{repo_id}")

    if not args.lora_only and not args.merged and not args.gguf:
        print("Specify --lora-only, --merged, and/or --gguf")
        sys.exit(1)

    print(f"\n  Done. View at https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
