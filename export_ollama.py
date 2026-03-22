"""
Export trained Abadd0n LoRA to Ollama as Abadd0n-4B.

Loads lora_model/, merges, exports to GGUF, writes a Modelfile with the Abadd0n
persona, and optionally runs `ollama create Abadd0n-4B`.

Usage (activate venv first):
    venv_win\\Scripts\\activate   # Windows
    source venv_wsl/bin/activate # WSL
    python export_ollama.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Must run in venv_win or venv_wsl
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import venv_check
venv_check.require_abaddon_venv()

from ollama_export import OLLAMA_MODEL_NAME, export_lora_to_ollama


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Abadd0n LoRA to Ollama (Abadd0n-4B)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="lora_model",
        help="Path to LoRA adapters (default: lora_model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="abadd0n_gguf",
        help="Directory for GGUF + Modelfile (default: abadd0n_gguf)",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="q4_k_m",
        help="GGUF quantization (default: q4_k_m)",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Do not run ollama create; only produce GGUF + Modelfile",
    )
    args = parser.parse_args()

    lora_path = Path(args.lora_path)
    if not lora_path.is_dir():
        raise SystemExit(f"LoRA path not found: {lora_path}")

    export_lora_to_ollama(
        lora_path,
        args.output_dir,
        model_name=OLLAMA_MODEL_NAME,
        gguf_quant=args.quant,
        run_ollama_create=not args.no_create,
    )
    print(f"\n  Done. Run: ollama run {OLLAMA_MODEL_NAME}")


if __name__ == "__main__":
    main()
