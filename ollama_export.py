"""
Ollama export helpers for Abadd0n-4B.

Provides Modelfile generation and full LoRA → GGUF → Ollama pipeline.
Used by unsloth_lora_train.py (post-training) and export_ollama.py (standalone).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OLLAMA_MODEL_NAME = "Abadd0n-4B"
DEFAULT_NUM_CTX = 2048
DEFAULT_TEMPERATURE = 0.8


def _find_gguf(gguf_dir: Path) -> Path | None:
    """Return first .gguf file in directory, or None."""
    matches = list(gguf_dir.glob("*.gguf"))
    return matches[0] if matches else None


def write_modelfile(
    gguf_dir: Path,
    model_name: str = OLLAMA_MODEL_NAME,
    persona: str = "",
    *,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Path:
    """
    Write an Ollama Modelfile into gguf_dir.
    FROM uses the first .gguf file found in gguf_dir.
    Returns path to the written Modelfile.
    """
    gguf_dir = Path(gguf_dir).resolve()
    gguf_file = _find_gguf(gguf_dir)
    if gguf_file is None:
        raise FileNotFoundError(f"No .gguf file found in {gguf_dir}")

    from_path = gguf_file.name
    lines = [
        f"FROM {from_path}",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER num_ctx {num_ctx}",
    ]
    if persona:
        if '"""' in persona:
            persona = persona.replace('"""', '\\"\\"\\"')
        lines.append(f'SYSTEM """{persona}"""')

    modelfile_path = gguf_dir / "Modelfile"
    modelfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return modelfile_path


def export_lora_to_ollama(
    lora_path: str | Path,
    gguf_output_dir: str | Path,
    *,
    model_name: str = OLLAMA_MODEL_NAME,
    gguf_quant: str = "q4_k_m",
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = DEFAULT_TEMPERATURE,
    run_ollama_create: bool = True,
) -> Path:
    """
    Load LoRA from lora_path, merge, export to GGUF, write Modelfile, optionally run ollama create.
    Returns path to the Modelfile.
    """
    from persona import PERSONA

    import pre_unsloth

    pre_unsloth.before_import()
    from unsloth import FastLanguageModel

    lora_path = Path(lora_path).resolve()
    gguf_dir = Path(gguf_output_dir).resolve()
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LoRA from {lora_path} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=num_ctx,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    FastLanguageModel.for_inference(model)

    merge_dir = gguf_dir.parent / "abadd0n_merged"
    print(f"Merging to {merge_dir} …")
    model.save_pretrained_merged(str(merge_dir), tokenizer, save_method="merged_16bit")

    print(f"Exporting GGUF ({gguf_quant}) to {gguf_dir} …")
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=gguf_quant)

    modelfile_path = write_modelfile(
        gguf_dir,
        model_name=model_name,
        persona=PERSONA,
        num_ctx=num_ctx,
        temperature=temperature,
    )
    print(f"  [OK] Modelfile written to {modelfile_path}")

    if run_ollama_create:
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            cmd = [ollama_bin, "create", model_name, "-f", str(modelfile_path)]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"  [OK] Ollama model {model_name!r} created")
        else:
            print(f"ollama not in PATH. Run manually:")
            print(f"  ollama create {model_name} -f {modelfile_path}")

    return modelfile_path
