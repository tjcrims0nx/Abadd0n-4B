"""Sanity check for WSL/native Linux venv: torch CUDA + pre_unsloth + Unsloth import.

Run from repo root with venv activated:  python linux/wsl_check.py

First Unsloth import can take many minutes (especially on /mnt/c); use PYTHONUNBUFFERED=1 for live output.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

print("torch", torch.__version__, flush=True)
print("cuda available", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), flush=True)

import pre_unsloth

pre_unsloth.before_import()

import torch._inductor.config as _inductor_config

assert "triton.enable_persistent_tma_matmul" in _inductor_config._allowed_keys, (
    "pre_unsloth should register GRPO torch.compile options for this PyTorch build"
)
print("pre_unsloth inductor compat OK", flush=True)

import unsloth  # noqa: F401

print("unsloth OK", flush=True)
