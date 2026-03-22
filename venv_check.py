"""
Ensure Abadd0n runs inside venv_win or venv_wsl.
Prevents loading the model from global Python (which may lack LoRA / updates).
"""

from __future__ import annotations

import os
import sys


def require_abaddon_venv() -> None:
    """Exit with instructions if not running in venv_win or venv_wsl."""
    prefix = getattr(sys, "prefix", "")
    venv_env = os.environ.get("VIRTUAL_ENV", "")
    for p in (prefix, venv_env):
        p_str = str(p).replace("\\", "/")
        if "venv_win" in p_str or "venv_wsl" in p_str:
            return  # OK

    msg = (
        "ERROR: Abadd0n must run inside venv_win or venv_wsl.\n"
        "Activate the project venv first:\n\n"
        "  Windows:  venv_win\\Scripts\\activate\n"
        "  WSL:      source venv_wsl/bin/activate\n\n"
        "Then:  python main.py   (or export_ollama.py, unsloth_lora_train.py)"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)
