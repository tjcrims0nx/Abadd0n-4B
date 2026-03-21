"""
Call before_import() immediately before `import unsloth`.

Unsloth prints "Will patch your computer..." then loads many patches and Triton
kernels with little further output — that phase often takes 3–15+ minutes on
first run (WSL + project on /mnt/c is especially slow). This is not a hang.

Unsloth's GRPO trainer patch injects torch.compile options such as
``triton.enable_persistent_tma_matmul`` (see unsloth/models/rl.py). That key
exists only on newer PyTorch; on 2.5.x ``torch.compile`` rejects unknown keys
and the patch fails. We register a compatible default before Unsloth imports.
"""

from __future__ import annotations

import os
import sys
import torch

_INDUCTOR_COMPAT_DONE = False


def _register_unsloth_compat_inductor_compile_options() -> None:
    """
    Ensure torch.compile(..., options={...}) accepts keys Unsloth injects for GRPO
    when running on PyTorch builds that do not yet define them in inductor config.
    """
    global _INDUCTOR_COMPAT_DONE
    if _INDUCTOR_COMPAT_DONE:
        return
    _INDUCTOR_COMPAT_DONE = True
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return

    # Flat keys as used by torch._TorchCompileInductorWrapper.apply_options /
    # torch._inductor.config (see torch/utils/_config_module.py SubConfigProxy).
    compat: dict[str, object] = {
        # Added in PyTorch ~2.6; Unsloth always injects this for CUDA GRPO (rl.py).
        "triton.enable_persistent_tma_matmul": False,
    }
    for key, default in compat.items():
        if key in inductor_config._allowed_keys:
            continue
        inductor_config._config[key] = default
        inductor_config._default[key] = default
        inductor_config._allowed_keys.add(key)


def before_import() -> None:
    _register_unsloth_compat_inductor_compile_options()

    # 1. Patch for torchao / unsloth compatibility (Windows)
    for i in range(1, 8):
        attr = f"int{i}"
        if not hasattr(torch, attr):
            setattr(torch, attr, torch.int8)

    # 2. Globally disable Flash Attention SDP (problematic on Windows/RTX 2050)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # 3. Output and console setup
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError, AttributeError):
        pass

    print(
        "Abadd0n: Loading Unsloth — after the sloth message the process can stay quiet for "
        "several minutes while Triton/JIT compiles (first run is slowest; not frozen).",
        flush=True,
    )

    _wsl_fast_compiler_caches()


def _wsl_fast_compiler_caches() -> None:
    if not os.environ.get("WSL_DISTRO_NAME"):
        return
    home = os.path.expanduser("~")
    if not home or home.startswith("/mnt/"):
        return
    for env_key, dirname in (
        ("TRITON_CACHE_DIR", "triton-abadd0n"),
        ("TORCHINDUCTOR_CACHE_DIR", "torch-inductor-abadd0n"),
    ):
        if env_key in os.environ:
            continue
        path = os.path.join(home, ".cache", dirname)
        os.makedirs(path, exist_ok=True)
        os.environ[env_key] = path
