"""Doctor CLI: diagnostics and health checks."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def run_doctor(args=None) -> int:
    """Run diagnostics."""
    # Minimal implementation: reuse check_torch / debug_unsloth style checks
    print("\nAbadd0n Doctor \u2014 diagnostics")
    try:
        from cli_theme import success, error as theme_error, muted
    except ImportError:
        def success(m, d=""): print(f"  \u2713 {m}" + (f" {d}" if d else ""))
        def theme_error(m, d=""): print(f"  \u2717 {m}" + (f" {d}" if d else ""))
        muted = lambda m: print(f"  {m}")

    checks = []
    try:
        import torch
        cuda = torch.cuda.is_available()
        checks.append(("PyTorch", True, f"CUDA={cuda}"))
    except Exception as e:
        checks.append(("PyTorch", False, str(e)))

    try:
        import pre_unsloth
        pre_unsloth.before_import()
        checks.append(("pre_unsloth", True, "inducer compat registered"))
    except Exception as e:
        checks.append(("pre_unsloth", False, str(e)))

    try:
        from unsloth import FastLanguageModel
        checks.append(("Unsloth", True, "import OK"))
    except Exception as e:
        checks.append(("Unsloth", False, str(e)))

    for name, ok, detail in checks:
        if ok:
            print(success(name, detail))
        else:
            print(theme_error(name, detail))

    return 0 if all(c[1] for c in checks) else 1
