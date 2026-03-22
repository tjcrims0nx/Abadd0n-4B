"""Onboarding CLI: first-run setup and config."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cli_theme import DIM, LABEL, RESET, muted


def run_onboarding(args=None) -> int:
    """Run the onboarding CLI."""
    print(f"\n{LABEL}Abadd0n Onboarding{RESET} {DIM}\u2014{RESET} {muted('first-run setup')}")
    print(muted("(Not yet implemented)"))
    return 0
