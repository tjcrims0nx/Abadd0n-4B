"""Send CLI: message delivery to sessions."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cli_theme import DIM, LABEL, RESET, muted


def run_send(args=None) -> int:
    """Run the send CLI."""
    print(f"\n{LABEL}Abadd0n Send{RESET} {DIM}\u2014{RESET} {muted('message delivery')}")
    print(muted("(Not yet implemented)"))
    return 0
