#!/usr/bin/env python3
"""
Abadd0n CLI entry point.

Usage (activate venv first):
  venv_win\\Scripts\\activate   # Windows
  source venv_wsl/bin/activate  # WSL
  source venv/bin/activate     # Native Linux
  python cli.py                 # Chat (default)
  python cli.py gateway         # Gateway WS control plane
  python cli.py agent           # Agent RPC runtime
  python cli.py doctor          # Diagnostics
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import venv_check
venv_check.require_abaddon_venv()


def main() -> int:
    args = sys.argv[1:]
    cmd = (args[0].lower() if args else "").strip()

    if cmd == "gateway":
        from core.gateway import run_gateway
        return run_gateway(args[1:])
    if cmd == "agent":
        from core.agent import run_agent
        return run_agent(args[1:])
    if cmd == "send":
        from core.send import run_send
        return run_send(args[1:])
    if cmd == "onboarding":
        from core.onboarding import run_onboarding
        return run_onboarding(args[1:])
    if cmd == "doctor":
        from core.doctor import run_doctor
        return run_doctor(args[1:])
    if cmd in ("-h", "--help", "help"):
        print(__doc__.strip())
        print("\nCommands: gateway, agent, send, onboarding, doctor")
        print("Default (no args): chat")
        return 0

    # Default: run chat (delegate to main.py)
    import main as chat_main
    chat_main.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
