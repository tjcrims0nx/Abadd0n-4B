"""
Agent RPC runtime with tool streaming and block streaming.
Inspired by OpenClaw: https://github.com/openclaw/openclaw

Usage:
  python main.py agent                  # Interactive chat with tools
  python main.py agent --message "..."  # Single-turn, stream response + tools (auto-approves writes)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cli_theme import DIM, LABEL, RESET, muted, success


def _tool_emit(tool: str, status: str, detail: str = "") -> None:
    """Emit tool event for streaming display."""
    parts = [f"  {DIM}[tool]{RESET} {LABEL}{tool}{RESET}"]
    if status:
        parts.append(f" {status}")
    if detail:
        parts.append(f" {DIM}{detail}{RESET}")
    print("".join(parts))


def run_agent(args: list[str] | None = None) -> int:
    """Run the agent runtime in RPC mode with tool and block streaming."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Abadd0n Agent — RPC runtime with tools")
    parser.add_argument("--message", "-m", type=str, help="Single-turn message (no interactive loop)")
    parser.add_argument("--tools", action="store_true", help="Show available tools and exit")
    ns = parser.parse_args(args or [])

    if ns.tools:
        _print_tools_help()
        return 0

    print(f"\n{LABEL}Abadd0n Agent{RESET} {DIM}\u2014{RESET} {muted('RPC runtime')}")
    print(muted("Tool streaming, block streaming"))
    print()

    # Lazy import to avoid circular dependency; main is loaded when we're invoked
    import main

    model, tokenizer = main.load_model_and_tokenizer()
    persona = main._system_content_with_skills()
    if len(persona) > len(main.PERSONA):
        print(muted("ClawHub skills loaded from project/skills/"))
    conversation_history = [{"role": "system", "content": persona}]

    if ns.message:
        return _run_single_turn(main, model, tokenizer, conversation_history, ns.message.strip())

    return _run_interactive(main, model, tokenizer, conversation_history)


def _run_single_turn(main_mod, model, tokenizer, conversation_history: list, message: str) -> int:
    """Single-turn: chat once, handle file edits (auto-approve writes when non-interactive)."""
    _tool_emit("chat", "streaming", "…")
    prev = os.environ.get("ABADDON_AUTO_APPROVE_WRITES")
    try:
        os.environ["ABADDON_AUTO_APPROVE_WRITES"] = "1"
        response = main_mod.chat(model, tokenizer, message, conversation_history)
        main_mod.handle_file_edits(response)
    finally:
        if prev is None:
            os.environ.pop("ABADDON_AUTO_APPROVE_WRITES", None)
        else:
            os.environ["ABADDON_AUTO_APPROVE_WRITES"] = prev
    return 0


def _run_interactive(main_mod, model, tokenizer, conversation_history: list) -> int:
    """Interactive loop: same as main chat but with agent tool streaming."""
    from cli_theme import BOX_V, BOLD, GRAY

    print(f"\n{GRAY}{BOX_V}{RESET} {BOLD}{LABEL}Agent{RESET} {DIM}\u00b7{RESET} {DIM}type to chat, exit to quit{RESET}")
    print(f"{GRAY}{BOX_V}{RESET} {DIM}Tools: write_file, read, ls, find, compile{RESET}\n")

    while True:
        try:
            prompt = input(f"\n{GRAY}{BOX_V}{RESET} {LABEL}Message{RESET} \u2192 ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Agent stopped{RESET}")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print(muted("Goodbye"))
            break
        if prompt.lower() in ("clear", "new"):
            conversation_history[:] = [{"role": "system", "content": main_mod._system_content_with_skills()}]
            print(success("New thread" if prompt.lower() == "new" else "Conversation cleared"))
            continue

        if prompt.strip().startswith("/"):
            from coding_tools import handle_slash_command

            colors = {
                "cyan": "\033[96m", "label": "\033[94m", "gray": "\033[90m",
                "dim": "\033[2m", "white": "\033[97m", "green": "\033[32m",
                "red": "\033[31m", "reset": "\033[0m",
            }
            handle_slash_command(prompt, main_mod.PROJECT_ROOT, colors)
            continue

        _tool_emit("chat", "streaming", "…")
        response = main_mod.chat(model, tokenizer, prompt, conversation_history)
        main_mod.handle_file_edits(response)

    return 0


def _print_tools_help() -> None:
    """Print available tools (OpenClaw-style)."""
    from cli_theme import BOX_H, BOX_TL, BOX_TR, BOX_V

    w = 56
    rule = f"{DIM}{BOX_H * w}{RESET}"
    print(f"\n{LABEL}Abadd0n Agent{RESET} {DIM}\u2014{RESET} {muted('available tools')}")
    print(rule)
    rows = [
        ("write_file", "Create/overwrite file (from model <write_file> blocks)"),
        ("read_file", "Read file contents"),
        ("list_dir", "List directory entries"),
        ("find_in_files", "Search file contents (needle, glob)"),
        ("run_bash", "Execute shell command (sandboxed)"),
        ("compile_python", "Syntax-check .py file"),
    ]
    for name, desc in rows:
        print(f"  {LABEL}{name:<18}{RESET} {DIM}{desc}{RESET}")
    print(rule)
    print(muted("Model outputs <write_file path=\"...\"> for file creation; agent parses and applies."))
    print()


if __name__ == "__main__":
    sys.exit(run_agent(sys.argv[1:]))
