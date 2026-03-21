"""
Local coding helpers for the Abadd0n CLI: read/list/search/compile under the project tree.
No network APIs — stdlib + pathlib only.
"""

from __future__ import annotations

import fnmatch
import os
import py_compile
import shutil
import shlex
import sys
from pathlib import Path

# Directories skipped when walking the repo (noise + large trees).
_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "venv_win",
        "venv_wsl",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "unsloth_compiled_cache",
        "lora_model",
        "outputs",
        ".cursor",
    }
)

_READ_MAX_BYTES = 400_000
_READ_MAX_LINES = 400
_FIND_MAX_HITS = 40
_TREE_MAX_DEPTH_DEFAULT = 2


def _resolve_under_roots(raw: str, roots: list[Path]) -> Path | None:
    raw = (raw or "").strip()
    if not raw or "\x00" in raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        cand = p.resolve()
    else:
        cand = (roots[0] / p).resolve()
    allow_out = os.environ.get("ABADDON_ALLOW_WRITES_OUTSIDE_ROOT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if allow_out:
        return cand if cand.is_file() or cand.is_dir() else None
    for root in roots:
        try:
            cand.relative_to(root)
            return cand
        except ValueError:
            continue
    return None


def _tool_roots(project_root: Path) -> list[Path]:
    roots = [project_root.resolve()]
    extra = os.environ.get("ABADDON_WRITE_ROOT", "").strip()
    if extra:
        roots.append(Path(extra).resolve())
    return roots


def _split_command(user_input: str) -> tuple[str, list[str]] | None:
    s = user_input.strip()
    if not s.startswith("/"):
        return None
    body = s[1:].strip()
    if not body:
        return "", []
    try:
        parts = shlex.split(body, posix=os.name != "nt")
    except ValueError as e:
        print(f"[tools] Bad quoting: {e}", file=sys.stderr)
        return None
    cmd = parts[0].lower()
    return cmd, parts[1:]


def _theme(colors: dict[str, str]) -> dict[str, str]:
    return {
        "label": colors.get("label") or colors.get("cyan", ""),
        "gray": colors.get("gray", ""),
        "dim": colors.get("dim", ""),
        "white": colors.get("white", ""),
        "green": colors.get("green", ""),
        "red": colors.get("red", ""),
        "reset": colors.get("reset", ""),
    }


def print_tools_help(colors: dict[str, str]) -> None:
    t = _theme(colors)
    lb, gy, dm, wt, rs = t["label"], t["gray"], t["dim"], t["white"], t["reset"]
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    rule_len = max(48, min(72, width - 6))
    print(f"{lb}ABADD0N local toolkit{rs} {gy}\u2014 offline commands (project sandbox){rs}")
    print(f"{dm}{'\u2500' * rule_len}{rs}")
    print(f"  {gy}{'Command'.ljust(22)}{rs}  {gy}Description{rs}")
    print(f"{dm}{'\u2500' * rule_len}{rs}")
    rows = [
        ("/tools, /help, /?", "Show this reference"),
        ("/read <path>", "Print file contents (byte & line caps apply)"),
        ("/ls [path]", "List directory entries (default: project root)"),
        ("/find <text>", "Search file contents; optional --glob pattern"),
        ("/tree [depth]", "Directory tree (depth 0\u20135, default 2)"),
        ("/compile <file.py>", "Syntax-check a Python file"),
        ("/learn", "Concise study notes for beginners"),
    ]
    col = max(len(cmd) for cmd, _ in rows) + 1
    for cmd, desc in rows:
        print(f"  {wt}{cmd.ljust(col)}{rs}{gy}{desc}{rs}")
    print(f"{dm}{'\u2500' * rule_len}{rs}")
    print(
        f"  {gy}Paths are resolved under the project root (and {lb}ABADDON_WRITE_ROOT{rs}{gy}, if set).{rs}"
    )


def _cmd_read(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    gray, reset, white = t["gray"], t["reset"], t["white"]
    if not args:
        print(f"{gray}Usage: /read <path>{reset}")
        return
    raw = args[0]
    p = _resolve_under_roots(raw, roots)
    if p is None or not p.is_file():
        print(f"{t['red']}✗{reset} {gray}Not found or not allowed:{reset} {white}{raw!r}{reset}")
        return
    try:
        data = p.read_bytes()
    except OSError as e:
        print(f"{t['red']}✗{reset} {gray}{e}{reset}")
        return
    if len(data) > _READ_MAX_BYTES:
        print(f"{gray}File too large; showing first {_READ_MAX_BYTES} bytes.{reset}")
        data = data[:_READ_MAX_BYTES]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > _READ_MAX_LINES:
        print(f"{gray}(truncated to {_READ_MAX_LINES} lines){reset}")
        lines = lines[:_READ_MAX_LINES]
    print(f"{gray}── {white}{p}{gray} ──{reset}")
    body = "\n".join(lines)
    if body:
        print(f"{t['dim']}{body}{reset}")


def _cmd_ls(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    gray, reset, white = t["gray"], t["reset"], t["white"]
    raw = args[0] if args else "."
    p = _resolve_under_roots(raw, roots)
    if p is None or not p.is_dir():
        print(f"{t['red']}✗{reset} {gray}Not a directory or not allowed:{reset} {white}{raw!r}{reset}")
        return
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except OSError as e:
        print(f"{t['red']}✗{reset} {gray}{e}{reset}")
        return
    print(f"{gray}── {white}{p}{gray} ──{reset}")
    for e in entries:
        mark = "/" if e.is_dir() else ""
        print(f"  {e.name}{mark}")


def _cmd_find(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    gray, reset, white = t["gray"], t["reset"], t["white"]
    if not args:
        print(f"{gray}Usage: /find <text> | /find several words --glob *.py{reset}")
        return
    if "--glob" in args:
        i = args.index("--glob")
        needle = " ".join(args[:i]).strip()
        glob_pat = args[i + 1] if i + 1 < len(args) else "*"
    elif len(args) == 1:
        needle, glob_pat = args[0], "*"
    elif len(args) == 2 and ("*" in args[1] or "?" in args[1]):
        needle, glob_pat = args[0], args[1]
    else:
        needle = " ".join(args)
        glob_pat = "*"
    hits = 0
    root = roots[0]
    print(f"{gray}Searching for {needle!r} (files: {glob_pat}) under {root}{reset}")
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if not fnmatch.fnmatch(name, glob_pat):
                continue
            fp = Path(dirpath) / name
            try:
                if needle not in fp.read_text(encoding="utf-8", errors="ignore"):
                    continue
            except OSError:
                continue
            rel = fp.relative_to(root)
            print(f"  {white}{rel}{reset}")
            hits += 1
            if hits >= _FIND_MAX_HITS:
                print(f"{gray}(stopped at {_FIND_MAX_HITS} hits){reset}")
                return
    if hits == 0:
        print(f"{gray}No matches.{reset}")


def _cmd_tree(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    gray, reset, white = t["gray"], t["reset"], t["white"]
    depth = _TREE_MAX_DEPTH_DEFAULT
    if args:
        try:
            depth = max(0, min(5, int(args[0])))
        except ValueError:
            print(f"{gray}Depth must be 0–5{reset}")
            return
    root = roots[0]

    def walk(p: Path, prefix: str, d: int) -> None:
        if d > depth:
            return
        try:
            kids = sorted(
                [x for x in p.iterdir() if x.name not in _SKIP_DIRS],
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except OSError:
            return
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            arm = "└── " if last else "├── "
            print(f"{prefix}{arm}{kid.name}{'/' if kid.is_dir() else ''}")
            if kid.is_dir() and d < depth:
                ext = "    " if last else "│   "
                walk(kid, prefix + ext, d + 1)

    print(f"{gray}── tree {white}{root}{gray} (depth {depth}) ──{reset}")
    print(f"{white}{root.name}/{reset}")
    walk(root, "", 0)


def _cmd_compile(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    gray, reset, white, green, red = t["gray"], t["reset"], t["white"], t["green"], t["red"]
    if not args:
        print(f"{gray}Usage:{reset} {white}/compile <file.py>{reset}")
        return
    p = _resolve_under_roots(args[0], roots)
    if p is None or not p.is_file():
        print(f"{red}✗{reset} {gray}Not found or not allowed:{reset} {white}{args[0]!r}{reset}")
        return
    if p.suffix.lower() != ".py":
        print(f"{red}✗{reset} {gray}Only .py files are supported.{reset}")
        return
    try:
        py_compile.compile(str(p), doraise=True)
        print(f"{green}✓{reset} {gray}Syntax OK{reset} {white}{p}{reset}")
    except py_compile.PyCompileError as e:
        print(f"{red}✗{reset} {gray}{e}{reset}")


def _cmd_learn(t: dict[str, str]) -> None:
    gray, reset, white = t["gray"], t["reset"], t["white"]
    tips = f"""
{gray}── Quick study sheet ──{reset}
• {white}Variable{reset}: {gray}name bound to a value (`x = 1`). Python checks types at runtime.{reset}
• {white}Function{reset}: {gray}reusable block (`def f(a): return a + 1`).{reset}
• {white}Iterate{reset}: {gray}prefer `for item in items:` over manual indices.{reset}
• {white}Errors{reset}: {gray}read tracebacks bottom-up; last line is the exception.{reset}
• {white}Debug{reset}: {gray}minimize the repro; use `/read` and `/find` in this CLI.{reset}
• {white}Next{reset}: {gray}paste a snippet and ask for an explanation or fix.{reset}
""".strip()
    print(tips)


def handle_slash_command(user_input: str, project_root: Path, colors: dict[str, str]) -> bool:
    """
    If user_input is a /command, run it and return True (caller should skip LLM).
    """
    parsed = _split_command(user_input)
    if parsed is None:
        return False
    cmd, args = parsed
    t = _theme(colors)
    roots = _tool_roots(project_root)

    if cmd in ("", "tools", "help", "?"):
        print_tools_help(colors)
        return True
    if cmd == "read":
        _cmd_read(args, roots, t)
        return True
    if cmd == "ls":
        _cmd_ls(args, roots, t)
        return True
    if cmd == "find":
        _cmd_find(args, roots, t)
        return True
    if cmd == "tree":
        _cmd_tree(args, roots, t)
        return True
    if cmd == "compile":
        _cmd_compile(args, roots, t)
        return True
    if cmd == "learn":
        _cmd_learn(t)
        return True

    print(f"{t['red']}Unknown command{t['reset']} {t['white']}/{cmd}{t['reset']}{t['gray']} · try {t['white']}/tools{t['reset']}")
    return True
