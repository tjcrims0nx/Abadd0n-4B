"""
Local coding helpers for the Abadd0n CLI: read/list/search/compile under the project tree.
No network APIs — stdlib + pathlib only.
Design: https://yannglt.com/writing/designing-for-command-line-interface
"""

from __future__ import annotations

import fnmatch
import os
import py_compile
import shutil
import shlex
import sys
from pathlib import Path

from cli_theme import BOLD, BOX_H, DIM, GRAY, LABEL, RESET, WHITE, error, muted, success

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
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    rule_len = max(48, min(72, width - 6))
    rule = f"{DIM}{BOX_H * rule_len}{RESET}"
    print(f"\n{LABEL}ABADD0N{RESET} {GRAY}local toolkit{RESET} {DIM}\u2014{RESET} {GRAY}offline commands{RESET}")
    print(rule)
    print(f"  {GRAY}{'Command':<24}{RESET}  {GRAY}Description{RESET}")
    print(rule)
    rows = [
        ("/tools, /help, /?", "Show this reference"),
        ("/read <path>", "Print file contents"),
        ("/ls [path]", "List directory"),
        ("/find <text>", "Search contents; --glob pattern"),
        ("/tree [depth]", "Directory tree (0\u20135)"),
        ("/compile <file.py>", "Syntax-check Python"),
        ("/grant", "Grant/revoke system file access"),
        ("/new", "Start new chat thread"),
        ("/docs <query>", "Search OpenClaw docs (docs.openclaw.ai)"),
        ("/fetch <url>", "Fetch URL (web_fetch)"),
        ("/patch <file>", "Apply OpenClaw-style patch file"),
        ("/skills", "Search bar for ClawHub skills"),
        ("/skills search [q]", "Search or browse clawhub.ai"),
        ("/skills install <slug>", "Install skill into agent (add --global for all projects)"),
        ("/learn", "Study notes"),
    ]
    col = max(len(cmd) for cmd, _ in rows) + 1
    for cmd, desc in rows:
        print(f"  {WHITE}{cmd:<{col}}{RESET}{GRAY}{desc}{RESET}")
    print(rule)
    print(f"  {GRAY}Paths under project root (and ABADDON_WRITE_ROOT){RESET}\n")


def _cmd_read(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    if not args:
        print(muted("Usage: /read <path>"))
        return
    raw = args[0]
    p = _resolve_under_roots(raw, roots)
    if p is None or not p.is_file():
        print(error("Not found or not allowed", raw))
        return
    try:
        data = p.read_bytes()
    except OSError as e:
        print(error("Read failed", str(e)))
        return
    if len(data) > _READ_MAX_BYTES:
        print(muted(f"File large; showing first {_READ_MAX_BYTES} bytes"))
        data = data[:_READ_MAX_BYTES]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > _READ_MAX_LINES:
        print(muted(f"(truncated to {_READ_MAX_LINES} lines)"))
        lines = lines[:_READ_MAX_LINES]
    print(f"\n  {GRAY}\u2500\u2500{RESET} {WHITE}{p}{RESET} {GRAY}\u2500\u2500{RESET}")
    body = "\n".join(lines)
    if body:
        print(f"  {DIM}{body}{RESET}")


def _cmd_ls(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    raw = args[0] if args else "."
    p = _resolve_under_roots(raw, roots)
    if p is None or not p.is_dir():
        print(error("Not a directory or not allowed", raw))
        return
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except OSError as e:
        print(error("List failed", str(e)))
        return
    print(f"\n  {GRAY}\u2500\u2500{RESET} {WHITE}{p}{RESET} {GRAY}\u2500\u2500{RESET}")
    for e in entries:
        mark = "/" if e.is_dir() else ""
        print(f"  {e.name}{mark}")


def _cmd_find(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    if not args:
        print(muted("Usage: /find <text> | /find text --glob *.py"))
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
    print(muted(f"Searching for {needle!r} (files: {glob_pat}) under {root}"))
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
            print(f"  {WHITE}{rel}{RESET}")
            hits += 1
            if hits >= _FIND_MAX_HITS:
                print(muted(f"(stopped at {_FIND_MAX_HITS} hits)"))
                return
    if hits == 0:
        print(muted("No matches."))


def _cmd_tree(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    depth = _TREE_MAX_DEPTH_DEFAULT
    if args:
        try:
            depth = max(0, min(5, int(args[0])))
        except ValueError:
            print(muted("Depth must be 0\u20135"))
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
            arm = "\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 "
            print(f"{prefix}{arm}{kid.name}{'/' if kid.is_dir() else ''}")
            if kid.is_dir() and d < depth:
                ext = "    " if last else "\u2502   "
                walk(kid, prefix + ext, d + 1)

    print(f"\n  {GRAY}\u2500\u2500{RESET} {WHITE}{root}{RESET} {GRAY}(depth {depth}){RESET}")
    print(f"  {WHITE}{root.name}/{RESET}")
    walk(root, "", 0)


def _cmd_compile(args: list[str], roots: list[Path], t: dict[str, str]) -> None:
    if not args:
        print(muted("Usage: /compile <file.py>"))
        return
    p = _resolve_under_roots(args[0], roots)
    if p is None or not p.is_file():
        print(error("Not found or not allowed", args[0]))
        return
    if p.suffix.lower() != ".py":
        print(error("Only .py files are supported", ""))
        return
    try:
        py_compile.compile(str(p), doraise=True)
        print(success("Syntax OK", str(p)))
    except py_compile.PyCompileError as e:
        print(error("Compile failed", str(e)))


def _cmd_learn(t: dict[str, str]) -> None:
    tips = f"""
  {GRAY}\u2500\u2500 Quick study sheet{RESET}
  {DIM}\u2022{RESET} {WHITE}Variable{RESET} {DIM}name bound to value (`x = 1`){RESET}
  {DIM}\u2022{RESET} {WHITE}Function{RESET} {DIM}reusable block (`def f(a): return a + 1`){RESET}
  {DIM}\u2022{RESET} {WHITE}Iterate{RESET} {DIM}prefer `for item in items:`{RESET}
  {DIM}\u2022{RESET} {WHITE}Errors{RESET} {DIM}read tracebacks bottom-up{RESET}
  {DIM}\u2022{RESET} {WHITE}Debug{RESET} {DIM}use /read and /find{RESET}
  {DIM}\u2022{RESET} {WHITE}Next{RESET} {DIM}paste a snippet and ask{RESET}

  {GRAY}\u2500\u2500 Coding experience - languages{RESET}
  {DIM}\u2022{RESET} {WHITE}Python{RESET} {DIM}versatile, beginner-friendly; AI, ML, web{RESET}
  {DIM}\u2022{RESET} {WHITE}JavaScript{RESET} {DIM}core for front/back-end; ~99% of sites{RESET}
  {DIM}\u2022{RESET} {WHITE}Java{RESET} {DIM}object-oriented; enterprise, Android{RESET}
  {DIM}\u2022{RESET} {WHITE}C#{RESET} {DIM}.NET; Unity, Windows, enterprise{RESET}
  {DIM}\u2022{RESET} {WHITE}C++{RESET} {DIM}high-performance; systems, Unreal, speed-critical{RESET}
  {DIM}\u2022{RESET} {WHITE}Go{RESET} {DIM}simple & efficient; cloud, DevOps, scalable{RESET}
  {DIM}\u2022{RESET} {WHITE}Rust{RESET} {DIM}memory safety, concurrency; systems, blockchain{RESET}
  {DIM}\u2022{RESET} {WHITE}TypeScript{RESET} {DIM}JS + static typing; large web apps{RESET}
  {DIM}\u2022{RESET} {WHITE}SQL{RESET} {DIM}query/manage relational data{RESET}
  {DIM}\u2022{RESET} {WHITE}PHP{RESET} {DIM}server-side web; WordPress, CMS{RESET}
  {DIM}\u2022{RESET} {WHITE}Swift{RESET} {DIM}Apple (iOS/macOS); safe, fast{RESET}
  {DIM}\u2022{RESET} {WHITE}Kotlin{RESET} {DIM}modern Java alternative; preferred for Android{RESET}
  {DIM}\u2022{RESET} {WHITE}HTML/CSS{RESET} {DIM}markup & styles for web content{RESET}
  {DIM}\u2022{RESET} {WHITE}Ruby{RESET} {DIM}dynamic; Rails, developer happiness{RESET}
  {DIM}\u2022{RESET} {WHITE}C{RESET} {DIM}low-level; OS, embedded, foundational{RESET}
""".strip()
    print(tips)


def _cmd_grant(colors: dict[str, str]) -> None:
    """Grant or revoke system file access (ABADDON_ALLOW_WRITES_OUTSIDE_ROOT)."""
    current = os.environ.get("ABADDON_ALLOW_WRITES_OUTSIDE_ROOT", "").lower() in ("1", "true", "yes")
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    rule_len = max(48, min(72, width - 6))
    rule = f"{DIM}{BOX_H * rule_len}{RESET}"
    label = colors.get("label", LABEL)
    print(f"\n{rule}")
    print(f"  {BOLD}{label}Grant system file access{RESET} {DIM}\u2014{RESET} {GRAY}allow writes outside project root{RESET}")
    print(rule)
    print(f"  {GRAY}Current{RESET} {WHITE}{'granted' if current else 'revoked'}{RESET}")
    print(f"  {GRAY}Danger{RESET} Granting allows writes to any path on disk.")
    print(rule)
    prompt = f"  {GRAY}Grant [y/n]{RESET} " if not current else f"  {GRAY}Revoke [y/n]{RESET} "
    try:
        raw = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(muted("Cancelled"))
        return
    if raw in ("y", "yes"):
        if current:
            os.environ.pop("ABADDON_ALLOW_WRITES_OUTSIDE_ROOT", None)
            print(success("Revoked", "writes restricted to project root"))
        else:
            os.environ["ABADDON_ALLOW_WRITES_OUTSIDE_ROOT"] = "1"
            print(success("Granted", "writes can target any path (use with caution)"))
    else:
        print(muted("No change"))
    print(rule)


def _cmd_skills(args: list[str], project_root: Path, colors: dict[str, str]) -> None:
    """ClawHub: /skills [search [q]] | /skills install <slug> [--global] | /skills browse"""
    try:
        from core.clawhub import search_skills, download_skill
    except ImportError as e:
        print(error("ClawHub", f"core.clawhub not available: {e}"))
        return
    sub = (args[0].lower() if args else "")
    use_global = "--global" in args or "-g" in args
    args = [a for a in args if a not in ("--global", "-g")]
    if use_global:
        skills_dir = os.environ.get("ABADDON_SKILLS_DIR", "").strip()
        skills_dir = Path(skills_dir) if skills_dir else Path.home() / ".abaddon" / "skills"
    else:
        skills_dir = project_root / "skills"
    label = colors.get("label", LABEL)
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    rule_len = max(48, min(72, width - 6))
    rule = f"{DIM}{BOX_H * rule_len}{RESET}"

    def _run_search(q: str, limit: int = 50) -> bool:
        r = search_skills(q, limit=limit)
        if "error" in r:
            print(error("ClawHub search", r["error"]))
            return False
        results = r.get("results", [])
        if not results:
            print(muted("No skills found"))
            return False
        print(f"\n{rule}")
        print(f"  {BOLD}{label}ClawHub{RESET} {DIM}\u2014{RESET} {GRAY}clawhub.ai (limit {min(limit, len(results))}){RESET}")
        print(rule)
        for i, s in enumerate(results[:limit], 1):
            slug = s.get("slug", "?")
            name = s.get("displayName", slug)
            summary = (s.get("summary", "") or "")[:70] + ("…" if len(s.get("summary", "") or "") > 70 else "")
            url = f"https://clawhub.ai/skills/{slug}"
            print(f"  {i}. {WHITE}{name}{RESET} {DIM}({slug}){RESET}")
            print(f"     {GRAY}{summary}{RESET}")
            print(f"     {DIM}{url}{RESET}")
        print(rule)
        print(muted("Install: /skills install <slug>  |  /skills install <slug> --global"))
        return True

    if sub == "search":
        q = " ".join(args[1:]).strip() if len(args) >= 2 else ""
        if not q:
            try:
                q = input(f"  {GRAY}Search skills (query): {RESET}").strip() or "the"
            except (EOFError, KeyboardInterrupt):
                print(muted("\nCancelled"))
                return
        _run_search(q)
        return
    if sub == "browse":
        _run_search("the", limit=50)
        return
    if sub == "install":
        slug_arg = args[1] if len(args) >= 2 else ""
        if not slug_arg:
            try:
                slug_arg = input(f"  {GRAY}Install skill (slug): {RESET}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(muted("\nCancelled"))
                return
        if not slug_arg:
            print(muted("Usage: /skills install <slug>"))
            return
        slug = slug_arg
        Path(skills_dir).mkdir(parents=True, exist_ok=True)
        r = download_skill(slug, Path(skills_dir))
        if not r.get("ok"):
            print(error("ClawHub install", r.get("error", "failed")))
            return
        dest = "agent (global)" if use_global else "agent (project)"
        print(success("Installed", f"{slug} -> {dest}"))
        if r.get("skill_md"):
            preview = (r["skill_md"] or "")[:200].replace("\n", " ")
            print(muted(f"SKILL.md: {preview}…"))
        return
    if sub == "" or sub == "help":
        q = ""
        try:
            q = input(f"  {GRAY}Search skills (query, or Enter to browse): {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            pass
        if q or True:
            _run_search(q or "the", limit=50)
        return
    # default: help
    print(f"\n{rule}")
    print(f"  {BOLD}{label}ClawHub skills{RESET} {DIM}\u2014{RESET} clawhub.ai")
    print(rule)
    print(f"  {GRAY}/skills{RESET}              Search bar (prompt for query)")
    print(f"  {GRAY}/skills search [q]{RESET}  Search or prompt for query")
    print(f"  {GRAY}/skills browse{RESET}      Browse ~50 skills")
    print(f"  {GRAY}/skills install <slug>{RESET}  Install to agent ({project_root / 'skills'})")
    print(f"  {GRAY}/skills install <slug> --global{RESET}  Install to global agent dir")
    if (project_root / "skills").exists():
        installed = [d.name for d in (project_root / "skills").iterdir() if d.is_dir()]
        if installed:
            print(f"  {GRAY}Installed{RESET} {', '.join(installed[:8])}{'…' if len(installed) > 8 else ''}")
    print(rule)


def _cmd_docs(args: list[str], colors: dict[str, str]) -> None:
    """Search OpenClaw docs: /docs <query>"""
    try:
        from core.docs_openclaw import search_docs
    except ImportError as e:
        print(error("Docs", f"core.docs_openclaw not available: {e}"))
        return
    q = " ".join(args).strip() if args else ""
    if not q:
        print(muted("Usage: /docs <query>  (e.g. /docs exec, /docs browser)"))
        print(muted("Searches https://docs.openclaw.ai/llms.txt"))
        return
    r = search_docs(q, limit=15)
    if "error" in r:
        print(error("Docs search", r["error"]))
        return
    results = r.get("results", [])
    if not results:
        print(muted("No matches"))
        return
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    rule_len = max(48, min(72, width - 6))
    rule = f"{DIM}{BOX_H * rule_len}{RESET}"
    label = colors.get("label", LABEL)
    print(f"\n{rule}")
    print(f"  {BOLD}{label}OpenClaw docs{RESET} {DIM}\u2014{RESET} {GRAY}docs.openclaw.ai{RESET}")
    print(rule)
    for i, item in enumerate(results[:15], 1):
        title = item.get("title", "?")
        url = item.get("url", "")
        print(f"  {i}. {WHITE}{title}{RESET}")
        print(f"     {DIM}{url}{RESET}")
    print(rule)


def _cmd_fetch(args: list[str], colors: dict[str, str]) -> None:
    """Fetch URL: /fetch <url> (web_fetch)"""
    try:
        from core.web_fetch import web_fetch
    except ImportError as e:
        print(error("Fetch", f"core.web_fetch not available: {e}"))
        return
    if not args:
        print(muted("Usage: /fetch <url>"))
        return
    url = args[0]
    r = web_fetch(url)
    if not r.get("ok"):
        print(error("Fetch", r.get("error", "failed")))
        return
    content = r.get("content", "")
    label = colors.get("label", LABEL)
    print(f"\n{label}Fetch{RESET} {DIM}\u2014{RESET} {url}")
    print(f"{DIM}{BOX_H * 60}{RESET}")
    for line in content.split("\n")[:80]:
        print(f"  {line[:100]}{'…' if len(line) > 100 else ''}")
    if content.count("\n") >= 80:
        print(muted("… (truncated)"))


def _cmd_patch(args: list[str], project_root: Path, roots: list[Path], colors: dict[str, str]) -> None:
    """Apply patch: /patch <path>"""
    try:
        from core.tools import apply_patch
    except ImportError as e:
        print(error("Patch", f"core.tools not available: {e}"))
        return
    if not args:
        print(muted("Usage: /patch <path>  (file containing *** Begin Patch ... *** End Patch)"))
        return
    raw = args[0]
    p = _resolve_under_roots(raw, roots)
    if p is None or not p.is_file():
        print(error("Patch", f"not found or not allowed: {raw}"))
        return
    try:
        patch_text = p.read_text(encoding="utf-8")
    except OSError as e:
        print(error("Patch", str(e)))
        return
    r = apply_patch(patch_text, project_root, roots)
    if r.get("errors"):
        for err in r["errors"]:
            print(error("Patch", err))
    if r.get("applied"):
        print(success("Applied", ", ".join(r["applied"])))


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
    if cmd == "grant":
        _cmd_grant(colors)
        return True
    if cmd == "skills":
        _cmd_skills(args, project_root, colors)
        return True
    if cmd == "docs":
        _cmd_docs(args, colors)
        return True
    if cmd == "fetch":
        _cmd_fetch(args, colors)
        return True
    if cmd == "patch":
        _cmd_patch(args, project_root, roots, colors)
        return True

    print(error("Unknown command", f"/{cmd} \u2014 try /tools"))
    return True
