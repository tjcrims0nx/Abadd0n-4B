"""
Agent tool implementations inspired by OpenClaw.
Tools: read_file, write_file, list_dir, find_in_files, run_bash, compile_python.
Returns structured results for tool streaming / RPC.
"""

from __future__ import annotations

import fnmatch
import os
import py_compile
import shlex
import subprocess
from pathlib import Path

_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "venv", "venv_win", "venv_wsl",
    "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "unsloth_compiled_cache", "lora_model", "outputs", ".cursor",
})

_READ_MAX_BYTES = 400_000
_READ_MAX_LINES = 400
_FIND_MAX_HITS = 40


def _resolve_path(raw: str, roots: list[Path]) -> Path | None:
    raw = (raw or "").strip()
    if not raw or "\x00" in raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        cand = p.resolve()
    else:
        cand = (roots[0] / p).resolve()
    allow_out = os.environ.get("ABADDON_ALLOW_WRITES_OUTSIDE_ROOT", "").lower() in ("1", "true", "yes")
    if allow_out:
        return cand
    for root in roots:
        try:
            cand.relative_to(root)
            return cand
        except ValueError:
            continue
    return None


def read_file(path: str, project_root: Path, roots: list[Path] | None = None) -> dict:
    """Read file contents. Returns {ok, content?, error?, path}."""
    roots = roots or [project_root]
    p = _resolve_path(path, roots)
    if p is None or not p.is_file():
        return {"ok": False, "error": f"Not found or not allowed: {path}"}
    try:
        data = p.read_bytes()
    except OSError as e:
        return {"ok": False, "error": str(e)}
    if len(data) > _READ_MAX_BYTES:
        data = data[:_READ_MAX_BYTES]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > _READ_MAX_LINES:
        lines = lines[:_READ_MAX_LINES]
    return {"ok": True, "content": "\n".join(lines), "path": str(p)}


def write_file(path: str, content: str, project_root: Path, roots: list[Path] | None = None) -> dict:
    """Write file. Returns {ok, path?, error?}. Caller must validate path via resolve_safe_write_path."""
    roots = roots or [project_root]
    p = _resolve_path(path, roots)
    if p is None:
        return {"ok": False, "error": f"Path not allowed: {path}"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.strip("\n\r"), encoding="utf-8", newline="\n")
        return {"ok": True, "path": str(p)}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def list_dir(path: str, project_root: Path, roots: list[Path] | None = None) -> dict:
    """List directory. Returns {ok, entries?, error?}."""
    roots = roots or [project_root]
    raw = path.strip() or "."
    p = _resolve_path(raw, roots)
    if p is None or not p.is_dir():
        return {"ok": False, "error": f"Not a directory or not allowed: {raw}"}
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except OSError as e:
        return {"ok": False, "error": str(e)}
    out = []
    for e in entries:
        out.append({"name": e.name, "dir": e.is_dir()})
    return {"ok": True, "path": str(p), "entries": out}


def find_in_files(
    needle: str,
    glob_pat: str = "*",
    project_root: Path | None = None,
    roots: list[Path] | None = None,
) -> dict:
    """Search file contents. Returns {ok, hits?, error?}."""
    if not project_root and roots:
        project_root = roots[0]
    if not project_root:
        return {"ok": False, "error": "No project root"}
    roots = roots or [project_root]
    root = roots[0]
    hits = []
    try:
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
                hits.append(str(fp.relative_to(root)))
                if len(hits) >= _FIND_MAX_HITS:
                    break
            if len(hits) >= _FIND_MAX_HITS:
                break
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "hits": hits}


def run_bash(
    cmd: str,
    cwd: Path | None = None,
    timeout_s: int = 30,
) -> dict:
    """Run shell command. Returns {ok, stdout?, stderr?, exit_code?, error?}."""
    try:
        proc = subprocess.run(
            cmd if os.name != "nt" else ["cmd", "/c", cmd],
            shell=os.name != "nt",
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": f"Timeout after {timeout_s}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def compile_python(path: str, project_root: Path, roots: list[Path] | None = None) -> dict:
    """Syntax-check Python file. Returns {ok, error?}."""
    roots = roots or [project_root]
    p = _resolve_path(path, roots)
    if p is None or not p.is_file():
        return {"ok": False, "error": f"Not found: {path}"}
    if p.suffix.lower() != ".py":
        return {"ok": False, "error": "Only .py files supported"}
    try:
        py_compile.compile(str(p), doraise=True)
        return {"ok": True}
    except py_compile.PyCompileError as e:
        return {"ok": False, "error": str(e)}


def tool_roots(project_root: Path) -> list[Path]:
    roots = [project_root.resolve()]
    extra = os.environ.get("ABADDON_WRITE_ROOT", "").strip()
    if extra:
        roots.append(Path(extra).resolve())
    return roots


def apply_patch(patch_text: str, project_root: Path, roots: list[Path] | None = None) -> dict:
    """
    Apply OpenClaw-style patch (*** Begin Patch ... *** End Patch).
    Returns {ok, applied?, errors?}.
    See: https://docs.openclaw.ai/tools/apply-patch
    """
    roots = roots or [project_root]
    if not patch_text.strip():
        return {"ok": False, "error": "empty patch"}
    if "*** Begin Patch" not in patch_text or "*** End Patch" not in patch_text:
        return {"ok": False, "error": "missing *** Begin Patch / *** End Patch"}
    block = patch_text.split("*** End Patch")[0].split("*** Begin Patch")[-1].strip()
    applied = []
    errors = []
    i = 0
    lines = block.split("\n")
    while i < len(lines):
        line = lines[i]
        if line.startswith("*** Add File:"):
            path = line.replace("*** Add File:", "").strip()
            content_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("***"):
                if lines[i].startswith("+"):
                    content_lines.append(lines[i][1:])
                i += 1
            p = _resolve_path(path, roots)
            if p is None:
                errors.append(f"path not allowed: {path}")
            else:
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("\n".join(content_lines), encoding="utf-8", newline="\n")
                    applied.append(str(p))
                except OSError as e:
                    errors.append(f"{path}: {e}")
            continue
        if line.startswith("*** Update File:"):
            path = line.replace("*** Update File:", "").strip()
            p = _resolve_path(path, roots)
            if p is None or not p.exists():
                errors.append(f"path not found/not allowed: {path}")
                i += 1
                while i < len(lines) and not lines[i].startswith("***"):
                    i += 1
                continue
            i += 1
            old_block, new_block = [], []
            while i < len(lines) and not lines[i].startswith("***"):
                ln = lines[i]
                if ln.startswith("-") and not ln.startswith("---"):
                    old_block.append(ln[1:])
                elif ln.startswith("+"):
                    new_block.append(ln[1:])
                i += 1
            try:
                orig = p.read_text(encoding="utf-8")
                old_str = "\n".join(old_block)
                new_str = "\n".join(new_block)
                if old_str in orig:
                    out = orig.replace(old_str, new_str, 1)
                    p.write_text(out, encoding="utf-8", newline="\n")
                    applied.append(str(p))
                else:
                    errors.append(f"{path}: hunk not found (old block mismatch)")
            except OSError as e:
                errors.append(f"{path}: {e}")
            continue
        if line.startswith("*** Delete File:"):
            path = line.replace("*** Delete File:", "").strip()
            p = _resolve_path(path, roots)
            if p and p.exists():
                try:
                    p.unlink()
                    applied.append(f"deleted {p}")
                except OSError as e:
                    errors.append(f"{path}: {e}")
            i += 1
            continue
        i += 1
    return {"ok": len(errors) == 0, "applied": applied, "errors": errors if errors else None}
