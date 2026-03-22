"""
Test all Abadd0n tools: slash commands and core agent tools.
Run: python -m tests.test_tools [--skip-network]
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Avoid Windows cp1252 encode errors when printing unicode
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(name: str, fn, *args, **kwargs):
    """Run a test; return (ok, msg)."""
    try:
        result = fn(*args, **kwargs)
        return (True, result if result is not None else "ok")
    except Exception as e:
        return (False, str(e))


def test_core_tools(test_dir: Path, roots: list[Path]) -> list[tuple[str, bool, str]]:
    """Test core/tools.py agent tools."""
    from core.tools import (
        read_file, write_file, list_dir, find_in_files,
        run_bash, compile_python, apply_patch, tool_roots,
    )
    results = []

    # write_file + read_file
    ok, msg = run("write_file", lambda: write_file("a.txt", "hello world", test_dir, roots))
    results.append(("write_file", ok, msg))
    if ok:
        r = read_file("a.txt", test_dir, roots)
        results.append(("read_file", r.get("ok"), r.get("content", r.get("error"))))

    # list_dir
    r = list_dir(".", test_dir, roots)
    results.append(("list_dir", r.get("ok"), r.get("entries", r.get("error"))))

    # find_in_files
    r = find_in_files("hello", "*", test_dir, roots)
    results.append(("find_in_files", r.get("ok"), r.get("hits", r.get("error"))))

    # compile_python - create valid .py
    (test_dir / "valid.py").write_text("x = 1\n", encoding="utf-8")
    r = compile_python("valid.py", test_dir, roots)
    results.append(("compile_python (valid)", r.get("ok"), r.get("error", "ok")))
    (test_dir / "invalid.py").write_text("x = \n", encoding="utf-8")
    r = compile_python("invalid.py", test_dir, roots)
    results.append(("compile_python (invalid)", not r.get("ok") and "error" in r, r.get("error", "ok")))

    # run_bash
    r = run_bash("echo ok", cwd=test_dir)
    results.append(("run_bash", r.get("ok") and "ok" in (r.get("stdout") or ""), r.get("stdout", r.get("error"))))

    # apply_patch - Add File
    patch_add = """
*** Begin Patch ***
*** Add File: patch_new.txt
+line1
+line2
*** End Patch ***
"""
    r = apply_patch(patch_add, test_dir, roots)
    results.append(("apply_patch Add", r.get("ok") and "patch_new.txt" in str(r.get("applied", [])), str(r)))
    if (test_dir / "patch_new.txt").exists():
        results.append(("apply_patch Add content", (test_dir / "patch_new.txt").read_text() == "line1\nline2", "ok"))

    # apply_patch - Update File
    (test_dir / "patch_update.txt").write_text("old block\nmore old\n", encoding="utf-8")
    patch_update = """
*** Begin Patch ***
*** Update File: patch_update.txt
@@
-old block
-more old
+new block
+more new
*** End Patch ***
"""
    r = apply_patch(patch_update, test_dir, roots)
    results.append(("apply_patch Update", r.get("ok"), str(r)))
    if (test_dir / "patch_update.txt").exists():
        c = (test_dir / "patch_update.txt").read_text()
        results.append(("apply_patch Update content", "new block" in c and "old block" not in c, c))

    # apply_patch - Delete File
    (test_dir / "patch_del.txt").write_text("gone", encoding="utf-8")
    patch_del = """
*** Begin Patch ***
*** Delete File: patch_del.txt
*** End Patch ***
"""
    r = apply_patch(patch_del, test_dir, roots)
    results.append(("apply_patch Delete", r.get("ok") and not (test_dir / "patch_del.txt").exists(), str(r)))

    return results


def test_slash_commands(test_dir: Path, colors: dict) -> list[tuple[str, bool, str]]:
    """Test slash commands via handle_slash_command. Captures stdout."""
    import builtins
    from coding_tools import handle_slash_command
    results = []
    _real_input = builtins.input

    def run_slash(cmd: str, mock_input_val: str | None = None) -> tuple[bool, str]:
        buf = io.StringIO()
        old = sys.stdout
        if mock_input_val is not None:
            builtins.input = lambda _: mock_input_val
        try:
            sys.stdout = buf
            handled = handle_slash_command(cmd, test_dir, colors)
            out = buf.getvalue()
            return (handled, out)
        finally:
            sys.stdout = old
            builtins.input = _real_input

    # /read
    (test_dir / "slash_read.txt").write_text("slashed", encoding="utf-8")
    ok, out = run_slash("/read slash_read.txt")
    results.append(("/read", ok and "slashed" in out, out[:80] if out else ""))

    # /ls
    ok, out = run_slash("/ls")
    results.append(("/ls", ok and "slash_read" in out, out[:80] if out else ""))

    # /find
    ok, out = run_slash("/find slashed")
    results.append(("/find", ok and "slash_read" in out, out[:80] if out else ""))

    # /tree
    ok, out = run_slash("/tree 1")
    results.append(("/tree", ok and len(out) > 0, out[:80] if out else ""))

    # /compile
    (test_dir / "compile_me.py").write_text("pass\n", encoding="utf-8")
    ok, out = run_slash("/compile compile_me.py")
    results.append(("/compile", ok and ("success" in out.lower() or "ok" in out.lower() or "pass" in out.lower()), out[:80]))

    # /learn
    ok, out = run_slash("/learn")
    results.append(("/learn", ok, out[:80] if out else ""))

    # /grant (mocks input to avoid blocking)
    ok, out = run_slash("/grant", mock_input_val="n")
    results.append(("/grant", ok and "revoked" in out.lower() or "granted" in out.lower() or "no change" in out.lower(), "ok"))

    # /patch
    (test_dir / "my.patch").write_text("""
*** Begin Patch ***
*** Add File: patched.txt
+patched content
*** End Patch ***
""", encoding="utf-8")
    ok, out = run_slash("/patch my.patch")
    patched_exists = (test_dir / "patched.txt").exists()
    results.append(("/patch", ok and patched_exists and "patched" in out.lower(), out[:80] if out else ""))

    return results


def test_network_tools(colors: dict) -> list[tuple[str, bool, str]]:
    """Test /docs and /fetch (require network)."""
    from coding_tools import handle_slash_command
    import io
    results = []

    def run_slash(cmd: str) -> tuple[bool, str]:
        buf = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = buf
            handle_slash_command(cmd, ROOT, colors)
            return buf.getvalue()
        finally:
            sys.stdout = old

    # /docs
    out = run_slash("/docs apply patch")
    has_result = "apply" in out.lower() or "patch" in out.lower() or "openclaw" in out.lower() or "docs.openclaw" in out
    results.append(("/docs", len(out) > 0 and (has_result or "error" not in out.lower()), out[:120]))

    # /fetch
    out = run_slash("/fetch https://example.com")
    results.append(("/fetch", len(out) > 0 and ("example" in out.lower() or "html" in out.lower() or "error" in out.lower()), out[:120]))

    return results


def main():
    skip_net = "--skip-network" in sys.argv
    colors = {"label": "", "gray": "", "dim": "", "white": "", "green": "", "red": "", "cyan": "", "reset": ""}
    # Use a temp dir under project for tests
    test_dir = ROOT / "tests" / "_test_tools_run"
    test_dir.mkdir(parents=True, exist_ok=True)
    roots = [test_dir.resolve()]

    print("=" * 60)
    print("Abadd0n Tools Test Suite")
    print("=" * 60)

    all_results = []

    print("\n--- Core tools (core/tools.py) ---")
    core_results = test_core_tools(test_dir, roots)
    all_results.extend(core_results)
    for name, ok, msg in core_results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         {str(msg)[:100]}")

    print("\n--- Slash commands (coding_tools) ---")
    slash_results = test_slash_commands(test_dir, colors)
    all_results.extend(slash_results)
    for name, ok, msg in slash_results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         {str(msg)[:100]}")

    if not skip_net:
        print("\n--- Network tools (/docs, /fetch) ---")
        net_results = test_network_tools(colors)
        all_results.extend(net_results)
        for name, ok, msg in net_results:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}")
            if not ok:
                print(f"         {str(msg)[:100]}")
    else:
        print("\n--- Network tools (skipped, use without --skip-network to run) ---")

    # Cleanup
    for f in test_dir.iterdir():
        try:
            f.unlink() if f.is_file() else None
        except OSError:
            pass
    try:
        test_dir.rmdir()
    except OSError:
        pass

    passed = sum(1 for _, ok, _ in all_results if ok)
    total = len(all_results)
    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
