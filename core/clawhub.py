"""
ClawHub API client for OpenClaw skills.
API: https://clawhub.ai — search, resolve, download skills.

See: https://clawhub.ai, https://docs.openclaw.ai/tools/clawhub
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

CLAWHUB_API_BASE = os.environ.get("CLAWHUB_API_BASE", "https://clawhub.ai")


def _get(url: str) -> dict | bytes:
    """GET request; returns parsed JSON or raw bytes."""
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Abadd0n/1.0"})
    with urlopen(req, timeout=30) as r:
        data = r.read()
        ct = r.headers.get("Content-Type", "")
    if "application/json" in ct:
        return json.loads(data.decode("utf-8"))
    return data


def _get_stream(url: str):
    """GET request for binary (e.g. zip)."""
    req = Request(url, headers={"User-Agent": "Abadd0n/1.0"})
    return urlopen(req, timeout=60)


def search_skills(q: str, limit: int = 50) -> dict:
    """
    Vector search ClawHub skills. Returns {results: [{slug, displayName, summary, ...}], ...}.
    Empty q uses a broad browse query to surface many skills.
    """
    q = (q or "").strip()
    if not q:
        q = "the"  # broad browse term to surface many skills
    limit = max(1, min(100, limit))
    url = f"{CLAWHUB_API_BASE}/api/v1/search?q={_quote(q)}&limit={limit}"
    try:
        out = _get(url)
        if isinstance(out, dict):
            return out
        return {"results": [], "error": "unexpected response"}
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        return {"results": [], "error": str(e)}


def resolve_skill(slug: str) -> dict:
    """Resolve skill version. Returns {slug, version, ...} or {error}."""
    slug = (slug or "").strip().lower()
    if not slug:
        return {"error": "slug required"}
    url = f"{CLAWHUB_API_BASE}/api/v1/resolve?slug={slug}"
    try:
        out = _get(url)
        if isinstance(out, dict) and "error" not in out:
            return out
        return out if isinstance(out, dict) else {"error": "unexpected response"}
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        return {"error": str(e)}


def download_skill(slug: str, dest_dir: Path) -> dict:
    """
    Download skill zip and extract to dest_dir/<slug>/.
    Returns {ok, path?, skill_md?, error?}.
    """
    slug = (slug or "").strip().lower()
    if not slug:
        return {"ok": False, "error": "slug required"}
    url = f"{CLAWHUB_API_BASE}/api/v1/download?slug={slug}"
    try:
        with _get_stream(url) as r:
            data = r.read()
    except HTTPError as e:
        if e.code == 429:
            return {"ok": False, "error": "Rate limit exceeded — try again later"}
        return {"ok": False, "error": f"HTTP {e.code}"}
    except URLError as e:
        return {"ok": False, "error": str(e)}

    dest = dest_dir / slug
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(data)
            zip_path = f.name
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest)
        finally:
            try:
                os.unlink(zip_path)
            except OSError:
                pass
    except (zipfile.BadZipFile, OSError) as e:
        return {"ok": False, "error": str(e)}

    skill_md = dest / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8", errors="replace") if skill_md.exists() else ""
    return {"ok": True, "path": str(dest), "skill_md": content[:8000] if content else None}


def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


def _load_skills_from_dir(skills_dir: Path) -> list[str]:
    """Load skill contents from a directory. Returns list of (slug, text) tuples as formatted strings."""
    if not skills_dir.is_dir():
        return []
    parts = []
    for slug_dir in sorted(skills_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        skill_md = slug_dir / "SKILL.md"
        if skill_md.exists():
            try:
                text = skill_md.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    parts.append(f"\n\n--- ClawHub skill: {slug_dir.name} ---\n{text}")
            except OSError:
                pass
    return parts


def load_installed_skills(project_root: Path) -> str:
    """
    Load all installed ClawHub skills from project/skills and optionally ABADDON_SKILLS_DIR.
    Returns concatenated content for injection into agent context.
    """
    parts = []
    # Project skills (agent gets these by default)
    proj_skills = project_root / "skills"
    parts.extend(_load_skills_from_dir(proj_skills))
    # Global skills dir (available to agent across all projects)
    global_dir = os.environ.get("ABADDON_SKILLS_DIR", "").strip()
    if global_dir:
        parts.extend(_load_skills_from_dir(Path(global_dir)))
    if not parts:
        return ""
    return "\n\n[ClawHub installed skills — apply their instructions when relevant]\n" + "".join(parts)
