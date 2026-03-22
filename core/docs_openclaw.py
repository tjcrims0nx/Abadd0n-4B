"""
OpenClaw docs search — mirrors openclaw docs.
Fetches https://docs.openclaw.ai/llms.txt and searches for matching pages.

Reference: https://docs.openclaw.ai/cli/docs
"""

from __future__ import annotations

import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DOCS_INDEX_URL = "https://docs.openclaw.ai/llms.txt"
_CACHE: str | None = None


def _fetch_index() -> str:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        req = Request(DOCS_INDEX_URL, headers={"User-Agent": "Abadd0n/1.0"})
        with urlopen(req, timeout=15) as r:
            _CACHE = r.read().decode("utf-8", errors="replace")
        return _CACHE
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to fetch docs index: {e}") from e


def search_docs(query: str, limit: int = 15) -> dict:
    """
    Search OpenClaw docs index. Returns {results: [{title, url}], error?}.
    Matches lines containing the query (case-insensitive).
    """
    query = (query or "").strip()
    if not query:
        return {"results": [], "error": "query required"}
    try:
        index = _fetch_index()
    except RuntimeError as e:
        return {"results": [], "error": str(e)}
    q_lower = query.lower()
    pattern = re.compile(re.escape(q_lower), re.IGNORECASE)
    results = []
    seen = set()
    for line in index.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if pattern.search(line):
            m = re.match(r"^-\s*\[([^\]]+)\]\(([^)]+)\)", line)
            if m:
                title, url = m.group(1), m.group(2)
                if url not in seen:
                    seen.add(url)
                    results.append({"title": title, "url": url})
                    if len(results) >= limit:
                        break
            else:
                if "http" in line and line not in seen:
                    results.append({"title": line[:80], "url": line})
                    seen.add(line)
                    if len(results) >= limit:
                        break
    return {"results": results}
