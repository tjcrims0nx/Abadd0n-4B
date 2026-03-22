"""
web_fetch — fetch URL and extract readable content.
Inspired by OpenClaw web_fetch: https://docs.openclaw.ai/tools/web

No API key required. Basic HTTP GET + simple HTML→text extraction.
"""

from __future__ import annotations

import re
from html import unescape
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

MAX_CHARS = 50000
MAX_RESPONSE_BYTES = 500_000
TIMEOUT = 30


def web_fetch(url: str, max_chars: int = MAX_CHARS) -> dict:
    """
    Fetch URL and extract text. Returns {ok, content?, error?, url?}.
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "url required"}
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "error": "url must be http or https"}
    try:
        req = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Abadd0n/1.0)"},
        )
        with urlopen(req, timeout=TIMEOUT) as r:
            data = r.read(MAX_RESPONSE_BYTES)
    except HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}", "url": url}
    except URLError as e:
        return {"ok": False, "error": str(e.reason), "url": url}

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

    content = _extract_text(text)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n… (truncated)"
    return {"ok": True, "content": content, "url": url}


def _extract_text(html: str) -> str:
    """Simple HTML→text: strip tags, decode entities, collapse whitespace."""
    html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
