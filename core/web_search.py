"""
Web search via Google. No API key required.
Uses googlesearch-python (scrapes Google).
"""

from __future__ import annotations

MAX_RESULTS = 8


def web_search(query: str, max_results: int = MAX_RESULTS) -> dict:
    """
    Search the web via Google. Returns {ok, results?: [{title, href, body}], error?}.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "Query required"}
    try:
        from googlesearch import search as google_search

        results = []
        try:
            # googlesearch-python: advanced=True returns SearchResult objects
            for r in google_search(query, num_results=max_results, advanced=True):
                results.append({
                    "title": getattr(r, "title", str(r) if not hasattr(r, "url") else "") or "",
                    "href": getattr(r, "url", str(r) if isinstance(r, str) else "") or "",
                    "body": (getattr(r, "description", "") or "")[:500],
                })
        except TypeError:
            # Fallback: basic search returns URLs only
            for url in google_search(query, num_results=max_results):
                results.append({"title": url, "href": url, "body": ""})
        return {"ok": True, "results": results}
    except ImportError:
        return {"ok": False, "error": "pip install googlesearch-python"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
