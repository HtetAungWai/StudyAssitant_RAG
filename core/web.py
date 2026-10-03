"""Free web search for Global Library mode (no API key needed).

Primary: DuckDuckGo via the `ddgs` package. Fallback: Wikipedia's public API.
Top results are optionally opened so the AI sees real page text, not just snippets.
"""
import re
from concurrent.futures import ThreadPoolExecutor

import requests

UA = {"User-Agent": "StudyRAG/1.0 (university capstone project)"}
MAX_CHARS = 1500


def _ddg(query: str, n: int) -> list[dict]:
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=n) or []
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in rows
        if r.get("href")
    ]


def _wikipedia(query: str, n: int) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": n,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "exchars": MAX_CHARS,
        "exlimit": "max",
    }
    r = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=UA, timeout=10)
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    out = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        title = p.get("title", "")
        out.append(
            {
                "title": f"{title} (Wikipedia)",
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                "snippet": (p.get("extract") or "").strip(),
            }
        )
    return out


def _page_text(url: str) -> str:
    """Best-effort readable text of a web page ('' on any failure)."""
    try:
        import trafilatura

        r = requests.get(url, headers=UA, timeout=8)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        text = trafilatura.extract(r.text) or ""
        return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]
    except Exception:
        return ""


def search_web(query: str, n: int = 5, read_pages: bool = True) -> tuple[list[dict], str]:
    """Return (chunks, engine). chunks = [{'text','source'(title),'location'(url)}]."""
    results, engine = [], "DuckDuckGo"
    try:
        results = _ddg(query, n)
    except Exception:
        results = []
    if not results:
        engine = "Wikipedia"
        try:
            results = _wikipedia(query, n)
        except Exception:
            return [], "none"

    texts = [r["snippet"] for r in results]
    if read_pages and engine == "DuckDuckGo":
        with ThreadPoolExecutor(max_workers=4) as pool:
            pages = list(pool.map(_page_text, [r["url"] for r in results[:4]]))
        for i, page in enumerate(pages):
            if len(page) > len(texts[i]):
                texts[i] = page

    chunks = [
        {"text": t[:MAX_CHARS], "source": r["title"] or r["url"], "location": r["url"]}
        for r, t in zip(results, texts)
        if t.strip()
    ]
    return chunks, engine
