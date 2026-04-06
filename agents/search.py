"""
Scholar Lens - Search Agent (Parallel Worker)

Each section gets its own search_worker_node instance
spawned by Send() API in graph.py — all run in parallel.

Per-section strategy:
  Introduction  → Playwright only (Wikipedia background)
  Conclusion    → No search (writer reads written sections)
  All others    → Serper finds URLs + Playwright scrapes full content

Reads  → section_title, topic, research_angle,
          key_themes, suggested_sources (from Send() mini-state)
Writes → search_results: { section_title: [content strings] }
         (merged into main state via merge_search_results reducer)
"""

import os
import requests
from playwright.sync_api import sync_playwright


# ─────────────────────────────────────────────────────────
# 1. SERPER — finds top URLs for a query
# ─────────────────────────────────────────────────────────

def _serper_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Calls Serper API with a query string.
    Returns list of { title, url, snippet } dicts.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY not set in environment")

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        },
        json={"q": query, "num": num_results},
        timeout=10
    )

    data = response.json()
    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("link", ""),
            "snippet": item.get("snippet", "")
        })
    return results


# ─────────────────────────────────────────────────────────
# 2. PLAYWRIGHT — scrapes full page content from a URL
# ─────────────────────────────────────────────────────────

def _scrape_url(url: str, timeout: int = 10000) -> str:
    """
    Uses Playwright headless Chromium to scrape full page text.
    Blocks images/css for speed.
    Falls back gracefully if page fails.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox"
                ]
            )
            page = browser.new_page()

            # Note: page.route() removed — causes asyncio.CancelledError
            # when Gradio's thread pool cancels pending route handlers.
            # Scraping is slightly slower without it but error-free.
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")

            content = page.evaluate("""() => {
                ['nav', 'footer', 'header', 'script', 'style', '.ads']
                    .forEach(sel => {
                        document.querySelectorAll(sel)
                            .forEach(el => el.remove())
                    });
                const main = document.querySelector(
                    'main, article, .content, #content, .post-content'
                );
                if (main) return main.innerText;
                return document.body.innerText;
            }""")

            browser.close()

            cleaned = " ".join(content.split())
            return cleaned[:3000]

    except Exception as e:
        return f"[Scrape failed for {url}: {str(e)}]"


# ─────────────────────────────────────────────────────────
# 3. QUERY BUILDER
# ─────────────────────────────────────────────────────────

def _build_query(
    section_title: str,
    topic: str,
    research_angle: str,
    key_themes: list[str],
    suggested_sources: list[str]
) -> str:
    """
    Builds the best search query for a section.
    Uses planner's suggested_sources first,
    falls back to topic + section + top key_theme.
    """
    section_words = set(section_title.lower().split())

    for suggested in suggested_sources:
        if section_words & set(suggested.lower().split()):
            return suggested

    top_theme = key_themes[0] if key_themes else ""
    return f"{topic} {section_title} {top_theme} academic research".strip()


# ─────────────────────────────────────────────────────────
# 4. SEARCH WORKER NODE — one instance per section
# ─────────────────────────────────────────────────────────

def search_worker_node(state: dict) -> dict:
    """
    Parallel worker — one spawned per section via Send() API.

    Receives a mini-state dict from Send() containing:
      section_title, topic, research_angle,
      key_themes, suggested_sources

    Returns partial search_results for this section only.
    LangGraph's merge_search_results reducer merges all workers.

    Three strategies based on section type:
      Introduction → Playwright only (Wikipedia)
      Conclusion   → No search (writer uses written sections)
      Others       → Serper + Playwright
    """

    section_title     = state["section_title"]
    topic             = state["topic"]
    research_angle    = state["research_angle"]
    key_themes        = state["key_themes"]
    suggested_sources = state["suggested_sources"]

    section_lower = section_title.lower()

    # ── Strategy 1: Introduction → Wikipedia only ────────
    if section_lower == "introduction":
        wiki_url = (
            f"https://en.wikipedia.org/wiki/"
            f"{topic.replace(' ', '_')}"
        )
        content = _scrape_url(wiki_url)
        return {
            "search_results": {
                section_title: [f"WIKIPEDIA BACKGROUND:\n{content}"]
            },
            "status": f"📖 Introduction: scraped Wikipedia background"
        }

    # ── Strategy 2: Conclusion → no search ───────────────
    if section_lower == "conclusion":
        return {
            "search_results": {
                section_title: []   # writer reads state["sections"] directly
            },
            "status": f"✅ Conclusion: no search needed — writer uses written sections"
        }

    # ── Strategy 3: All others → Serper + Playwright ─────
    query          = _build_query(
        section_title, topic, research_angle,
        key_themes, suggested_sources
    )
    serper_results = _serper_search(query, num_results=3)
    section_content = []

    for result in serper_results:
        # Always keep snippet as fallback
        section_content.append(
            f"SOURCE: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"SNIPPET: {result['snippet']}"
        )
        # Playwright scrapes full content
        full_content = _scrape_url(result["url"])
        if full_content and not full_content.startswith("[Scrape failed"):
            section_content.append(
                f"FULL CONTENT FROM '{result['title']}':\n{full_content}"
            )

    return {
        "search_results": {section_title: section_content},
        "status": f"🔍 {section_title}: found {len(serper_results)} sources"
    }