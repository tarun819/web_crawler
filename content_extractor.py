"""
Content extraction using trafilatura + BS4 link discovery.

Two separate concerns, two separate tools:
- trafilatura: extracts clean main-content text and page title.
  Battle-tested against thousands of real-world site layouts.
- BS4 + lxml: walks <a href> tags for link discovery only.
  trafilatura doesn't expose raw href values, so we use BS4 for this.

No network calls, no async code — pure HTML-in, structured-data-out.
"""
import logging
from typing import Optional


import trafilatura

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_text(html: str) -> Optional[str]:
    """
    Extract clean main-content text from raw HTML using trafilatura.

    trafilatura uses a combination of heuristics and machine-learned rules
    to strip navigation, footers, sidebars, and boilerplate — returning
    only the primary content of the page.

    Returns clean text string, or None if extraction fails.
    """
    try:
        text = trafilatura.extract(html, include_links=False, include_comments=False)
        return text
    except Exception as e:
        logger.warning(f"trafilatura extraction failed: {e}")
        return None


def extract_title(html: str) -> str:
    """
    Extract page title from raw HTML using trafilatura's metadata extraction.

    Falls back to BS4 <title> tag or first <h1> if metadata extraction
    returns nothing.
    """
    try:
        metadata = trafilatura.extract_metadata(html)
        if metadata and metadata.title:
            return metadata.title.strip()
    except Exception:
        pass

    # Fallback: BS4 <title> or <h1>
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def extract_links(html: str) -> list[str]:
    """
    Extract all discoverable links from raw HTML using BS4.

    This is kept separate from trafilatura because trafilatura focuses
    on content extraction and doesn't expose raw href values for crawl
    frontier discovery.

    Filters out:
    - javascript: pseudo-links
    - mailto: and tel: links
    - Empty or fragment-only (#) hrefs
    """
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if (
            href
            and not href.startswith("javascript:")
            and not href.startswith("mailto:")
            and not href.startswith("tel:")
            and href != "#"
        ):
            links.append(href)
    return links
