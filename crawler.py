"""
Asynchronous single-worker BFS web crawler.

Coordinates:
- HTTP fetching with retry/backoff (httpx)
- robots.txt compliance (urllib.robotparser)
- Per-domain politeness rate-limiting
- BFS traversal via collections.deque
- Session deadline (hard stop at ~150s)
- Response-size cap (skip pages > 2MB)

Delegates URL logic to url_utils.py and content parsing to content_extractor.py.
"""
import asyncio
import logging
import time
import urllib.robotparser
from collections import deque
from typing import Callable, Optional

import httpx

import config
from url_utils import (
    normalize_url,
    extract_domain,
    is_same_domain,
    should_skip_extension,
    is_public_url,
)
from content_extractor import extract_text, extract_title, extract_links

logger = logging.getLogger(__name__)

# HTTP headers to identify our crawler respectfully (includes repository info to satisfy Wikimedia & web standards)
HEADERS = {
    "User-Agent": "RAGSearchCrawler/1.0 (+https://github.com/tarun819/web_crawler; bot@educational-search.org)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

# In-memory state for robots.txt caching and politeness tracking
_robots_cache: dict[str, str] = {}
_last_request_time: dict[str, float] = {}


# =====================================================================
# robots.txt & Politeness
# =====================================================================

async def fetch_robots_txt(domain: str, client: httpx.AsyncClient) -> str:
    """Fetch and cache robots.txt for a domain."""
    if domain in _robots_cache:
        return _robots_cache[domain]

    robots_url = f"https://{domain}/robots.txt"
    robots_text = ""
    try:
        response = await client.get(robots_url, follow_redirects=True, timeout=5.0)
        if response.status_code == 200:
            robots_text = response.text
    except Exception as e:
        logger.debug(f"Could not fetch robots.txt for {domain} (defaulting to allow): {e}")

    _robots_cache[domain] = robots_text
    return robots_text


def is_allowed(url: str, robots_txt: str, user_agent: str = "RAGSearchCrawler") -> bool:
    """Check if URL is allowed to be crawled according to robots.txt."""
    if not robots_txt.strip():
        return True

    clean_robots = robots_txt.lstrip("\ufeff")
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url("http://dummy.com/robots.txt")
    parser.parse(clean_robots.splitlines())
    return parser.can_fetch(user_agent, url)


async def wait_for_politeness(domain: str, delay: float = config.CRAWL_DELAY) -> None:
    """Ensure at least `delay` seconds have passed since last request to domain."""
    now = time.time()
    last_time = _last_request_time.get(domain, 0.0)
    elapsed = now - last_time
    if elapsed < delay:
        await asyncio.sleep(delay - elapsed)
    _last_request_time[domain] = time.time()


# =====================================================================
# HTTP Fetching with Retries
# =====================================================================

async def fetch_page(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Fetch page HTML asynchronously with retry backoff for 429/5xx status codes.

    Safety checks:
    - Skips non-HTML content types
    - Skips pages exceeding MAX_RESPONSE_SIZE (~2MB) to prevent memory issues

    Returns HTML string or None if fetch fails or content is non-HTML.
    """
    domain = extract_domain(url)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            await wait_for_politeness(domain)
            response = await client.get(url, follow_redirects=True, timeout=config.REQUEST_TIMEOUT)

            # Check for rate limits or server errors
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < config.MAX_RETRIES:
                    backoff = 2 ** attempt
                    logger.warning(f"HTTP {response.status_code} for {url}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"HTTP {response.status_code} for {url} after {config.MAX_RETRIES} retries.")
                    return None

            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} skipping {url}")
                return None

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logger.debug(f"Skipping non-HTML content-type '{content_type}' at {url}")
                return None

            # Response-size cap: skip oversized pages (> 2MB) before parsing
            if len(response.content) > config.MAX_RESPONSE_SIZE:
                logger.warning(f"Skipping oversized page ({len(response.content)} bytes): {url}")
                return None

            return response.text

        except (httpx.TimeoutException, httpx.RequestError) as e:
            if attempt < config.MAX_RETRIES:
                backoff = 2 ** attempt
                logger.warning(f"Network error ({e}) fetching {url}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
            else:
                logger.error(f"Failed to fetch {url} after {config.MAX_RETRIES} attempts: {e}")
                return None
        except Exception as e:
            # Catches unexpected non-network errors (e.g., encoding, decoding, or parsing issues)
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    return None


# =====================================================================
# Main Async BFS Crawl Loop
# =====================================================================

async def crawl(
    seed_url: str,
    max_pages: int = config.MAX_PAGES,
    max_depth: int = config.MAX_DEPTH,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[dict]:
    """
    Asynchronous single-worker BFS web crawler.

    Safety features:
    - SSRF validation: rejects private/loopback/link-local IPs
    - Session deadline: hard stop at CRAWL_SESSION_TIMEOUT (~150s)
    - Response-size cap: skips pages exceeding MAX_RESPONSE_SIZE (~2MB)
    - Per-page timeout: asyncio.wait_for wraps each fetch

    Args:
        seed_url: The starting URL
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum link depth from the seed URL
        progress_callback: Optional callback(current_url, pages_crawled, total_found)

    Returns:
        List of crawled page dicts: [{"url": ..., "title": ..., "text": ..., "depth": ...}]
    """
    if not is_public_url(seed_url):
        logger.error(f"Cannot crawl non-public, local, or private URL: {seed_url}")
        return []

    normalized_seed = normalize_url(seed_url)
    target_domain = extract_domain(normalized_seed)

    queue: deque[tuple[str, int]] = deque()
    queue.append((normalized_seed, 0))

    visited: set[str] = {normalized_seed}
    results: list[dict] = []

    # Session deadline: record start time for hard stop
    session_start = time.monotonic()

    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=httpx.Timeout(config.REQUEST_TIMEOUT),
    ) as client:
        # Check robots.txt once upfront
        robots_txt = await fetch_robots_txt(target_domain, client)

        while queue and len(results) < max_pages:
            # Hard stop: session deadline exceeded
            elapsed = time.monotonic() - session_start
            if elapsed > config.CRAWL_SESSION_TIMEOUT:
                logger.warning(
                    f"Session deadline ({config.CRAWL_SESSION_TIMEOUT}s) reached "
                    f"after {len(results)} pages. Stopping crawl."
                )
                break

            current_url, current_depth = queue.popleft()

            # Check robots.txt permission
            if not is_allowed(current_url, robots_txt):
                logger.info(f"Skipping (blocked by robots.txt): {current_url}")
                continue

            if progress_callback:
                progress_callback(current_url, len(results), len(visited))

            # Fetch page HTML with per-page timeout guard
            try:
                html = await asyncio.wait_for(
                    fetch_page(current_url, client),
                    timeout=config.REQUEST_TIMEOUT + 2,  # small buffer over per-request timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Per-page timeout exceeded for {current_url}, skipping.")
                html = None

            if not html:
                continue

            # Extract clean content (trafilatura) & discovered links (BS4)
            text = extract_text(html)
            title = extract_title(html)
            links = extract_links(html)

            if not text or not text.strip():
                logger.debug(f"Skipping page with empty text: {current_url}")
                continue

            page_data = {
                "url": current_url,
                "domain": target_domain,
                "title": title or current_url,
                "text": text,
                "depth": current_depth,
            }
            results.append(page_data)
            logger.info(f"[{len(results)}/{max_pages}] Crawled: {current_url} (depth={current_depth})")

            # BFS link discovery
            if current_depth < max_depth:
                for raw_link in links:
                    normalized_link = normalize_url(raw_link, base_url=current_url)

                    if should_skip_extension(normalized_link):
                        continue

                    if not is_same_domain(normalized_link, target_domain):
                        continue

                    if normalized_link not in visited:
                        visited.add(normalized_link)
                        queue.append((normalized_link, current_depth + 1))

    logger.info(f"Crawl finished for domain '{target_domain}'. Successfully extracted {len(results)} pages.")
    return results
