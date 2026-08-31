"""
Test suite for the crawler pipeline:
- URL normalization & domain matching (url_utils)
- SSRF validation (url_utils.is_public_url)
- Content extraction via trafilatura (content_extractor)
- Live crawl with session deadline (crawler)
"""
import asyncio
import logging
from url_utils import normalize_url, is_same_domain, is_public_url
from content_extractor import extract_text, extract_title, extract_links
from crawler import is_allowed, crawl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_url_normalization():
    print("=" * 60)
    print("TEST 1: URL Normalization & Domain Matching")
    print("=" * 60)

    assert normalize_url("https://Docs.Python.ORG/3/tutorial/#intro") == "https://docs.python.org/3/tutorial"
    assert normalize_url("https://example.com/page/") == "https://example.com/page"
    assert normalize_url("sub/page.html", base_url="https://example.com/docs/") == "https://example.com/docs/sub/page.html"
    assert normalize_url("https://example.com/search?q=test&utm_source=twitter") == "https://example.com/search?q=test"

    assert is_same_domain("https://docs.python.org/3/", "python.org")
    assert is_same_domain("https://sub.docs.python.org/", "python.org")
    assert not is_same_domain("https://google.com", "python.org")
    print("✅ URL normalization passed!\n")


def test_ssrf_validation():
    print("=" * 60)
    print("TEST 2: SSRF & Public URL Validation")
    print("=" * 60)

    # Valid public URLs
    assert is_public_url("https://docs.python.org/3/") is True
    assert is_public_url("https://httpbin.org/html") is True

    # Malicious / Private / Local / SSRF URLs (MUST ALL BE REJECTED)
    assert is_public_url("http://127.0.0.1:8000") is False, "Failed to block 127.0.0.1"
    assert is_public_url("http://localhost:6379") is False, "Failed to block localhost"
    assert is_public_url("http://169.254.169.254/latest/meta-data") is False, "Failed to block AWS metadata IP"
    assert is_public_url("http://10.0.0.5/admin") is False, "Failed to block 10.x.x.x private range"
    assert is_public_url("http://192.168.1.1/") is False, "Failed to block 192.168.x.x private range"
    assert is_public_url("http://172.16.0.1/") is False, "Failed to block 172.16.x.x private range"
    assert is_public_url("file:///etc/passwd") is False, "Failed to block file:// scheme"
    assert is_public_url("ftp://example.com/file") is False, "Failed to block ftp:// scheme"

    print("✅ SSRF protection: all private/local/invalid schemes successfully blocked!\n")


def test_content_extraction():
    print("=" * 60)
    print("TEST 3: Content Extraction (trafilatura + BS4 links)")
    print("=" * 60)

    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Python Asyncio Guide - Docs</title></head>
    <body>
        <header><nav><a href="/home">Home</a> | <a href="/about">About</a></nav></header>
        <div class="sidebar">
            <ul><li><a href="/menu1">Menu 1</a></li></ul>
        </div>

        <main class="docs-content">
            <h1>Asyncio in Python</h1>
            <p>Asyncio is a library to write concurrent code using the async/await syntax.</p>
            <p>It is used as a foundation for multiple Python asynchronous frameworks.</p>
            <pre><code>import asyncio

async def main():
    print('Hello')
    await asyncio.sleep(1)
    print('World')</code></pre>
            <a href="/tutorial/advanced">Advanced Tutorial</a>
            <a href="https://external.com/out">External Resource</a>
            <a href="javascript:void(0)">Noop</a>
        </main>

        <footer><p>Copyright 2026</p></footer>
    </body>
    </html>
    """

    # Test trafilatura text extraction
    text = extract_text(sample_html)
    print(f"Extracted text preview:\n{text[:200] if text else 'None'}...")
    assert text is not None, "trafilatura returned None"
    assert len(text) > 20, "Extracted text is too short"

    # Test title extraction
    title = extract_title(sample_html)
    print(f"Title: '{title}'")
    assert title, "Title extraction returned empty"

    # Test BS4 link discovery
    links = extract_links(sample_html)
    print(f"Discovered links: {links}")
    assert "/tutorial/advanced" in links
    assert "javascript:void(0)" not in links
    print("✅ Content extraction passed!\n")


async def test_live_crawl():
    print("=" * 60)
    print("TEST 4: Live Crawl (httpbin.org/html)")
    print("=" * 60)

    # 1. Test that SSRF blocked URL returns empty results immediately
    bad_crawl = await crawl("http://127.0.0.1:8000/internal", max_pages=1)
    assert bad_crawl == [], "Crawler should immediately return empty on SSRF URL"

    # 2. Test valid crawl
    pages = await crawl("https://httpbin.org/html", max_pages=3, max_depth=1)
    print(f"Pages crawled: {len(pages)}")
    assert len(pages) >= 1
    assert "url" in pages[0]
    assert len(pages[0]["text"]) > 50
    print(f"Sample page title: '{pages[0]['title']}'")
    print("✅ Live crawl passed!\n")


async def main():
    print("\n🚀 Testing Crawler (trafilatura + SSRF + Safety Limits)...\n")
    test_url_normalization()
    test_ssrf_validation()
    test_content_extraction()
    await test_live_crawl()
    print("=" * 60)
    print("🎉 ALL CRAWLER & SECURITY TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
