"""
URL normalization, domain extraction, link filtering, and SSRF validation utilities.

Pure string/parsing logic & network security validation — no heavy ML dependencies,
uses only Python's standard library.
"""
import ipaddress
import socket
import urllib.parse
from typing import Optional


# Non-HTML static asset extensions to skip during crawling
SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".7z", ".mp3", ".mp4", ".avi", ".mov",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".xml", ".json", ".csv", ".xlsx", ".doc", ".docx",
}


def is_public_url(url: str) -> bool:
    """
    SSRF (Server-Side Request Forgery) protection:
    Validates that a URL uses http/https and resolves to a public, globally routable IP.
    
    Rejects:
    - Non-http(s) schemes (e.g., file://, ftp://, gopher://)
    - Loopback addresses (e.g., 127.0.0.1, localhost, ::1)
    - Private IP ranges (e.g., 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Cloud metadata / link-local addresses (e.g., 169.254.169.254)
    - Reserved and multicast IP ranges
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname to all associated IP addresses
        addr_info = socket.getaddrinfo(hostname, None)
        if not addr_info:
            return False

        for *_, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            # is_global automatically rejects private, loopback, link-local, and multicast IPs
            if not ip.is_global:
                return False

        return True
    except Exception:
        return False


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Normalize a URL:
    - Resolves relative URLs against base_url
    - Converts scheme and hostname to lowercase
    - Strips URL fragments (#section)
    - Strips trailing slashes from path (except for root '/')
    - Strips marketing/tracking query parameters (utm_*)
    """
    if base_url:
        url = urllib.parse.urljoin(base_url, url)

    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path

    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")

    # Strip utm_* query params while preserving functional params
    query = ""
    if parsed.query:
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        filtered = [(k, v) for k, v in params if not k.lower().startswith("utm_")]
        if filtered:
            query = urllib.parse.urlencode(filtered)

    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, query, ""))


def extract_domain(url: str) -> str:
    """Extract lowercase netloc/hostname from URL."""
    return urllib.parse.urlparse(url).netloc.lower()


def is_same_domain(url: str, target_domain: str) -> bool:
    """Check if URL belongs to target domain or its subdomains."""
    domain = extract_domain(url)
    target = target_domain.lower()
    return domain == target or domain.endswith("." + target)


def should_skip_extension(url: str) -> bool:
    """Check if URL path ends with a non-HTML static asset extension."""
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)
