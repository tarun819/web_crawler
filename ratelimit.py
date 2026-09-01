"""
In-memory sliding-window rate limiter.

Uses a deque of timestamps per client IP to enforce a maximum
number of requests per time window. No Redis or external services required.
"""
import time
from collections import deque

# Per-IP request timestamp storage
_request_log: dict[str, deque] = {}

# Rate limit config
MAX_REQUESTS = 10      # Maximum requests allowed per window
WINDOW_SECONDS = 60    # Time window in seconds


def check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """
    Check rate limit for an IP and record the request if allowed.

    Returns:
        (is_limited: bool, remaining_requests: int)
    """
    now = time.monotonic()

    if client_ip not in _request_log:
        _request_log[client_ip] = deque()

    timestamps = _request_log[client_ip]

    # Purge timestamps older than the window
    while timestamps and (now - timestamps[0]) > WINDOW_SECONDS:
        timestamps.popleft()

    # Check if rate limit exceeded
    if len(timestamps) >= MAX_REQUESTS:
        return True, 0

    # Record this request
    timestamps.append(now)
    remaining = MAX_REQUESTS - len(timestamps)
    return False, remaining
