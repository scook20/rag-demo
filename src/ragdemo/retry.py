"""Retry with exponential backoff and jitter for rate limits and transient server errors."""

from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

import httpx
from google.genai import errors

T = TypeVar("T")


MAX_SERVER_HINT_WAIT = 120.0  # a quota reset further away than this (a daily cap) is not worth waiting for

_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def server_retry_delay(exc: BaseException) -> float | None:
    """Seconds the API asked us to wait (RetryInfo.retryDelay in a 429 body), if it said."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.ServerError):  # 5xx
        return True
    if isinstance(exc, errors.ClientError):  # 4xx: only rate limit and request timeout are transient
        if getattr(exc, "code", None) == 429:
            hint = server_retry_delay(exc)
            return hint is None or hint <= MAX_SERVER_HINT_WAIT  # per-minute limit: wait; daily quota: stop
        return getattr(exc, "code", None) == 408
    return isinstance(exc, httpx.TransportError)  # connection resets, timeouts


def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[BaseException], bool] = is_retryable,
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            wait = delay * (0.5 + random.random() / 2)  # jitter: 50%-100% of the backoff
            hint = server_retry_delay(exc)
            if hint is not None:
                wait = max(wait, hint + 1.0)  # honour the server's own retry delay
            sleep(wait)
