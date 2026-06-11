"""
Async rate limiter and retry utilities for LLM API calls.

Provides per-provider rate limiting and configurable retry logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

from aiolimiter import AsyncLimiter
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default rate limits per provider (requests per minute)
DEFAULT_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "openai": (500, 60),      # 500 req / 60 sec
    "anthropic": (100, 60),   # 100 req / 60 sec
    "google": (300, 60),      # 300 req / 60 sec
    "together": (200, 60),    # 200 req / 60 sec
    "deepseek": (200, 60),    # 200 req / 60 sec
    "fireworks": (200, 60),   # 200 req / 60 sec
    "default": (50, 60),      # Conservative default
}


class ProviderRateLimiter:
    """
    Manages per-provider rate limiters with configurable RPM.

    Uses aiolimiter.AsyncLimiter for token-bucket rate limiting and
    asyncio.Semaphore for concurrent request limiting.
    """

    def __init__(self, max_concurrency: int = 5):
        self._limiters: dict[str, AsyncLimiter] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _get_limiter(self, provider: str) -> AsyncLimiter:
        """Get or create a rate limiter for a provider."""
        if provider not in self._limiters:
            rate, period = DEFAULT_RATE_LIMITS.get(
                provider.lower(), DEFAULT_RATE_LIMITS["default"]
            )
            self._limiters[provider] = AsyncLimiter(rate, period)
            logger.info(
                f"Created rate limiter for '{provider}': {rate} req / {period}s"
            )
        return self._limiters[provider]

    async def acquire(self, provider: str) -> None:
        """Acquire both rate limit token and concurrency semaphore."""
        limiter = self._get_limiter(provider)
        await self._semaphore.acquire()
        await limiter.acquire()

    def release(self) -> None:
        """Release the concurrency semaphore."""
        self._semaphore.release()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


# Singleton rate limiter instance
_global_limiter: ProviderRateLimiter | None = None


def get_rate_limiter(max_concurrency: int = 5) -> ProviderRateLimiter:
    """Get or create the global rate limiter."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = ProviderRateLimiter(max_concurrency=max_concurrency)
    return _global_limiter


def create_retry_decorator(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    jitter: float = 1.0,
    retry_exceptions: tuple[type[Exception], ...] | None = None,
):
    """
    Create a tenacity retry decorator with exponential backoff + jitter.

    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time in seconds.
        max_wait: Maximum wait time in seconds.
        jitter: Maximum jitter in seconds.
        retry_exceptions: Tuple of exception types to retry on.
            If None, retries on all exceptions.

    Returns:
        A tenacity retry decorator.
    """
    if retry_exceptions is None:
        retry_exceptions = (Exception,)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=min_wait,
            max=max_wait,
            jitter=jitter,
        ),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/{max_attempts} after "
            f"{retry_state.outcome.exception().__class__.__name__}: "
            f"{retry_state.outcome.exception()}"
        ),
        reraise=True,
    )


async def rate_limited_call(
    provider: str,
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 5,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with rate limiting and retry logic.

    Args:
        provider: Provider name for rate limiting.
        func: Async function to call.
        *args: Positional arguments for func.
        max_retries: Maximum retry attempts.
        **kwargs: Keyword arguments for func.

    Returns:
        Result of the function call.
    """
    limiter = get_rate_limiter()
    retry_decorator = create_retry_decorator(max_attempts=max_retries)

    @retry_decorator
    async def _call_with_retry():
        await limiter.acquire(provider)
        try:
            return await func(*args, **kwargs)
        finally:
            limiter.release()

    return await _call_with_retry()
