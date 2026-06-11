"""
Abstract base class for all JugaadReasoning-1K scrapers.

Provides shared functionality:
- Async httpx client with configurable timeout and headers
- Request caching to disk (data/raw/{source}/)
- Rate limiting via configurable sleep between requests
- JSON export of scraped RawCase objects
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from jugaad_bench.models import RawCase
from jugaad_bench.utils.config import find_project_root

logger = logging.getLogger(__name__)


class BaseScraper(abc.ABC):
    """Abstract base for all data scrapers in the pipeline.

    Subclasses must implement :meth:`scrape` and set :attr:`source_name`.

    Args:
        source_name: Short identifier used for cache directory naming
            (e.g. ``"nif_pdf"``, ``"nif_web"``).
        rate_limit_seconds: Minimum seconds between consecutive HTTP requests.
        timeout_seconds: Per-request HTTP timeout.
        headers: Extra HTTP headers merged with defaults.
        max_retries: Number of automatic retries on transient HTTP errors.
    """

    def __init__(
        self,
        source_name: str,
        rate_limit_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.source_name = source_name
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self._project_root = find_project_root()
        self._cache_dir = self._project_root / "data" / "raw" / source_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        default_headers = {
            "User-Agent": (
                "JugaadBench/1.0 (academic research; "
                "+https://github.com/AKHegde22/Jugadoo)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if headers:
            default_headers.update(headers)

        transport = httpx.AsyncHTTPTransport(retries=max_retries)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers=default_headers,
            transport=transport,
            follow_redirects=True,
        )
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def scrape(self) -> list[RawCase]:
        """Execute the scraping workflow and return extracted cases.

        Must be implemented by every concrete scraper.
        """
        ...

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """Sleep enough to respect the configured rate limit."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            delay = self.rate_limit_seconds - elapsed
            logger.debug("Rate limiting: sleeping %.2fs", delay)
            await asyncio.sleep(delay)
        self._last_request_time = asyncio.get_event_loop().time()

    def _cache_key(self, url: str) -> str:
        """Generate a filesystem-safe cache key from a URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_cache_path(self, url: str, extension: str = ".html") -> Path:
        """Return the cache file path for a given URL."""
        return self._cache_dir / f"{self._cache_key(url)}{extension}"

    def _read_cache(self, url: str, extension: str = ".html") -> str | None:
        """Read a cached response if it exists."""
        cache_path = self._get_cache_path(url, extension)
        if cache_path.exists():
            logger.debug("Cache hit: %s -> %s", url, cache_path.name)
            return cache_path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, url: str, content: str, extension: str = ".html") -> None:
        """Write a response to the cache."""
        cache_path = self._get_cache_path(url, extension)
        cache_path.write_text(content, encoding="utf-8")
        logger.debug("Cached: %s -> %s", url, cache_path.name)

    async def fetch(self, url: str, use_cache: bool = True) -> str:
        """Fetch a URL with rate limiting and optional caching.

        Args:
            url: The URL to fetch.
            use_cache: Whether to use/store cached responses.

        Returns:
            The response body as text.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses after retries.
        """
        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        await self._rate_limit()
        logger.info("Fetching: %s", url)

        response = await self._client.get(url)
        response.raise_for_status()
        text = response.text

        if use_cache:
            self._write_cache(url, text)

        return text

    async def fetch_json(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> dict[str, Any]:
        """Fetch a URL and parse the JSON response.

        Args:
            url: The URL to fetch.
            params: Optional query parameters.
            use_cache: Whether to use/store cached responses.

        Returns:
            Parsed JSON as a dictionary.
        """
        cache_url = url
        if params:
            sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            cache_url = f"{url}?{sorted_params}"

        if use_cache:
            cached = self._read_cache(cache_url, extension=".json")
            if cached is not None:
                return json.loads(cached)

        await self._rate_limit()
        logger.info("Fetching JSON: %s", url)

        response = await self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if use_cache:
            self._write_cache(cache_url, json.dumps(data, ensure_ascii=False), extension=".json")

        return data

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_candidates(cases: list[RawCase], output_path: Path) -> None:
        """Save scraped RawCase objects to a JSON file.

        Args:
            cases: List of RawCase objects to export.
            output_path: Destination file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [case.model_dump(mode="json") for case in cases]
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Exported %d candidates to %s", len(cases), output_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    async def __aenter__(self) -> "BaseScraper":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
