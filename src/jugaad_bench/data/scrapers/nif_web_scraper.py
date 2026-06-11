"""
NIF Innovation Portal Web Scraper.

Scrapes the NIF Innovation Portal (innovation.nif.org.in) to extract
innovation entries from paginated category listing pages.

Features:
- Respects robots.txt directives
- Rate limits at configurable interval (default: 1 req / 2 sec)
- Handles HTTP errors gracefully with retries
- Parses innovation cards into structured RawCase objects
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from jugaad_bench.models import RawCase
from jugaad_bench.data.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class _RobotsTxtParser:
    """Minimal robots.txt parser for checking disallowed paths."""

    def __init__(self) -> None:
        self.disallowed_paths: list[str] = []
        self.crawl_delay: float | None = None

    def parse(self, robots_text: str) -> None:
        """Parse robots.txt content and extract rules for all user agents."""
        applies_to_us = False
        for line in robots_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip().lower()
                applies_to_us = agent == "*" or "jugaad" in agent
            elif applies_to_us and line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    self.disallowed_paths.append(path)
            elif applies_to_us and line.lower().startswith("crawl-delay:"):
                try:
                    self.crawl_delay = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

    def is_allowed(self, path: str) -> bool:
        """Check if a path is allowed by robots.txt."""
        for disallowed in self.disallowed_paths:
            if path.startswith(disallowed):
                return False
        return True


class _SimpleHTMLExtractor(HTMLParser):
    """Lightweight HTML parser to extract innovation cards from NIF pages.

    Looks for structured content blocks containing title, description,
    innovator name, and location information.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cards: list[dict[str, str]] = []
        self._current_card: dict[str, str] | None = None
        self._current_tag: str = ""
        self._current_classes: list[str] = []
        self._capture_text: bool = False
        self._text_buffer: list[str] = []
        self._capture_target: str = ""
        self._in_card: bool = False
        self._pagination_links: list[str] = []

    @property
    def cards(self) -> list[dict[str, str]]:
        return self._cards

    @property
    def pagination_links(self) -> list[str]:
        return self._pagination_links

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        classes = attr_dict.get("class", "").lower().split()

        # Detect innovation card containers
        if tag == "div" and any(
            cls in classes
            for cls in (
                "innovation-card",
                "card",
                "innovation-item",
                "innovation_item",
                "node",
                "views-row",
                "item-list",
            )
        ):
            self._in_card = True
            self._current_card = {}

        # Title detection within a card
        if self._in_card and tag in ("h2", "h3", "h4"):
            self._capture_text = True
            self._capture_target = "title"
            self._text_buffer = []

        if self._in_card and tag == "a" and "title" not in (self._current_card or {}):
            # Title may be in an anchor within a heading
            if self._capture_target == "title":
                href = attr_dict.get("href", "")
                if self._current_card is not None and href:
                    self._current_card["link"] = href

        # Description / body text
        if self._in_card and tag == "div" and any(
            cls in classes
            for cls in (
                "description",
                "body",
                "summary",
                "field-content",
                "field-item",
                "content",
                "innovation-desc",
            )
        ):
            self._capture_text = True
            self._capture_target = "description"
            self._text_buffer = []

        # Innovator name
        if self._in_card and tag in ("span", "div", "p") and any(
            cls in classes
            for cls in ("innovator", "author", "name", "innovator-name", "field-name")
        ):
            self._capture_text = True
            self._capture_target = "innovator"
            self._text_buffer = []

        # Location
        if self._in_card and tag in ("span", "div", "p") and any(
            cls in classes for cls in ("location", "place", "state", "address")
        ):
            self._capture_text = True
            self._capture_target = "location"
            self._text_buffer = []

        # Category
        if self._in_card and tag in ("span", "div", "a") and any(
            cls in classes for cls in ("category", "tag", "sector", "domain")
        ):
            self._capture_text = True
            self._capture_target = "category"
            self._text_buffer = []

        # Pagination links
        if tag == "a" and any(
            cls in classes for cls in ("pager-next", "next", "page-link")
        ):
            href = attr_dict.get("href", "")
            if href:
                self._pagination_links.append(href)

        # Also look for pagination via rel="next"
        if tag == "a" and attr_dict.get("rel") == "next":
            href = attr_dict.get("href", "")
            if href:
                self._pagination_links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if self._capture_text and tag in (
            "h2", "h3", "h4", "div", "span", "p", "a",
        ):
            text = " ".join(self._text_buffer).strip()
            if text and self._current_card is not None:
                self._current_card[self._capture_target] = text
            self._capture_text = False
            self._capture_target = ""
            self._text_buffer = []

        # End of card
        if tag == "div" and self._in_card and self._current_card:
            if self._current_card.get("title") or self._current_card.get("description"):
                self._cards.append(self._current_card)
            self._current_card = None
            self._in_card = False

    def handle_data(self, data: str) -> None:
        if self._capture_text:
            cleaned = data.strip()
            if cleaned:
                self._text_buffer.append(cleaned)


class _FallbackHTMLExtractor:
    """Regex-based fallback extractor for when the HTML parser finds nothing.

    Uses regex patterns to find innovation-related content blocks.
    """

    # Match content that looks like an innovation entry
    _ENTRY_RE = re.compile(
        r"<(?:div|article|section)[^>]*class=[\"'][^\"']*"
        r"(?:innovation|node|views-row|item)[^\"']*[\"'][^>]*>"
        r"(.*?)"
        r"</(?:div|article|section)>",
        re.DOTALL | re.IGNORECASE,
    )
    _TAG_RE = re.compile(r"<[^>]+>")
    _WHITESPACE_RE = re.compile(r"\s+")

    @classmethod
    def extract(cls, html: str) -> list[dict[str, str]]:
        """Extract innovation entries using regex patterns."""
        cards: list[dict[str, str]] = []
        for match in cls._ENTRY_RE.finditer(html):
            content = match.group(1)
            text = cls._TAG_RE.sub(" ", content)
            text = cls._WHITESPACE_RE.sub(" ", text).strip()
            if len(text) >= 50:
                # Try to extract a title from heading tags
                title_match = re.search(
                    r"<h[2-4][^>]*>(.*?)</h[2-4]>", content, re.DOTALL
                )
                title = None
                if title_match:
                    title = cls._TAG_RE.sub("", title_match.group(1)).strip()
                cards.append(
                    {"title": title or "", "description": text}
                )
        return cards


class NIFWebScraper(BaseScraper):
    """Scraper for the NIF Innovation Portal at innovation.nif.org.in.

    Args:
        base_url: Root URL of the NIF portal.
        rate_limit_seconds: Seconds between requests (default 2.0).
        max_pages: Maximum number of pages to scrape per category.
    """

    # Known category/listing paths on the NIF portal
    _CATEGORY_PATHS: list[str] = [
        "/innovation",
        "/innovation/agriculture",
        "/innovation/energy",
        "/innovation/engineering",
        "/innovation/general-machine",
        "/innovation/food-processing",
        "/innovation/textile",
        "/innovation/electrical-electronic",
        "/innovation/chemical",
        "/innovation/others",
    ]

    def __init__(
        self,
        base_url: str = "https://innovation.nif.org.in",
        rate_limit_seconds: float = 2.0,
        max_pages: int = 500,
    ) -> None:
        super().__init__(
            source_name="nif_web",
            rate_limit_seconds=rate_limit_seconds,
        )
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        self._robots: _RobotsTxtParser | None = None

    async def scrape(self) -> list[RawCase]:
        """Scrape innovation entries from the NIF portal.

        Returns:
            List of RawCase objects from all categories.
        """
        # Respect robots.txt
        await self._load_robots_txt()

        all_cases: list[RawCase] = []
        seen_titles: set[str] = set()

        for category_path in self._CATEGORY_PATHS:
            if not self._is_path_allowed(category_path):
                logger.info("Skipping disallowed path: %s", category_path)
                continue

            logger.info("Scraping category: %s", category_path)
            try:
                cases = await self._scrape_category(category_path, seen_titles)
                all_cases.extend(cases)
                logger.info(
                    "Got %d entries from %s (total: %d)",
                    len(cases),
                    category_path,
                    len(all_cases),
                )
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %d on %s — skipping category",
                    exc.response.status_code,
                    category_path,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "Network error on %s: %s — skipping category",
                    category_path,
                    exc,
                )
            except Exception:
                logger.exception("Unexpected error on %s", category_path)

        logger.info("Total NIF web entries scraped: %d", len(all_cases))
        return all_cases

    # ------------------------------------------------------------------
    # Robots.txt
    # ------------------------------------------------------------------

    async def _load_robots_txt(self) -> None:
        """Load and parse robots.txt from the portal."""
        self._robots = _RobotsTxtParser()
        try:
            robots_url = f"{self.base_url}/robots.txt"
            text = await self.fetch(robots_url, use_cache=True)
            self._robots.parse(text)

            # Apply crawl-delay from robots.txt if it's stricter
            if (
                self._robots.crawl_delay is not None
                and self._robots.crawl_delay > self.rate_limit_seconds
            ):
                logger.info(
                    "Applying robots.txt crawl-delay: %.1fs",
                    self._robots.crawl_delay,
                )
                self.rate_limit_seconds = self._robots.crawl_delay

        except (httpx.HTTPStatusError, httpx.RequestError):
            logger.info("Could not fetch robots.txt — proceeding with defaults")

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed by robots.txt."""
        if self._robots is None:
            return True
        return self._robots.is_allowed(path)

    # ------------------------------------------------------------------
    # Category scraping
    # ------------------------------------------------------------------

    async def _scrape_category(
        self, category_path: str, seen_titles: set[str]
    ) -> list[RawCase]:
        """Scrape all pages of a single category listing."""
        cases: list[RawCase] = []
        current_url = f"{self.base_url}{category_path}"
        pages_scraped = 0

        while current_url and pages_scraped < self.max_pages:
            try:
                html = await self.fetch(current_url, use_cache=True)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    logger.debug("Page not found: %s", current_url)
                    break
                raise

            # Parse HTML for innovation cards
            cards, next_url = self._parse_listing_page(html, current_url)

            for card in cards:
                title = card.get("title", "").strip()
                # Deduplicate within the same scrape run
                dedup_key = title.lower() if title else card.get("description", "")[:80].lower()
                if dedup_key in seen_titles:
                    continue
                seen_titles.add(dedup_key)

                raw_case = self._card_to_raw_case(card, current_url)
                if raw_case is not None:
                    cases.append(raw_case)

            pages_scraped += 1

            # Follow pagination
            if next_url and next_url != current_url:
                current_url = next_url
            else:
                break

        return cases

    def _parse_listing_page(
        self, html: str, page_url: str
    ) -> tuple[list[dict[str, str]], str | None]:
        """Parse an HTML listing page for innovation cards and pagination.

        Returns:
            Tuple of (cards, next_page_url_or_None).
        """
        parser = _SimpleHTMLExtractor()
        try:
            parser.feed(html)
        except Exception:
            logger.warning("HTML parse error on %s", page_url)

        cards = parser.cards

        # Fallback extraction if HTML parser found nothing
        if not cards:
            cards = _FallbackHTMLExtractor.extract(html)

        # Determine next page URL
        next_url: str | None = None
        if parser.pagination_links:
            raw_next = parser.pagination_links[0]
            next_url = urljoin(page_url, raw_next)
            # Validate same domain
            if urlparse(next_url).netloc != urlparse(self.base_url).netloc:
                next_url = None
        else:
            # Try to find ?page=N pattern
            page_match = re.search(r"[?&]page=(\d+)", page_url)
            current_page = int(page_match.group(1)) if page_match else 0
            if cards:  # Only paginate forward if we found content
                if "?" in page_url and "page=" in page_url:
                    next_url = re.sub(
                        r"page=\d+", f"page={current_page + 1}", page_url
                    )
                else:
                    sep = "&" if "?" in page_url else "?"
                    next_url = f"{page_url}{sep}page={current_page + 1}"

        return cards, next_url

    def _card_to_raw_case(
        self, card: dict[str, str], page_url: str
    ) -> RawCase | None:
        """Convert a parsed card dict into a RawCase."""
        title = card.get("title", "").strip()
        description = card.get("description", "").strip()
        innovator = card.get("innovator", "").strip() or None
        location = card.get("location", "").strip() or None
        category = card.get("category", "").strip() or None

        # Build the raw text from available fields
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if description:
            text_parts.append(description)
        if innovator:
            text_parts.append(f"Innovator: {innovator}")
        if location:
            text_parts.append(f"Location: {location}")

        raw_text = "\n".join(text_parts)

        if len(raw_text) < 20:
            return None

        try:
            return RawCase(
                source="nif_web",
                url_or_path=page_url,
                raw_text=raw_text,
                title=title or None,
                innovator_name=innovator,
                location=location,
                category=category,
            )
        except Exception:
            logger.warning("Failed to create RawCase from card: %s", title)
            return None
