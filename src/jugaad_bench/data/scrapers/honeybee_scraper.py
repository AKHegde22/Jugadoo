"""
Honey Bee Network / SRISTI Newsletter Archive Scraper.

Scrapes the Honey Bee Network newsletter archives at sristi.org/hbnew to
extract innovation entries from newsletter editions.

The archive is a PHP-based site with issue listing pages and individual
issue pages containing multiple innovation write-ups.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from jugaad_bench.models import RawCase
from jugaad_bench.data.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# HTML tag stripping
# --------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = _TAG_RE.sub(" ", html)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Newsletter index parser
# --------------------------------------------------------------------------


class _NewsletterIndexParser(HTMLParser):
    """Parses the newsletter archive index page to find issue links."""

    def __init__(self) -> None:
        super().__init__()
        self.issue_links: list[dict[str, str]] = []
        self._current_href: str = ""
        self._capturing: bool = False
        self._text_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            attr_dict = {k: v or "" for k, v in attrs}
            href = attr_dict.get("href", "")
            # Match links that point to newsletter issues
            if any(
                pattern in href.lower()
                for pattern in ("issue", "edition", "newsletter", "volume", "vol", "hb")
            ):
                self._current_href = href
                self._capturing = True
                self._text_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capturing:
            title = " ".join(self._text_buffer).strip()
            if self._current_href and title:
                self.issue_links.append({"url": self._current_href, "title": title})
            self._capturing = False
            self._current_href = ""
            self._text_buffer = []

    def handle_data(self, data: str) -> None:
        if self._capturing:
            cleaned = data.strip()
            if cleaned:
                self._text_buffer.append(cleaned)


# --------------------------------------------------------------------------
# Newsletter issue content parser
# --------------------------------------------------------------------------


class _NewsletterContentParser(HTMLParser):
    """Parses a single newsletter issue page to extract innovation entries.

    Innovation entries in Honey Bee newsletters typically appear as:
    - Headed by a bold title or sub-heading (h3/h4/strong)
    - Followed by descriptive paragraphs
    - Often include innovator attribution and location
    """

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict[str, str]] = []
        self._current_entry: dict[str, str] = {}
        self._current_tag: str = ""
        self._in_content: bool = False
        self._capturing_title: bool = False
        self._capturing_body: bool = False
        self._title_buffer: list[str] = []
        self._body_buffer: list[str] = []
        self._depth: int = 0
        self._content_depth: int = -1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        classes = attr_dict.get("class", "").lower().split()
        self._current_tag = tag

        # Detect the main content container
        if tag == "div" and any(
            cls in classes
            for cls in (
                "content",
                "entry-content",
                "post-content",
                "article-body",
                "field-item",
                "newsletter-content",
                "node-content",
                "main-content",
            )
        ):
            self._in_content = True
            self._content_depth = self._depth

        if tag == "div":
            self._depth += 1

        if not self._in_content:
            return

        # Sub-headings typically delimit individual innovations
        if tag in ("h2", "h3", "h4", "h5"):
            # Flush previous entry if any
            self._flush_entry()
            self._capturing_title = True
            self._title_buffer = []

        # Bold text can also serve as title in some editions
        if tag in ("strong", "b") and not self._capturing_title:
            # Only treat as title if we're not currently in an entry
            if not self._current_entry.get("title"):
                self._capturing_title = True
                self._title_buffer = []

        # Paragraph = body text
        if tag == "p":
            self._capturing_body = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self._depth -= 1
            if self._in_content and self._depth < self._content_depth:
                self._in_content = False
                self._flush_entry()

        if tag in ("h2", "h3", "h4", "h5") and self._capturing_title:
            self._capturing_title = False
            title_text = " ".join(self._title_buffer).strip()
            if title_text and len(title_text) >= 5:
                self._current_entry["title"] = title_text

        if tag in ("strong", "b") and self._capturing_title:
            self._capturing_title = False
            title_text = " ".join(self._title_buffer).strip()
            if title_text and len(title_text) >= 5:
                self._current_entry["title"] = title_text

        if tag == "p" and self._capturing_body:
            self._capturing_body = False
            body_text = " ".join(self._body_buffer).strip()
            if body_text:
                existing = self._current_entry.get("body", "")
                self._current_entry["body"] = (
                    f"{existing}\n{body_text}" if existing else body_text
                )
            self._body_buffer = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if not cleaned:
            return

        if self._capturing_title:
            self._title_buffer.append(cleaned)
        elif self._capturing_body:
            self._body_buffer.append(cleaned)
        elif self._in_content and self._current_entry.get("title"):
            # Capture loose text as body content
            existing = self._current_entry.get("body", "")
            self._current_entry["body"] = (
                f"{existing} {cleaned}" if existing else cleaned
            )

    def _flush_entry(self) -> None:
        """Save the current entry if it has enough content."""
        if self._current_entry.get("title") or self._current_entry.get("body"):
            body = self._current_entry.get("body", "").strip()
            title = self._current_entry.get("title", "").strip()
            # Only keep entries with meaningful content
            if len(body) >= 30 or (title and len(body) >= 15):
                self.entries.append(dict(self._current_entry))
        self._current_entry = {}
        self._body_buffer = []


# --------------------------------------------------------------------------
# Regex-based fallback extractor
# --------------------------------------------------------------------------

# Matches content blocks that look like innovation descriptions
_INNOVATION_BLOCK_RE = re.compile(
    r"<(?:h[2-5]|strong|b)[^>]*>(.*?)</(?:h[2-5]|strong|b)>"
    r"(.*?)(?=<(?:h[2-5]|strong|b)[^>]*>|</(?:div|article|body)>)",
    re.DOTALL | re.IGNORECASE,
)

# Matches innovator name patterns
_INNOVATOR_RE = re.compile(
    r"(?:Innovator|Inventor|Developed by|Created by|By)\s*[:\-–]?\s*"
    r"([A-Z][a-zA-Z\s\.]+?)(?:[,\.\n]|$)",
    re.IGNORECASE,
)

# Matches location patterns
_LOCATION_RE = re.compile(
    r"(?:Location|Place|State|District|Village|From)\s*[:\-–]?\s*"
    r"([A-Za-z][A-Za-z\s,]+?)(?:[,\.\n]|$)",
    re.IGNORECASE,
)


class HoneyBeeScraper(BaseScraper):
    """Scraper for Honey Bee Network / SRISTI newsletter archives.

    Args:
        base_url: Root URL of the Honey Bee Network site.
        rate_limit_seconds: Seconds between requests (default 2.0).
        max_issues: Maximum number of newsletter issues to scrape.
    """

    # Common archive index paths on the SRISTI/HBN site
    _INDEX_PATHS: list[str] = [
        "/hbnew/",
        "/hbnew/index.php",
        "/hbnew/archive.php",
        "/hbnew/newsletters.php",
        "/hbnew/?page=newsletters",
        "/hbnew/?page=archive",
        "/hbnew/honeybee.php",
    ]

    def __init__(
        self,
        base_url: str = "https://sristi.org/hbnew",
        rate_limit_seconds: float = 2.0,
        max_issues: int = 200,
    ) -> None:
        super().__init__(
            source_name="honeybee",
            rate_limit_seconds=rate_limit_seconds,
        )
        self.base_url = base_url.rstrip("/")
        self.max_issues = max_issues

    async def scrape(self) -> list[RawCase]:
        """Scrape innovation entries from the Honey Bee Network archives.

        Returns:
            List of RawCase objects from all newsletter issues.
        """
        # Step 1: Discover newsletter issue links
        issue_links = await self._discover_issues()
        logger.info("Discovered %d newsletter issues", len(issue_links))

        if not issue_links:
            logger.warning(
                "No newsletter issues discovered — check if the site structure "
                "has changed"
            )
            return []

        # Step 2: Scrape each issue
        all_cases: list[RawCase] = []
        seen_titles: set[str] = set()

        for idx, issue in enumerate(issue_links[: self.max_issues]):
            issue_url = issue["url"]
            issue_title = issue.get("title", f"Issue {idx + 1}")
            logger.info(
                "Scraping issue %d/%d: %s",
                idx + 1,
                min(len(issue_links), self.max_issues),
                issue_title,
            )

            try:
                cases = await self._scrape_issue(issue_url, issue_title, seen_titles)
                all_cases.extend(cases)
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %d on issue %s — skipping",
                    exc.response.status_code,
                    issue_url,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "Network error on issue %s: %s — skipping",
                    issue_url,
                    exc,
                )
            except Exception:
                logger.exception("Error scraping issue: %s", issue_url)

        logger.info("Total Honey Bee Network entries scraped: %d", len(all_cases))
        return all_cases

    # ------------------------------------------------------------------
    # Issue discovery
    # ------------------------------------------------------------------

    async def _discover_issues(self) -> list[dict[str, str]]:
        """Discover newsletter issue links from the archive index.

        Tries multiple known index paths and returns a deduplicated list.
        """
        all_issues: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for index_path in self._INDEX_PATHS:
            url = f"{self.base_url}{index_path}" if not index_path.startswith("http") else index_path
            try:
                html = await self.fetch(url, use_cache=True)
                issues = self._parse_index_page(html, url)
                for issue in issues:
                    full_url = urljoin(url, issue["url"])
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        issue["url"] = full_url
                        all_issues.append(issue)
            except httpx.HTTPStatusError:
                logger.debug("Index path not found: %s", url)
            except httpx.RequestError:
                logger.debug("Cannot reach: %s", url)

        # If structured index didn't find issues, try scraping the main page
        # for any links that look like newsletter issues
        if not all_issues:
            all_issues = await self._discover_issues_fallback()

        return all_issues

    async def _discover_issues_fallback(self) -> list[dict[str, str]]:
        """Fallback: scrape the main page for any newsletter-like links."""
        issues: list[dict[str, str]] = []
        try:
            html = await self.fetch(self.base_url, use_cache=True)
            # Find all links that might be newsletter issues
            link_re = re.compile(
                r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                re.DOTALL | re.IGNORECASE,
            )
            for match in link_re.finditer(html):
                href = match.group(1)
                text = _strip_html(match.group(2)).strip()
                if not text or len(text) < 3:
                    continue
                # Look for links that suggest newsletter content
                href_lower = href.lower()
                text_lower = text.lower()
                if any(
                    kw in href_lower or kw in text_lower
                    for kw in (
                        "issue",
                        "edition",
                        "newsletter",
                        "volume",
                        "vol",
                        "honey",
                        "innovation",
                        "grassroot",
                    )
                ):
                    full_url = urljoin(self.base_url, href)
                    issues.append({"url": full_url, "title": text})
        except Exception:
            logger.debug("Fallback discovery failed")

        return issues

    @staticmethod
    def _parse_index_page(html: str, page_url: str) -> list[dict[str, str]]:
        """Parse a newsletter archive index page for issue links."""
        parser = _NewsletterIndexParser()
        try:
            parser.feed(html)
        except Exception:
            logger.warning("HTML parse error on index page: %s", page_url)
        return parser.issue_links

    # ------------------------------------------------------------------
    # Issue scraping
    # ------------------------------------------------------------------

    async def _scrape_issue(
        self, issue_url: str, issue_title: str, seen_titles: set[str]
    ) -> list[RawCase]:
        """Scrape a single newsletter issue page for innovation entries."""
        html = await self.fetch(issue_url, use_cache=True)
        entries = self._parse_issue_page(html)

        # Fallback: regex-based extraction
        if not entries:
            entries = self._extract_entries_regex(html)

        cases: list[RawCase] = []
        for entry in entries:
            title = entry.get("title", "").strip()
            body = entry.get("body", "").strip()

            # Deduplicate
            dedup_key = title.lower() if title else body[:80].lower()
            if dedup_key in seen_titles:
                continue
            seen_titles.add(dedup_key)

            raw_case = self._entry_to_raw_case(entry, issue_url, issue_title)
            if raw_case is not None:
                cases.append(raw_case)

        return cases

    @staticmethod
    def _parse_issue_page(html: str) -> list[dict[str, str]]:
        """Parse a newsletter issue page for innovation entries using HTML parser."""
        parser = _NewsletterContentParser()
        try:
            parser.feed(html)
        except Exception:
            logger.warning("HTML parse error on issue page")
        return parser.entries

    @staticmethod
    def _extract_entries_regex(html: str) -> list[dict[str, str]]:
        """Fallback regex-based extraction of innovation entries."""
        entries: list[dict[str, str]] = []
        for match in _INNOVATION_BLOCK_RE.finditer(html):
            title = _strip_html(match.group(1)).strip()
            body = _strip_html(match.group(2)).strip()
            if len(body) >= 30:
                entry: dict[str, str] = {"title": title, "body": body}

                # Try to extract innovator
                innovator_match = _INNOVATOR_RE.search(body)
                if innovator_match:
                    entry["innovator"] = innovator_match.group(1).strip()

                # Try to extract location
                location_match = _LOCATION_RE.search(body)
                if location_match:
                    entry["location"] = location_match.group(1).strip()

                entries.append(entry)

        return entries

    def _entry_to_raw_case(
        self,
        entry: dict[str, str],
        issue_url: str,
        issue_title: str,
    ) -> RawCase | None:
        """Convert a parsed entry dict to a RawCase."""
        title = entry.get("title", "").strip()
        body = entry.get("body", "").strip()
        innovator = entry.get("innovator") or None
        location = entry.get("location") or None

        # Build raw text
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        text_parts.append(body)
        if innovator:
            text_parts.append(f"Innovator: {innovator}")
        if location:
            text_parts.append(f"Location: {location}")
        text_parts.append(f"Source: Honey Bee Network Newsletter — {issue_title}")

        raw_text = "\n".join(text_parts)

        if len(raw_text) < 20:
            return None

        try:
            return RawCase(
                source="honeybee",
                url_or_path=issue_url,
                raw_text=raw_text,
                title=title or None,
                innovator_name=innovator,
                location=location,
                category=None,
            )
        except Exception:
            logger.warning("Failed to create RawCase from HBN entry: %s", title)
            return None
