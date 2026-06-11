"""
NIF Award Book PDF Scraper.

Parses NIF (National Innovation Foundation) Award Book PDFs using pdfplumber
to extract individual innovation entries as RawCase objects.

Heuristics used for segmentation:
- Numbered entry patterns (e.g., "1.", "01.", "[1]")
- Title patterns (ALL-CAPS lines or bold-style text)
- Innovator name + location patterns ("Name — Location" or "by Name, State")
- Section boundary markers (horizontal rules, page breaks with headers)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber

from jugaad_bench.models import RawCase
from jugaad_bench.data.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Regex patterns for innovation entry detection
# --------------------------------------------------------------------------

# Matches numbered entries like "1.", "01.", "1)", "(1)", "[1]"
_NUMBERED_ENTRY_RE = re.compile(
    r"^[\s]*(?:\[?\(?\d{1,3}[\.\)\]]+)\s+",
    re.MULTILINE,
)

# Matches lines that look like titles (mostly uppercase, min 5 chars)
_TITLE_LINE_RE = re.compile(
    r"^[\s]*[A-Z][A-Z\s\-/:,]{4,}$",
    re.MULTILINE,
)

# Matches innovator attribution lines:
# "Innovator: Name" or "By Name" or "Name, State" or "Name – District, State"
_INNOVATOR_RE = re.compile(
    r"(?:Innovator\s*[:\-–]\s*|(?:By|Developed\s+by|Invented\s+by)\s+)"
    r"([A-Z][a-zA-Z\s\.]+)",
    re.IGNORECASE,
)

# Matches location patterns: "State" or "District, State" or "Village, District, State"
_LOCATION_RE = re.compile(
    r"(?:State\s*[:\-–]\s*|(?:from|of)\s+)"
    r"([A-Za-z\s,]+?)(?:\.|$|\n)",
    re.IGNORECASE,
)

# Category markers often found in NIF books
_CATEGORY_RE = re.compile(
    r"(?:Category|Sector|Domain)\s*[:\-–]\s*(.+?)(?:\.|$|\n)",
    re.IGNORECASE,
)

# Detects section headers that split between innovations
_SECTION_HEADER_RE = re.compile(
    r"^[\s]*(?:CHAPTER|SECTION|PART)\s+[IVXLCDM\d]+",
    re.IGNORECASE | re.MULTILINE,
)


class NIFPDFScraper(BaseScraper):
    """Scraper that extracts innovation cases from NIF Award Book PDFs.

    Args:
        pdf_directory: Path to the directory containing NIF PDF files.
        min_entry_length: Minimum character count for a text block to be
            considered a valid innovation entry.
    """

    def __init__(
        self,
        pdf_directory: str | Path,
        min_entry_length: int = 100,
    ) -> None:
        super().__init__(source_name="nif_pdf")
        self.pdf_directory = Path(pdf_directory)
        self.min_entry_length = min_entry_length

        if not self.pdf_directory.exists():
            logger.warning(
                "PDF directory does not exist, creating: %s", self.pdf_directory
            )
            self.pdf_directory.mkdir(parents=True, exist_ok=True)

    async def scrape(self) -> list[RawCase]:
        """Parse all PDFs in the directory and extract innovation entries.

        Returns:
            List of RawCase objects, one per detected innovation entry.
        """
        cases: list[RawCase] = []
        pdf_files = sorted(self.pdf_directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in %s", self.pdf_directory)
            return cases

        for pdf_path in pdf_files:
            logger.info("Processing PDF: %s", pdf_path.name)
            try:
                entries = self._extract_entries_from_pdf(pdf_path)
                for entry in entries:
                    case = self._entry_to_raw_case(entry, pdf_path)
                    if case is not None:
                        cases.append(case)
                logger.info(
                    "Extracted %d entries from %s", len(entries), pdf_path.name
                )
            except Exception:
                logger.exception("Failed to process PDF: %s", pdf_path.name)

        logger.info("Total NIF PDF entries extracted: %d", len(cases))
        return cases

    # ------------------------------------------------------------------
    # PDF parsing
    # ------------------------------------------------------------------

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract all text from a PDF, handling multi-column layouts.

        Uses pdfplumber's word-level extraction with bounding box analysis
        to detect and properly order multi-column text.
        """
        all_text_parts: list[str] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = self._extract_page_text(page)
                    if page_text.strip():
                        all_text_parts.append(page_text)
                except Exception:
                    logger.warning(
                        "Failed to extract page %d from %s",
                        page_num,
                        pdf_path.name,
                    )

        return "\n\n".join(all_text_parts)

    def _extract_page_text(self, page: pdfplumber.page.Page) -> str:
        """Extract text from a single page, detecting multi-column layout.

        If the page has two distinct columns (words clustered on left and
        right halves), we extract each column separately and concatenate.
        """
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
        )

        if not words:
            # Fallback to simple text extraction
            return page.extract_text() or ""

        # Determine if this is a multi-column layout
        page_width = page.width
        midpoint = page_width / 2.0

        left_words = [w for w in words if float(w["x0"]) < midpoint - 20]
        right_words = [w for w in words if float(w["x0"]) >= midpoint + 20]

        # Consider it multi-column if both sides have significant content
        if len(left_words) > 10 and len(right_words) > 10:
            left_text = self._words_to_text(left_words)
            right_text = self._words_to_text(right_words)
            return f"{left_text}\n\n{right_text}"

        # Single column: use default extraction
        return page.extract_text() or ""

    @staticmethod
    def _words_to_text(words: list[dict]) -> str:
        """Convert a list of pdfplumber word dicts to continuous text.

        Groups words by y-coordinate (line) and joins with spaces.
        """
        if not words:
            return ""

        # Sort by vertical then horizontal position
        sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

        lines: list[str] = []
        current_line_words: list[str] = []
        current_top: float = float(sorted_words[0]["top"])
        line_tolerance = 5.0  # pixels

        for word in sorted_words:
            word_top = float(word["top"])
            if abs(word_top - current_top) > line_tolerance:
                # New line
                if current_line_words:
                    lines.append(" ".join(current_line_words))
                current_line_words = [word["text"]]
                current_top = word_top
            else:
                current_line_words.append(word["text"])

        if current_line_words:
            lines.append(" ".join(current_line_words))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Entry segmentation
    # ------------------------------------------------------------------

    def _extract_entries_from_pdf(self, pdf_path: Path) -> list[dict[str, str]]:
        """Split full PDF text into individual innovation entry blocks.

        Returns a list of dicts with keys: 'text', 'title' (optional),
        'innovator' (optional), 'location' (optional), 'category' (optional).
        """
        full_text = self._extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            return []

        # Try numbered-entry segmentation first (most reliable)
        entries = self._segment_by_numbered_entries(full_text)

        # Fall back to title-based segmentation
        if len(entries) < 3:
            entries = self._segment_by_titles(full_text)

        # Fall back to page-break segmentation
        if len(entries) < 3:
            entries = self._segment_by_page_breaks(full_text)

        # Parse metadata from each entry
        parsed: list[dict[str, str]] = []
        for text_block in entries:
            if len(text_block.strip()) < self.min_entry_length:
                continue
            parsed.append(self._parse_entry_metadata(text_block))

        return parsed

    def _segment_by_numbered_entries(self, text: str) -> list[str]:
        """Segment text by numbered entry markers (e.g., '1.', '2.', etc.)."""
        matches = list(_NUMBERED_ENTRY_RE.finditer(text))
        if len(matches) < 2:
            return []

        segments: list[str] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segment = text[start:end].strip()
            segments.append(segment)

        return segments

    def _segment_by_titles(self, text: str) -> list[str]:
        """Segment text by uppercase title lines."""
        matches = list(_TITLE_LINE_RE.finditer(text))
        if len(matches) < 2:
            return []

        # Filter out section headers and very short matches
        title_matches = [
            m
            for m in matches
            if not _SECTION_HEADER_RE.match(m.group())
            and len(m.group().strip()) >= 5
        ]

        if len(title_matches) < 2:
            return []

        segments: list[str] = []
        for i, match in enumerate(title_matches):
            start = match.start()
            end = (
                title_matches[i + 1].start()
                if i + 1 < len(title_matches)
                else len(text)
            )
            segment = text[start:end].strip()
            segments.append(segment)

        return segments

    def _segment_by_page_breaks(self, text: str) -> list[str]:
        """Segment text by double-newline page breaks."""
        blocks = re.split(r"\n{3,}", text)
        return [b.strip() for b in blocks if len(b.strip()) >= self.min_entry_length]

    def _parse_entry_metadata(self, text: str) -> dict[str, str]:
        """Extract title, innovator, location, and category from entry text."""
        result: dict[str, str] = {"text": text}

        # Extract title from the first non-empty line
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines:
            first_line = lines[0]
            # Remove leading numbering
            title_candidate = re.sub(r"^[\[\(]?\d{1,3}[\.\)\]]+\s*", "", first_line)
            if 5 <= len(title_candidate) <= 200:
                result["title"] = title_candidate

        # Extract innovator name
        innovator_match = _INNOVATOR_RE.search(text)
        if innovator_match:
            result["innovator"] = innovator_match.group(1).strip()

        # Extract location
        location_match = _LOCATION_RE.search(text)
        if location_match:
            result["location"] = location_match.group(1).strip()

        # Extract category
        category_match = _CATEGORY_RE.search(text)
        if category_match:
            result["category"] = category_match.group(1).strip()

        return result

    # ------------------------------------------------------------------
    # RawCase construction
    # ------------------------------------------------------------------

    def _entry_to_raw_case(
        self, entry: dict[str, str], pdf_path: Path
    ) -> RawCase | None:
        """Convert a parsed entry dict into a RawCase.

        Returns None if the entry text is too short to be meaningful.
        """
        text = entry.get("text", "")
        if len(text.strip()) < self.min_entry_length:
            return None

        try:
            return RawCase(
                source="nif_pdf",
                url_or_path=str(pdf_path),
                raw_text=text,
                title=entry.get("title"),
                innovator_name=entry.get("innovator"),
                location=entry.get("location"),
                category=entry.get("category"),
            )
        except Exception:
            logger.warning("Failed to create RawCase from entry in %s", pdf_path.name)
            return None
