"""
JugaadReasoning-1K Data Scrapers.

This package provides scrapers for collecting grassroots innovation data from:
- NIF Award Book PDFs (pdfplumber-based)
- NIF Innovation Portal (innovation.nif.org.in)
- Honey Bee Network / SRISTI newsletters (sristi.org/hbnew)
- YouTube channels (YouTube Data API v3 + youtube-transcript-api)
"""

from jugaad_bench.data.scrapers.base_scraper import BaseScraper
from jugaad_bench.data.scrapers.honeybee_scraper import HoneyBeeScraper
from jugaad_bench.data.scrapers.nif_pdf_scraper import NIFPDFScraper
from jugaad_bench.data.scrapers.nif_web_scraper import NIFWebScraper
from jugaad_bench.data.scrapers.youtube_scraper import YouTubeScraper

__all__ = [
    "BaseScraper",
    "HoneyBeeScraper",
    "NIFPDFScraper",
    "NIFWebScraper",
    "YouTubeScraper",
]
