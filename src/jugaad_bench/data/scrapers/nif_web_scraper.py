"""
NIF Innovation Portal Web Scraper.

Scrapes the NIF Innovation Portal (innovation.nif.org.in) using its AJAX search API
to extract innovation entries.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import httpx

from jugaad_bench.models import RawCase
from jugaad_bench.data.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class NIFWebScraper(BaseScraper):
    def __init__(
        self,
        base_url: str = "https://innovation.nif.org.in",
        rate_limit_seconds: float = 2.0,
        max_pages: int = 50,
    ) -> None:
        super().__init__(
            source_name="nif_web",
            rate_limit_seconds=rate_limit_seconds,
        )
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        # 1: Engineering, 2: Veterinary, 3: Agriculture, 4: Human Health, etc.
        self._categories = [1, 2, 3, 4] 

    async def scrape(self) -> list[RawCase]:
        all_cases: list[RawCase] = []
        seen_titles: set[str] = set()
        
        api_url = f"{self.base_url}/innovation/serach" # Yes, 'serach' is the typo on NIF

        for cat_id in self._categories:
            logger.info("Scraping category ID: %d", cat_id)
            for page in range(self.max_pages):
                start = page * 10
                
                # We need to manually sleep because we aren't using self.fetch for POST
                await self._rate_limit()
                
                try:
                    response = await self._client.post(
                        api_url,
                        data={
                            "start": start,
                            "cat_id": cat_id,
                            "view_type": "list"
                        }
                    )
                    response.raise_for_status()
                    html = response.text
                except Exception as e:
                    logger.warning("Error fetching cat %d page %d: %s", cat_id, page, e)
                    break
                    
                if not html.strip():
                    break
                    
                # Extract cards via simple regex
                # Each item is in <div class="recent-listing-box-container-item list-view-item">...
                blocks = html.split('<div class="recent-listing-box-container-item list-view-item">')
                
                cases_found = 0
                for block in blocks[1:]:
                    title_match = re.search(r'<h3>(.*?)</h3>', block)
                    link_match = re.search(r'<a href="([^"]*innovation/detail/[^"]*)"', block)
                    innovator_match = re.search(r'<span><strong>By\s*:\s*</strong>([^<]*)</span>', block)
                    desc_match = re.search(r'<p>(.*?)</p>', block, re.DOTALL)
                    
                    if title_match and desc_match:
                        title = title_match.group(1).strip()
                        url = link_match.group(1) if link_match else ""
                        innovator = innovator_match.group(1).strip() if innovator_match else None
                        desc = desc_match.group(1).strip()
                        
                        # Clean up description (remove 'View More' link)
                        desc = re.sub(r'<a[^>]*>.*?</a>', '', desc).strip()
                        
                        dedup_key = title.lower()
                        if dedup_key in seen_titles:
                            continue
                        seen_titles.add(dedup_key)
                        
                        raw_text = f"Title: {title}\nDescription: {desc}\n"
                        if innovator:
                            raw_text += f"Innovator: {innovator}\n"
                            
                        cases_found += 1
                        all_cases.append(
                            RawCase(
                                source="nif_web",
                                url_or_path=url,
                                raw_text=raw_text,
                                title=title,
                                innovator_name=innovator,
                                location=None,
                                category=str(cat_id)
                            )
                        )
                        
                if cases_found == 0:
                    break
                    
        return all_cases
