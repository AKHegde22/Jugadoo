#!/usr/bin/env python3
"""
Script 01: Scrape raw innovation data from configured sources.

Runs all enabled scrapers (NIF PDF, NIF Web, Honey Bee, YouTube) and
saves raw candidate cases to data/raw/ for subsequent LLM-assisted extraction.

Usage:
    python scripts/01_scrape_seeds.py
    python scripts/01_scrape_seeds.py --sources nif_pdf nif_web
    python scripts/01_scrape_seeds.py --pdf-dir /path/to/pdfs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import RawCase
from jugaad_bench.utils.config import load_config, resolve_data_path

console = Console()
logger = logging.getLogger("jugaad_bench.scrape")


async def run_scrapers(args: argparse.Namespace) -> None:
    """Execute all enabled scrapers and save results."""
    config = load_config(args.config)
    raw_dir = resolve_data_path(config, "raw_data")

    all_cases: list[RawCase] = []
    sources = args.sources or ["nif_pdf", "nif_web", "honeybee", "youtube"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # --- NIF PDF Scraper ---
        if "nif_pdf" in sources:
            task = progress.add_task("Scraping NIF Award Book PDFs...", total=None)
            try:
                from jugaad_bench.data.scrapers.nif_pdf_scraper import NIFPDFScraper

                pdf_dir = args.pdf_dir or config.scrapers.nif_pdf.get("pdf_directory", "data/raw/nif_pdfs") if config.scrapers and config.scrapers.nif_pdf else "data/raw/nif_pdfs"
                project_root = Path(__file__).resolve().parent.parent
                pdf_path = project_root / pdf_dir
                pdf_path.mkdir(parents=True, exist_ok=True)

                scraper = NIFPDFScraper(pdf_directory=pdf_path)
                cases = await scraper.scrape()
                all_cases.extend(cases)
                progress.update(task, description=f"✓ NIF PDF: {len(cases)} cases extracted")
                logger.info(f"NIF PDF scraper: extracted {len(cases)} cases")
            except Exception as e:
                progress.update(task, description=f"✗ NIF PDF: {e}")
                logger.error(f"NIF PDF scraper failed: {e}", exc_info=True)

        # --- NIF Web Scraper ---
        if "nif_web" in sources:
            task = progress.add_task("Scraping NIF Innovation Portal...", total=None)
            try:
                from jugaad_bench.data.scrapers.nif_web_scraper import NIFWebScraper

                scraper_config = config.scrapers.nif_web if config.scrapers and config.scrapers.nif_web else {}
                scraper = NIFWebScraper(
                    base_url=scraper_config.get("base_url", "https://innovation.nif.org.in"),
                    rate_limit_seconds=scraper_config.get("rate_limit_seconds", 2.0),
                    max_pages=args.max_pages or scraper_config.get("max_pages", 500),
                )
                cases = await scraper.scrape()
                all_cases.extend(cases)
                progress.update(task, description=f"✓ NIF Web: {len(cases)} cases extracted")
                logger.info(f"NIF Web scraper: extracted {len(cases)} cases")
            except Exception as e:
                progress.update(task, description=f"✗ NIF Web: {e}")
                logger.error(f"NIF Web scraper failed: {e}", exc_info=True)

        # --- Honey Bee Scraper ---
        if "honeybee" in sources:
            task = progress.add_task("Scraping Honey Bee Network archives...", total=None)
            try:
                from jugaad_bench.data.scrapers.honeybee_scraper import HoneyBeeScraper

                scraper_config = config.scrapers.honeybee if config.scrapers and config.scrapers.honeybee else {}
                scraper = HoneyBeeScraper(
                    base_url=scraper_config.get("base_url", "https://sristi.org/hbnew"),
                    rate_limit_seconds=scraper_config.get("rate_limit_seconds", 2.0),
                )
                cases = await scraper.scrape()
                all_cases.extend(cases)
                progress.update(task, description=f"✓ Honey Bee: {len(cases)} cases extracted")
                logger.info(f"Honey Bee scraper: extracted {len(cases)} cases")
            except Exception as e:
                progress.update(task, description=f"✗ Honey Bee: {e}")
                logger.error(f"Honey Bee scraper failed: {e}", exc_info=True)

        # --- YouTube Scraper ---
        if "youtube" in sources:
            task = progress.add_task("Extracting from YouTube channels...", total=None)
            try:
                from jugaad_bench.data.scrapers.youtube_scraper import YouTubeScraper

                scraper_config = config.scrapers.youtube if config.scrapers and config.scrapers.youtube else {}
                scraper = YouTubeScraper(
                    channels=scraper_config.get("channels", []),
                    max_videos_per_channel=scraper_config.get("max_videos_per_channel", 100),
                )
                cases = await scraper.scrape()
                all_cases.extend(cases)
                progress.update(task, description=f"✓ YouTube: {len(cases)} cases extracted")
                logger.info(f"YouTube scraper: extracted {len(cases)} cases")
            except Exception as e:
                progress.update(task, description=f"✗ YouTube: {e}")
                logger.error(f"YouTube scraper failed: {e}", exc_info=True)

    # Save all cases
    output_path = raw_dir / "all_raw_cases.json"
    cases_data = [c.model_dump() for c in all_cases]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]✓ Total: {len(all_cases)} raw cases saved to {output_path}[/]")

    # Print domain distribution estimate
    domain_counts: dict[str, int] = {}
    for case in all_cases:
        cat = case.category or "unknown"
        domain_counts[cat] = domain_counts.get(cat, 0) + 1
    if domain_counts:
        console.print("\n[bold]Category Distribution:[/]")
        for cat, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {cat}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape raw innovation data from configured sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["nif_pdf", "nif_web", "honeybee", "youtube"],
        help="Which scrapers to run (default: all enabled)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Override directory containing NIF PDF files",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override maximum pages to scrape from web sources",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    asyncio.run(run_scrapers(args))


if __name__ == "__main__":
    main()
