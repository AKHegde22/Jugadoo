#!/usr/bin/env python3
"""
Script 02: Extract structured SeedTuples from raw scraped data.

Uses LLM-assisted extraction to convert unstructured RawCase objects
into validated SeedTuple schemas for human review and curation.

Usage:
    python scripts/02_extract_seed_tuples.py
    python scripts/02_extract_seed_tuples.py --input data/raw/all_raw_cases.json --limit 150
    python scripts/02_extract_seed_tuples.py --finalize data/seeds/seed_candidates_reviewed.json
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
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import RawCase, SeedTuple, Domain
from jugaad_bench.data.seed_extractor import SeedExtractor
from jugaad_bench.utils.config import load_config, resolve_data_path

console = Console()
logger = logging.getLogger("jugaad_bench.extract")


async def extract_seeds(args: argparse.Namespace) -> None:
    """Run LLM-assisted seed extraction on raw cases."""
    config = load_config(args.config)
    seeds_dir = resolve_data_path(config, "seeds")
    raw_dir = resolve_data_path(config, "raw_data")

    # Load raw cases
    input_path = Path(args.input) if args.input else raw_dir / "all_raw_cases.json"
    if not input_path.exists():
        console.print(f"[bold red]Error: Input file not found: {input_path}[/]")
        console.print("[dim]Run script 01_scrape_seeds.py first to collect raw data.[/]")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_cases = [RawCase.model_validate(d) for d in raw_data]
    console.print(f"[bold]Loaded {len(raw_cases)} raw cases from {input_path}[/]")

    # Apply limit
    if args.limit and args.limit < len(raw_cases):
        raw_cases = raw_cases[:args.limit]
        console.print(f"[dim]Limited to first {args.limit} cases[/]")

    # Initialize extractor
    extractor = SeedExtractor(
        model=args.model or "gpt-4o",
    )

    # Extract seed tuples
    console.print(f"\n[bold cyan]Extracting seed tuples using {extractor.model}...[/]\n")

    candidates = await extractor.extract_seeds(
        raw_cases=raw_cases,
        target_count=config.data.seed_target_count,
        domain_distribution=config.data.domain_distribution,
    )

    console.print(f"\n[bold green]✓ Extracted {len(candidates)} seed tuple candidates[/]")

    # Deduplication is already handled internally by extract_seeds
    unique_candidates = candidates
    console.print(f"[bold]Extracted {len(unique_candidates)} unique seeds[/]")

    # Show domain distribution
    domain_counts = {}
    for seed in unique_candidates:
        domain_counts[seed.domain.value] = domain_counts.get(seed.domain.value, 0) + 1
    console.print("\n[bold]Domain Distribution:[/]")
    for domain, count in sorted(domain_counts.items()):
        target = config.data.domain_distribution.get(domain, 0)
        status = "✓" if count >= target else "✗"
        console.print(f"  {status} {domain}: {count} (target: {target})")

    # Save candidates for human review
    output_path = seeds_dir / "seed_candidates.json"
    candidates_data = [s.model_dump() for s in unique_candidates]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold]Candidates saved to {output_path}[/]")
    console.print(
        "[bold yellow]⚠ Please review and curate the candidates manually.[/]\n"
        "[dim]After review, save the final 100 seeds as data/seeds/seeds_100.json "
        "and run: python scripts/02_extract_seed_tuples.py --finalize data/seeds/seeds_100.json[/]"
    )


def finalize_seeds(args: argparse.Namespace) -> None:
    """Validate and finalize human-reviewed seed tuples."""
    config = load_config(args.config)
    seeds_dir = resolve_data_path(config, "seeds")

    input_path = Path(args.finalize)
    if not input_path.exists():
        console.print(f"[bold red]Error: File not found: {input_path}[/]")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)

    # Validate each seed
    seeds: list[SeedTuple] = []
    errors: list[tuple[int, str]] = []
    for i, data in enumerate(seeds_data):
        try:
            seed = SeedTuple.model_validate(data)
            seeds.append(seed)
        except Exception as e:
            errors.append((i, str(e)))

    if errors:
        console.print(f"\n[bold red]Validation errors in {len(errors)} entries:[/]")
        for idx, err in errors[:10]:
            console.print(f"  Entry {idx}: {err}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more")

    # Re-assign sequential IDs
    for i, seed in enumerate(seeds):
        seed.seed_id = f"SEED_{i+1:03d}"

    # Check distribution
    domain_counts = {}
    for seed in seeds:
        domain_counts[seed.domain.value] = domain_counts.get(seed.domain.value, 0) + 1

    console.print(f"\n[bold green]✓ Validated {len(seeds)} seeds[/]")
    console.print("\n[bold]Final Domain Distribution:[/]")
    target_dist = config.data.domain_distribution
    for domain in Domain:
        count = domain_counts.get(domain.value, 0)
        target = target_dist.get(domain.value, 0)
        status = "✓" if count >= target else "⚠"
        console.print(f"  {status} {domain.value}: {count}/{target}")

    # Save finalized seeds
    output_path = seeds_dir / "seeds_100.json"
    final_data = [s.model_dump() for s in seeds]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]✓ Finalized {len(seeds)} seeds saved to {output_path}[/]")
    console.print("[dim]Ready for mutation: python scripts/03_mutate_seeds.py[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured SeedTuples from raw scraped data.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Input raw cases JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Max raw cases to process")
    parser.add_argument("--model", type=str, default=None, help="LLM model for extraction (default: gpt-4o)")
    parser.add_argument("--finalize", type=str, default=None, help="Finalize reviewed seeds JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    if args.finalize:
        finalize_seeds(args)
    else:
        asyncio.run(extract_seeds(args))


if __name__ == "__main__":
    main()
