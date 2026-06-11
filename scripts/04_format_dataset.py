#!/usr/bin/env python3
"""
Script 04: Format mutation matrix into final benchmark datasets.

Uses the LLM Formatting Clerk to convert abstract mutation matrix rows
into natural language MCQ and Open Generation evaluation formats.

Usage:
    python scripts/04_format_dataset.py
    python scripts/04_format_dataset.py --input data/mutations/mutation_matrix_1000.json
    python scripts/04_format_dataset.py --limit 50  # Process only 50 rows
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import MutatedInstance
from jugaad_bench.data.formatting_clerk import FormattingClerk
from jugaad_bench.utils.config import load_config, resolve_data_path

console = Console()
logger = logging.getLogger("jugaad_bench.format")


async def run_formatting(args: argparse.Namespace) -> None:
    """Execute the formatting clerk."""
    config = load_config(args.config)
    mutations_dir = resolve_data_path(config, "mutations")
    benchmark_dir = resolve_data_path(config, "benchmark")

    # Load mutation matrix
    input_path = Path(args.input) if args.input else mutations_dir / "mutation_matrix_1000.json"
    if not input_path.exists():
        console.print(f"[bold red]Error: Mutation matrix not found: {input_path}[/]")
        console.print("[dim]Run script 03_mutate_seeds.py first.[/]")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        mutations_data = json.load(f)

    mutations = [MutatedInstance.model_validate(d) for d in mutations_data]
    console.print(f"[bold]Loaded {len(mutations)} mutation instances from {input_path}[/]")

    if args.limit and args.limit < len(mutations):
        mutations = mutations[:args.limit]
        console.print(f"[dim]Limited to first {args.limit} instances[/]")

    # Initialize formatting clerk
    clerk = FormattingClerk(
        model=args.model or "gpt-4o",
        max_retries=3,
    )

    console.print(
        f"\n[bold cyan]Formatting {len(mutations)} problems "
        f"into MCQ + Open Generation formats...[/]\n"
    )

    # Format all problems
    problems = await clerk.format_batch(
        mutations=mutations,
        console=console,
    )

    console.print(f"\n[bold green]✓ Formatted {len(problems)} benchmark problems[/]")

    # Save MCQ format
    mcq_path = benchmark_dir / "jugaad_reasoning_1k_mcq.json"
    mcq_data = []
    for p in problems:
        mcq_entry = {
            "problem_id": p.problem_id,
            "domain": p.domain.value,
            "metadata": p.metadata.model_dump(),
            "prompt_context": p.prompt_context,
            "applied_constraints": p.applied_constraints.model_dump(),
            "available_inventory": p.available_inventory,
            "mcq_options": p.mcq_options.model_dump(),
            "ground_truth_option": p.ground_truth_option,
        }
        mcq_data.append(mcq_entry)

    with open(mcq_path, "w", encoding="utf-8") as f:
        json.dump(mcq_data, f, indent=2, ensure_ascii=False)
    console.print(f"  MCQ dataset: {mcq_path}")

    # Save Open Generation format
    opengen_path = benchmark_dir / "jugaad_reasoning_1k_opengen.json"
    opengen_data = []
    for p in problems:
        opengen_entry = {
            "problem_id": p.problem_id,
            "domain": p.domain.value,
            "metadata": p.metadata.model_dump(),
            "prompt_context": p.prompt_context,
            "applied_constraints": p.applied_constraints.model_dump(),
            "available_inventory": p.available_inventory,
            "ground_truth_synthesis_rubric": p.ground_truth_synthesis_rubric.model_dump(),
        }
        opengen_data.append(opengen_entry)

    with open(opengen_path, "w", encoding="utf-8") as f:
        json.dump(opengen_data, f, indent=2, ensure_ascii=False)
    console.print(f"  Open-Gen dataset: {opengen_path}")

    # Save complete benchmark (both formats)
    full_path = benchmark_dir / "jugaad_reasoning_1k_full.json"
    full_data = [p.model_dump() for p in problems]
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"  Full dataset: {full_path}")

    # Domain breakdown
    domain_counts: dict[str, int] = {}
    gt_dist: dict[str, int] = {}
    for p in problems:
        domain_counts[p.domain.value] = domain_counts.get(p.domain.value, 0) + 1
        gt_dist[p.ground_truth_option] = gt_dist.get(p.ground_truth_option, 0) + 1

    console.print("\n[bold]Domain Distribution:[/]")
    for d, c in sorted(domain_counts.items()):
        console.print(f"  {d}: {c}")
    console.print("\n[bold]Ground Truth Option Distribution:[/]")
    for opt, c in sorted(gt_dist.items()):
        console.print(f"  {opt}: {c} ({c/len(problems)*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format mutation matrix into benchmark datasets.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Input mutation matrix JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, help="LLM model for formatting")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    asyncio.run(run_formatting(args))


if __name__ == "__main__":
    main()
