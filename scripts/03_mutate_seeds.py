#!/usr/bin/env python3
"""
Script 03: Mutate 100 seed tuples into 1,000-row constraint matrix.

Cross-multiplies each seed with 10 unique constraint profiles from
the 27-combination constraint parameter space.

Usage:
    python scripts/03_mutate_seeds.py
    python scripts/03_mutate_seeds.py --input data/seeds/seeds_100.json --limit 50
    python scripts/03_mutate_seeds.py --dry-run
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

from jugaad_bench.models import SeedTuple
from jugaad_bench.data.mutation_engine import MutationEngine
from jugaad_bench.utils.config import load_config, resolve_data_path

console = Console()
logger = logging.getLogger("jugaad_bench.mutate")


async def run_mutations(args: argparse.Namespace) -> None:
    """Execute the mutation engine."""
    config = load_config(args.config)
    seeds_dir = resolve_data_path(config, "seeds")
    mutations_dir = resolve_data_path(config, "mutations")

    # Load seeds
    input_path = Path(args.input) if args.input else seeds_dir / "seeds_100.json"
    if not input_path.exists():
        console.print(f"[bold red]Error: Seeds file not found: {input_path}[/]")
        console.print("[dim]Run script 02_extract_seed_tuples.py first.[/]")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)

    seeds = [SeedTuple.model_validate(d) for d in seeds_data]
    console.print(f"[bold]Loaded {len(seeds)} seed tuples from {input_path}[/]")

    # Apply limit
    if args.limit and args.limit < len(seeds):
        seeds = seeds[:args.limit]
        console.print(f"[dim]Limited to first {args.limit} seeds[/]")

    # Initialize engine
    engine = MutationEngine(config=config)

    expected_total = len(seeds) * config.data.mutations_per_seed
    console.print(
        f"\n[bold cyan]Generating {expected_total} mutations "
        f"({len(seeds)} seeds × {config.data.mutations_per_seed} mutations each)...[/]\n"
    )

    if args.dry_run:
        # Show what would be generated
        console.print("[bold yellow]DRY RUN — no LLM calls will be made[/]\n")
        constraint_combos = engine.generate_constraint_space()
        console.print(f"Total constraint combinations: {len(constraint_combos)}")
        for i, seed in enumerate(seeds[:3]):
            profiles = engine.select_profiles_for_seed(seed, constraint_combos)
            console.print(f"\n[bold]Seed {seed.seed_id}:[/] {seed.target_goal[:80]}...")
            for j, profile in enumerate(profiles[:3]):
                console.print(
                    f"  Mutation {j+1}: {profile.budget} | {profile.environment} | {profile.infrastructure}"
                )
            if len(profiles) > 3:
                console.print(f"  ... and {len(profiles) - 3} more")
        console.print(f"\n[bold]Would generate {expected_total} total mutations.[/]")
        return

    # Run mutations
    mutations = await engine.mutate_all(seeds=seeds, console=console)

    # Save mutation matrix
    output_path = mutations_dir / "mutation_matrix_1000.json"
    mutations_data = [m.model_dump() for m in mutations]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mutations_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]✓ Generated {len(mutations)} mutations saved to {output_path}[/]")

    # Verify distribution
    budget_counts: dict[str, int] = {}
    env_counts: dict[str, int] = {}
    infra_counts: dict[str, int] = {}
    for m in mutations:
        budget_counts[m.applied_constraints.budget] = budget_counts.get(m.applied_constraints.budget, 0) + 1
        env_counts[m.applied_constraints.environment] = env_counts.get(m.applied_constraints.environment, 0) + 1
        infra_counts[m.applied_constraints.infrastructure] = infra_counts.get(m.applied_constraints.infrastructure, 0) + 1

    console.print("\n[bold]Constraint Distribution:[/]")
    console.print("  Budget:")
    for k, v in sorted(budget_counts.items()):
        console.print(f"    {k}: {v}")
    console.print("  Environment:")
    for k, v in sorted(env_counts.items()):
        console.print(f"    {k}: {v}")
    console.print("  Infrastructure:")
    for k, v in sorted(infra_counts.items()):
        console.print(f"    {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutate seeds into constraint matrix.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Input seeds JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Max seeds to process")
    parser.add_argument("--dry-run", action="store_true", help="Show mutations without LLM calls")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    asyncio.run(run_mutations(args))


if __name__ == "__main__":
    main()
