#!/usr/bin/env python3
"""
Script 05: Run the benchmark against all configured models.

Executes both MCQ and Open Generation evaluations under zero-shot
conditions with temperature=0.0 for deterministic, reproducible results.

Usage:
    python scripts/05_run_benchmark.py
    python scripts/05_run_benchmark.py --models gpt-4o claude-3.5-sonnet
    python scripts/05_run_benchmark.py --format mcq --limit 100
    python scripts/05_run_benchmark.py --dry-run  # Cost estimation only
    python scripts/05_run_benchmark.py --resume    # Resume from checkpoint
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
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import BenchmarkProblem
from jugaad_bench.eval.runner import BenchmarkRunner
from jugaad_bench.utils.config import load_config, resolve_data_path

console = Console()
logger = logging.getLogger("jugaad_bench.benchmark")


async def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the benchmark evaluation."""
    config = load_config(args.config)
    benchmark_dir = resolve_data_path(config, "benchmark")
    results_dir = resolve_data_path(config, "results")

    # Load benchmark dataset
    full_path = benchmark_dir / "jugaad_reasoning_1k_full.json"
    if not full_path.exists():
        console.print(f"[bold red]Error: Benchmark dataset not found: {full_path}[/]")
        console.print("[dim]Run script 04_format_dataset.py first.[/]")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        problems_data = json.load(f)

    problems = [BenchmarkProblem.model_validate(d) for d in problems_data]
    console.print(f"[bold]Loaded {len(problems)} benchmark problems[/]")

    if args.limit and args.limit < len(problems):
        problems = problems[:args.limit]
        console.print(f"[dim]Limited to {args.limit} problems[/]")

    # Filter models if specified
    all_model_configs = config.models.all_models
    if args.models:
        all_model_configs = [m for m in all_model_configs if m.name in args.models]
        if not all_model_configs:
            console.print(f"[bold red]No matching models found for: {args.models}[/]")
            available = [m.name for m in config.models.all_models]
            console.print(f"[dim]Available models: {available}[/]")
            return

    console.print(f"[bold]Models to evaluate: {[m.name for m in all_model_configs]}[/]")

    # Override dry_run from args
    if args.dry_run:
        config.eval.dry_run = True

    # Initialize runner
    runner = BenchmarkRunner(config=config)

    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN — estimating costs without API calls[/]\n")
        estimates = runner.dry_run(problems=problems)

        table = Table(title="Cost Estimation")
        table.add_column("Model", style="cyan")
        table.add_column("Format", style="green")
        table.add_column("Est. Input Tokens", justify="right")
        table.add_column("Est. Output Tokens", justify="right")
        table.add_column("Est. Cost (USD)", justify="right", style="bold")

        total_cost = 0.0
        for est in estimates:
            table.add_row(
                est["model"],
                est["format"],
                f"{est['input_tokens']:,}",
                f"{est['output_tokens']:,}",
                f"${est['cost']:.2f}",
            )
            total_cost += est["cost"]

        table.add_section()
        table.add_row("TOTAL", "", "", "", f"${total_cost:.2f}", style="bold red")

        console.print(table)
        return

    # Determine formats to run
    formats = []
    if args.format in (None, "mcq"):
        formats.append("mcq")
    if args.format in (None, "opengen"):
        formats.append("opengen")

    # Run benchmark
    console.print(f"\n[bold cyan]Running benchmark ({', '.join(formats)})...[/]\n")

    all_completions = await runner.run_all(
        problems=problems,
        model_configs=all_model_configs,
    )

    # Save summary
    summary = {
        "total_models": len(all_model_configs),
        "total_problems": len(problems),
        "formats": formats,
        "completions_per_model": {
            model_name: len(completions)
            for model_name, completions in all_completions.items()
        },
    }
    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold green]✓ Benchmark complete. Results saved to {results_dir}[/]")
    console.print(f"[dim]Next step: python scripts/06_run_judge.py[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark against configured models.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model names to evaluate (default: all configured)",
    )
    parser.add_argument(
        "--format", choices=["mcq", "opengen"], default=None,
        help="Run only one format (default: both)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
