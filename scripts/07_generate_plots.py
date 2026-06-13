#!/usr/bin/env python3
"""
Script 07: Generate publication-quality visualization plots.

Produces three core plots for the research paper:
1. Discriminative vs. Generative Gap (Grouped Bar Chart)
2. Performance Degradation Under Scarcity (Line Graph)
3. Domain Breakdown of Failure Modes (Stacked Bar Chart)

Usage:
    python scripts/07_generate_plots.py
    python scripts/07_generate_plots.py --output-dir figures/
    python scripts/07_generate_plots.py --style ieee
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import EvalResult
from jugaad_bench.analytics.plots import generate_all_plots
from jugaad_bench.analytics.result_logger import ResultLogger
from jugaad_bench.utils.config import load_config, resolve_data_path, find_project_root

console = Console()
logger = logging.getLogger("jugaad_bench.plots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality plots.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input eval_log.jsonl file (default: data/results/eval_log.jsonl)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: plots/)",
    )
    parser.add_argument(
        "--style", type=str, default=None,
        choices=["science", "ieee", "nature", "default"],
        help="Plot style (default: from config)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=None,
        choices=["pdf", "png", "svg", "eps"],
        help="Output formats (default: from config)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    config = load_config(args.config)
    results_dir = resolve_data_path(config, "results")
    root = find_project_root()

    # Load evaluation results
    input_path = Path(args.input) if args.input else results_dir / "eval_log.jsonl"
    if not input_path.exists():
        console.print(f"[bold red]Error: Evaluation log not found: {input_path}[/]")
        console.print("[dim]Run script 06_run_judge.py first.[/]")
        return

    result_logger = ResultLogger(output_path=input_path)
    results = result_logger.load_all_results()

    if not results:
        console.print("[bold red]No evaluation results found in the log.[/]")
        return

    console.print(f"[bold]Loaded {len(results)} evaluation results[/]")
    for r in results:
        console.print(f"  {r.model_under_test}: MCQ={r.mcq_global_accuracy:.3f}, OpenGen={r.open_gen_global_average_score:.2f}")

    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else root / config.plots.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override plot settings from args
    style = args.style or config.plots.style
    formats = args.formats or config.plots.formats
    dpi = config.plots.dpi

    console.print(f"\n[bold cyan]Generating plots (style={style}, formats={formats})...[/]\n")

    # Generate all plots
    generated_files = generate_all_plots(
        results=results,
        output_dir=output_dir,
        dpi=dpi,
        figsize=tuple(config.plots.figsize_double_column),
    )

    console.print(f"\n[bold green]✓ Generated {len(generated_files)} plot files:[/]")
    for f in generated_files:
        console.print(f"  {f}")


if __name__ == "__main__":
    main()
