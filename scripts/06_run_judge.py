#!/usr/bin/env python3
"""
Script 06: Run LLM-as-a-Judge on open-generation model outputs.

Applies the two-stage verification protocol:
1. Python Keyword Guard Filter (auto-fail on forbidden terms)
2. LLM-as-a-Judge (structured scoring on clean responses)

Usage:
    python scripts/06_run_judge.py
    python scripts/06_run_judge.py --models gpt-4o --limit 100
    python scripts/06_run_judge.py --calibrate data/results/human_labels.json
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

from jugaad_bench.models import (
    BenchmarkProblem,
    CompletionResult,
    JudgeScore,
    KeywordGuardResult,
    FilterResult,
    EvalResult,
)
from jugaad_bench.eval.keyword_guard import KeywordGuard
from jugaad_bench.eval.judge import JugaadJudge
from jugaad_bench.eval.metrics import (
    mcq_accuracy,
    open_gen_average_score,
    cohens_kappa,
    domain_breakdown,
    failure_mode_analysis,
    budget_tier_analysis,
    aggregate_eval_result,
)
from jugaad_bench.analytics.result_logger import ResultLogger
from jugaad_bench.utils.config import load_config, resolve_data_path

console = Console()
logger = logging.getLogger("jugaad_bench.judge")


async def run_judge(args: argparse.Namespace) -> None:
    """Execute the verification protocol on open-gen outputs."""
    config = load_config(args.config)
    benchmark_dir = resolve_data_path(config, "benchmark")
    results_dir = resolve_data_path(config, "results")

    # Load benchmark problems
    full_path = benchmark_dir / "jugaad_reasoning_1k_full.json"
    if not full_path.exists():
        console.print(f"[bold red]Error: Benchmark dataset not found: {full_path}[/]")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        problems_data = json.load(f)
    problems = [BenchmarkProblem.model_validate(d) for d in problems_data]
    problem_map = {p.problem_id: p for p in problems}

    # Find all model completion files
    completion_files = sorted(results_dir.glob("*_opengen_completions.jsonl"))
    if not completion_files:
        console.print("[bold red]No open-gen completion files found.[/]")
        console.print(f"[dim]Expected files matching: {results_dir}/*_opengen_completions.jsonl[/]")
        console.print("[dim]Run script 05_run_benchmark.py first.[/]")
        return

    if args.models:
        completion_files = [
            f for f in completion_files
            if any(m in f.stem for m in args.models)
        ]

    console.print(f"[bold]Found {len(completion_files)} completion files to judge[/]")

    # Initialize components
    keyword_guard = KeywordGuard()
    judge = JugaadJudge(
        provider=config.eval.judge.provider,
        model=config.eval.judge.model,
        temperature=config.eval.judge.temperature,
    )
    result_logger = ResultLogger(output_path=results_dir / "eval_log.jsonl")

    for comp_file in completion_files:
        model_name = comp_file.stem.replace("_opengen_completions", "")
        console.print(f"\n[bold cyan]Judging {model_name}...[/]")

        # Load completions
        completions: list[CompletionResult] = []
        with open(comp_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completions.append(CompletionResult.model_validate_json(line))

        if args.limit:
            completions = completions[:args.limit]

        console.print(f"  Loaded {len(completions)} completions")

        # Stage 1: Keyword Guard
        guard_results: list[KeywordGuardResult] = []
        clean_completions: list[tuple[CompletionResult, BenchmarkProblem]] = []
        auto_fail_count = 0

        for comp in completions:
            problem = problem_map.get(comp.problem_id)
            if problem is None:
                logger.warning(f"Problem {comp.problem_id} not found in benchmark")
                continue

            guard_result = keyword_guard.check(
                response=comp.raw_output,
                forbidden_keywords=problem.ground_truth_synthesis_rubric.forbidden_keywords,
                problem_id=comp.problem_id,
                model_name=model_name,
            )
            guard_results.append(guard_result)

            if guard_result.filter_result == FilterResult.AUTO_FAIL:
                auto_fail_count += 1
            else:
                clean_completions.append((comp, problem))

        console.print(
            f"  Keyword Guard: {auto_fail_count} auto-fails, "
            f"{len(clean_completions)} clean for judge"
        )

        # Stage 2: LLM-as-a-Judge on clean responses
        judge_scores: list[JudgeScore] = []
        if clean_completions:
            pairs = [(problem, comp.raw_output) for comp, problem in clean_completions]
            judge_scores = await judge.batch_evaluate(
                problems=pairs,
            )

        # Build auto-fail scores for keyword-guarded responses
        all_scores: list[JudgeScore] = []
        judge_idx = 0
        for guard_result in guard_results:
            if guard_result.filter_result == FilterResult.AUTO_FAIL:
                all_scores.append(JudgeScore(
                    reasoning=f"AUTO_FAIL: forbidden keywords detected: {guard_result.triggered_keywords}",
                    constraint_adherence=0,
                    inventory_utilization=0,
                    physical_viability=0,
                    total_score=0,
                ))
            else:
                if judge_idx < len(judge_scores):
                    all_scores.append(judge_scores[judge_idx])
                    judge_idx += 1

        # Load MCQ completions if available for full eval result
        mcq_file = results_dir / f"{model_name}_mcq_completions.jsonl"
        mcq_completions: list[CompletionResult] = []
        if mcq_file.exists():
            with open(mcq_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        mcq_completions.append(CompletionResult.model_validate_json(line))

        # Compute aggregate metrics
        ground_truth = {p.problem_id: p.ground_truth_option for p in problems}
        eval_result = aggregate_eval_result(
            model_name=model_name,
            mcq_completions=mcq_completions,
            opengen_completions=completions,
            judge_scores=all_scores,
            guard_results=guard_results,
            problems=problems,
        )

        # Log result
        result_logger.log_eval_result(eval_result)

        # Display summary
        table = Table(title=f"Results: {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="bold")
        table.add_row("MCQ Accuracy", f"{eval_result.mcq_global_accuracy:.3f}")
        table.add_row("Open Gen Avg Score", f"{eval_result.open_gen_global_average_score:.2f}/3.0")
        table.add_row("Constraint Violations", str(eval_result.failure_modes.constraint_violations))
        table.add_row("Physical Hallucinations", str(eval_result.failure_modes.physical_hallucinations))
        table.add_row("Task Abandonment", str(eval_result.failure_modes.task_abandonment))
        console.print(table)

        # Save detailed judge scores
        scores_path = results_dir / f"{model_name}_judge_scores.jsonl"
        with open(scores_path, "w", encoding="utf-8") as f:
            for score in all_scores:
                f.write(score.model_dump_json() + "\n")

    console.print(f"\n[bold green]✓ Judging complete. Results logged to {results_dir / 'eval_log.jsonl'}[/]")
    console.print("[dim]Next step: python scripts/07_generate_plots.py[/]")


def run_calibration(args: argparse.Namespace) -> None:
    """Compute Cohen's Kappa between human and LLM judge labels."""
    config = load_config(args.config)

    human_labels_path = Path(args.calibrate)
    if not human_labels_path.exists():
        console.print(f"[bold red]Error: Human labels file not found: {human_labels_path}[/]")
        return

    with open(human_labels_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    # Expected format: [{"problem_id": "...", "total_score": 0-3}, ...]
    human_labels = {d["problem_id"]: d["total_score"] for d in human_data}

    # Load corresponding LLM judge scores
    results_dir = resolve_data_path(config, "results")
    judge_files = sorted(results_dir.glob("*_judge_scores.jsonl"))

    for judge_file in judge_files:
        model_name = judge_file.stem.replace("_judge_scores", "")
        llm_labels: dict[str, int] = {}

        with open(judge_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    score = JudgeScore.model_validate_json(line)
                    # Match by position (human labels should be in same order)
                    problem_ids = list(human_labels.keys())
                    if i < len(problem_ids):
                        llm_labels[problem_ids[i]] = score.total_score

        # Compute kappa on overlapping entries
        common_ids = set(human_labels.keys()) & set(llm_labels.keys())
        if len(common_ids) < 10:
            console.print(f"[yellow]Insufficient overlap for {model_name}: {len(common_ids)} entries[/]")
            continue

        h_labels = [human_labels[pid] for pid in sorted(common_ids)]
        l_labels = [llm_labels[pid] for pid in sorted(common_ids)]

        kappa = cohens_kappa(h_labels, l_labels)
        threshold = config.eval.judge.kappa_threshold

        status = "✓ VALID" if kappa >= threshold else "✗ INVALID"
        color = "green" if kappa >= threshold else "red"

        console.print(
            f"[bold {color}]{status}[/] {model_name}: "
            f"κ = {kappa:.3f} (threshold: {threshold:.2f}, "
            f"n = {len(common_ids)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge verification protocol.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--calibrate", type=str, default=None,
        help="Path to human labels JSON for Cohen's Kappa calibration",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    if args.calibrate:
        run_calibration(args)
    else:
        asyncio.run(run_judge(args))


if __name__ == "__main__":
    main()
