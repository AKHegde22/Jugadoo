"""
Benchmark execution engine for JugaadReasoning-1K.

Runs MCQ and open-generation prompts across all configured models, with
checkpoint support, Rich progress bars, and dry-run cost estimation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from jugaad_bench.eval.llm_client import BaseLLMProvider, LLMClientFactory
from jugaad_bench.models import BenchmarkProblem, CompletionResult, ModelConfig, PipelineConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_MCQ_SYSTEM_PROMPT = (
    "You are an expert in practical problem-solving under resource constraints. "
    "Answer the multiple-choice question below by selecting the BEST option."
)

_OPENGEN_SYSTEM_PROMPT = (
    "You are a resourceful engineer who must solve real-world problems using "
    "only the materials at hand. Describe a practical, physically viable solution."
)


def _format_mcq_prompt(problem: BenchmarkProblem) -> str:
    """Build the MCQ prompt text."""
    opts = problem.mcq_options
    constraints = problem.applied_constraints
    inventory = ", ".join(problem.available_inventory)

    return (
        f"{problem.prompt_context}\n\n"
        f"Constraints:\n"
        f"- Budget: {constraints.budget}\n"
        f"- Environment: {constraints.environment}\n"
        f"- Infrastructure: {constraints.infrastructure}\n\n"
        f"Available materials: {inventory}\n\n"
        f"Options:\n"
        f"A) {opts.A}\n"
        f"B) {opts.B}\n"
        f"C) {opts.C}\n"
        f"D) {opts.D}\n\n"
        f"Choose the best option (A, B, C, or D):"
    )


def _format_opengen_prompt(problem: BenchmarkProblem) -> str:
    """Build the open-generation prompt text."""
    constraints = problem.applied_constraints
    inventory = "\n".join(f"  - {item}" for item in problem.available_inventory)

    return (
        f"{problem.prompt_context}\n\n"
        f"Constraints:\n"
        f"- Budget: {constraints.budget}\n"
        f"- Environment: {constraints.environment}\n"
        f"- Infrastructure: {constraints.infrastructure}\n\n"
        f"Available inventory:\n{inventory}\n\n"
        f"Describe a solution using only the available materials. "
        f"Explain the physical mechanism and how each material is used."
    )


_OPTION_RE = re.compile(r"\b([A-D])\b")


def _extract_option(response: str) -> str | None:
    """Extract the first A/B/C/D letter from a model response."""
    # First try to find a clear answer pattern
    for pattern in [
        r"(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\b([A-D])\)",
        r"\b([A-D])\b",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

_CHECKPOINT_DIR = Path("eval_outputs") / "checkpoints"


def _checkpoint_path(model_name: str, mode: str) -> Path:
    """Return the checkpoint JSONL path for a model + mode."""
    safe_name = re.sub(r"[^\w\-]", "_", model_name)
    return _CHECKPOINT_DIR / f"{safe_name}_{mode}.jsonl"


def _save_checkpoint(result: CompletionResult, model_name: str, mode: str) -> None:
    """Append a single completion result to the checkpoint file."""
    path = _checkpoint_path(model_name, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")


def _load_checkpoint(model_name: str, mode: str) -> list[CompletionResult]:
    """Load previously checkpointed results."""
    path = _checkpoint_path(model_name, mode)
    results: list[CompletionResult] = []
    if not path.exists():
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(CompletionResult.model_validate_json(line))
    logger.info("Loaded %d checkpointed results from %s", len(results), path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """
    Orchestrates benchmark evaluation across all configured models.

    Args:
        config: The full pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.eval_cfg = config.eval
        self._providers: dict[str, BaseLLMProvider] = {}

    def _get_provider(self, model_config: ModelConfig) -> BaseLLMProvider:
        """Lazily create and cache a provider for a model."""
        if model_config.name not in self._providers:
            self._providers[model_config.name] = LLMClientFactory.create(model_config)
        return self._providers[model_config.name]

    # ── MCQ ──────────────────────────────────────────────────────────────

    async def run_mcq(
        self,
        problems: list[BenchmarkProblem],
        model_config: ModelConfig,
    ) -> list[CompletionResult]:
        """
        Run MCQ evaluation for a single model.

        Args:
            problems: List of benchmark problems.
            model_config: Model to evaluate.

        Returns:
            List of completion results with ``selected_option`` populated.
        """
        provider = self._get_provider(model_config)

        # Resume from checkpoint
        existing = _load_checkpoint(model_config.name, "mcq")
        done_ids = {r.problem_id for r in existing}
        remaining = [p for p in problems if p.problem_id not in done_ids]

        if not remaining:
            logger.info("MCQ already complete for %s (checkpoint)", model_config.name)
            return existing

        sem = asyncio.Semaphore(self.eval_cfg.concurrency)
        results: list[CompletionResult] = list(existing)
        checkpoint_counter = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"MCQ • {model_config.name}", total=len(remaining)
            )

            async def _run_one(problem: BenchmarkProblem) -> CompletionResult:
                nonlocal checkpoint_counter
                async with sem:
                    prompt = _format_mcq_prompt(problem)
                    result = await provider.complete(
                        prompt=prompt,
                        system_prompt=_MCQ_SYSTEM_PROMPT,
                        temperature=self.eval_cfg.temperature,
                        max_tokens=model_config.max_tokens,
                    )
                    result.problem_id = problem.problem_id
                    result.selected_option = _extract_option(result.raw_output)

                    checkpoint_counter += 1
                    if checkpoint_counter % self.eval_cfg.checkpoint_every == 0:
                        _save_checkpoint(result, model_config.name, "mcq")
                    else:
                        _save_checkpoint(result, model_config.name, "mcq")

                    progress.advance(task)
                    return result

            batch_results = await asyncio.gather(
                *[_run_one(p) for p in remaining], return_exceptions=True
            )

        for r in batch_results:
            if isinstance(r, BaseException):
                logger.error("MCQ error for %s: %s", model_config.name, r)
                continue
            results.append(r)

        return results

    # ── Open Generation ──────────────────────────────────────────────────

    async def run_open_gen(
        self,
        problems: list[BenchmarkProblem],
        model_config: ModelConfig,
    ) -> list[CompletionResult]:
        """
        Run open-generation evaluation for a single model.

        Args:
            problems: List of benchmark problems.
            model_config: Model to evaluate.

        Returns:
            List of completion results.
        """
        provider = self._get_provider(model_config)

        existing = _load_checkpoint(model_config.name, "opengen")
        done_ids = {r.problem_id for r in existing}
        remaining = [p for p in problems if p.problem_id not in done_ids]

        if not remaining:
            logger.info(
                "Open-gen already complete for %s (checkpoint)", model_config.name
            )
            return existing

        sem = asyncio.Semaphore(self.eval_cfg.concurrency)
        results: list[CompletionResult] = list(existing)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"OpenGen • {model_config.name}", total=len(remaining)
            )

            async def _run_one(problem: BenchmarkProblem) -> CompletionResult:
                async with sem:
                    prompt = _format_opengen_prompt(problem)
                    result = await provider.complete(
                        prompt=prompt,
                        system_prompt=_OPENGEN_SYSTEM_PROMPT,
                        temperature=self.eval_cfg.temperature,
                        max_tokens=model_config.max_tokens,
                    )
                    result.problem_id = problem.problem_id
                    _save_checkpoint(result, model_config.name, "opengen")
                    progress.advance(task)
                    return result

            batch_results = await asyncio.gather(
                *[_run_one(p) for p in remaining], return_exceptions=True
            )

        for r in batch_results:
            if isinstance(r, BaseException):
                logger.error("OpenGen error for %s: %s", model_config.name, r)
                continue
            results.append(r)

        return results

    # ── Full run ─────────────────────────────────────────────────────────

    async def run_all(
        self,
        problems: list[BenchmarkProblem],
        model_configs: list[ModelConfig] | None = None,
    ) -> dict[str, dict[str, list[CompletionResult]]]:
        """
        Run MCQ + open-gen for every configured model.

        Args:
            problems: Full list of benchmark problems.
            model_configs: Optional list of specific models to evaluate.

        Returns:
            Nested dict: ``{model_name: {"mcq": [...], "opengen": [...]}}``
        """
        all_results: dict[str, dict[str, list[CompletionResult]]] = {}
        all_models = model_configs if model_configs is not None else self.config.models.all_models

        logger.info(
            "Starting benchmark run: %d models × %d problems",
            len(all_models),
            len(problems),
        )

        for model_cfg in all_models:
            logger.info("═══ Running model: %s ═══", model_cfg.name)
            mcq = await self.run_mcq(problems, model_cfg)
            opengen = await self.run_open_gen(problems, model_cfg)
            all_results[model_cfg.name] = {"mcq": mcq, "opengen": opengen}
            logger.info(
                "Completed %s: %d MCQ, %d OpenGen",
                model_cfg.name,
                len(mcq),
                len(opengen),
            )

        return all_results

    # ── Dry run ──────────────────────────────────────────────────────────

    def dry_run(self, problems: list[BenchmarkProblem]) -> dict[str, Any]:
        """
        Estimate token counts and costs without making any API calls.

        Args:
            problems: Full list of benchmark problems.

        Returns:
            Dict with per-model estimates for tokens and cost.
        """
        estimates: dict[str, Any] = {}
        all_models = self.config.models.all_models

        for model_cfg in all_models:
            provider = self._get_provider(model_cfg)

            total_mcq_input = 0
            total_opengen_input = 0

            for problem in problems:
                mcq_prompt = _format_mcq_prompt(problem)
                og_prompt = _format_opengen_prompt(problem)
                total_mcq_input += provider.count_tokens(mcq_prompt)
                total_opengen_input += provider.count_tokens(og_prompt)

            # Estimate output tokens (heuristic: MCQ ≈ 50 tokens, OpenGen ≈ 300)
            est_mcq_output = len(problems) * 50
            est_opengen_output = len(problems) * 300

            mcq_cost = provider.estimate_cost(total_mcq_input, est_mcq_output)
            opengen_cost = provider.estimate_cost(total_opengen_input, est_opengen_output)

            estimates[model_cfg.name] = {
                "model_id": model_cfg.model_id,
                "provider": model_cfg.provider,
                "num_problems": len(problems),
                "mcq": {
                    "total_input_tokens": total_mcq_input,
                    "est_output_tokens": est_mcq_output,
                    "est_cost_usd": round(mcq_cost, 4),
                },
                "opengen": {
                    "total_input_tokens": total_opengen_input,
                    "est_output_tokens": est_opengen_output,
                    "est_cost_usd": round(opengen_cost, 4),
                },
                "total_est_cost_usd": round(mcq_cost + opengen_cost, 4),
            }

        grand_total = sum(e["total_est_cost_usd"] for e in estimates.values())
        return {
            "per_model": estimates,
            "grand_total_usd": round(grand_total, 4),
            "num_models": len(all_models),
            "num_problems": len(problems),
            "timestamp": datetime.utcnow().isoformat(),
        }
