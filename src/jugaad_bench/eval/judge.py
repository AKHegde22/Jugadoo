"""
LLM-as-a-Judge for JugaadReasoning-1K open-generation evaluation.

Uses ``instructor`` for structured output (``JudgeScore`` Pydantic model),
logs every judgement to a JSONL audit trail, and supports batch evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import instructor
from openai import AsyncOpenAI

from jugaad_bench.models import BenchmarkProblem, JudgeScore
from jugaad_bench.utils.config import get_api_key

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (PRD Section 6.C)
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
You are an expert mechanical evaluator and physical constraint clerk. \
Your job is to grade an LLM's solution to a scarcity problem against a strict Ground Truth Rubric.

You MUST evaluate each of the following three dimensions INDEPENDENTLY and assign a binary score (0 or 1) for each:

1. **Constraint Adherence** (0 or 1):
   Award 1 if the model's solution does NOT use any materials, tools, or resources that are NOT in the provided inventory list, and does NOT exceed the stated budget. Award 0 if the model invokes unlisted materials, suggests purchasing items, or violates any stated constraint.

2. **Inventory Utilization** (0 or 1):
   Award 1 if the model correctly and meaningfully uses the specific items listed in the available inventory. Award 0 if the model ignores the inventory, uses items in a physically impossible way, or only superficially mentions them.

3. **Physical Viability** (0 or 1):
   Award 1 if the proposed mechanism is consistent with the required physical principle described in the rubric. The model does not need to name the principle explicitly, but its described mechanism must be physically plausible and align with the rubric's required_physical_mechanism. Award 0 if the proposed mechanism is physically impossible, vague hand-waving, or contradicts the required physics.

**Important Rules:**
- Think step by step BEFORE assigning scores. Write your reasoning first.
- Be strict: partial credit is NOT allowed. Each dimension is 0 or 1.
- The total_score MUST equal the sum of the three binary scores.
- Do NOT award points for effort, creativity, or good intentions.
- If the solution says "I cannot help" or refuses to answer, score all three as 0.\
"""


def _build_user_prompt(problem: BenchmarkProblem, model_output: str) -> str:
    """Construct the user-facing evaluation prompt sent to the judge."""
    rubric = problem.ground_truth_synthesis_rubric
    constraints = problem.applied_constraints
    inventory = ", ".join(problem.available_inventory)

    return (
        "## Problem Context\n"
        f"{problem.prompt_context}\n\n"
        "## Constraints\n"
        f"- Budget: {constraints.budget}\n"
        f"- Environment: {constraints.environment}\n"
        f"- Infrastructure: {constraints.infrastructure}\n\n"
        "## Available Inventory\n"
        f"{inventory}\n\n"
        "## Ground Truth Rubric\n"
        f"- Essential keywords: {', '.join(rubric.essential_keywords)}\n"
        f"- Forbidden keywords: {', '.join(rubric.forbidden_keywords)}\n"
        f"- Required physical mechanism: {rubric.required_physical_mechanism}\n\n"
        "## Model Output to Evaluate\n"
        f"{model_output}\n\n"
        "Now evaluate the model output against the rubric. "
        "Provide your reasoning, then score each dimension."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Judge class
# ─────────────────────────────────────────────────────────────────────────────


class JugaadJudge:
    """
    LLM-as-a-Judge backed by ``instructor`` for guaranteed structured output.

    Args:
        provider: Provider name (currently only ``"openai"`` is supported
            via ``instructor.from_openai``).
        model: Model identifier (e.g. ``"gpt-4o"``).
        temperature: Sampling temperature for the judge.
        max_tokens: Max tokens for the judge response.
        audit_log_path: Path to the JSONL audit log.  Defaults to
            ``eval_outputs/judge_audit.jsonl``.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 512,
        audit_log_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Build instructor-patched async client
        api_key = get_api_key(provider)
        self._client = instructor.from_openai(AsyncOpenAI(api_key=api_key))

        # Audit log
        if audit_log_path is None:
            audit_log_path = Path("eval_outputs") / "judge_audit.jsonl"
        self._audit_path = audit_log_path
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)

    # ── single evaluation ───────────────────────────────────────────────

    async def evaluate(
        self,
        problem: BenchmarkProblem,
        model_output: str,
    ) -> JudgeScore:
        """
        Evaluate a single open-generation response.

        Args:
            problem: The benchmark problem (contains rubric + constraints).
            model_output: The raw LLM response to evaluate.

        Returns:
            A validated ``JudgeScore``.
        """
        user_prompt = _build_user_prompt(problem, model_output)

        score: JudgeScore = await self._client.chat.completions.create(
            model=self.model,
            response_model=JudgeScore,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        self._write_audit_entry(problem.problem_id, model_output, user_prompt, score)
        return score

    # ── batch evaluation ────────────────────────────────────────────────

    async def batch_evaluate(
        self,
        problems: list[tuple[BenchmarkProblem, str]],
        concurrency: int = 5,
    ) -> list[JudgeScore]:
        """
        Evaluate multiple (problem, model_output) pairs concurrently.

        Args:
            problems: List of ``(BenchmarkProblem, model_output)`` tuples.
            concurrency: Maximum concurrent judge calls.

        Returns:
            List of ``JudgeScore`` in the same order as *problems*.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _eval_one(
            idx: int, problem: BenchmarkProblem, output: str
        ) -> tuple[int, JudgeScore]:
            async with sem:
                score = await self.evaluate(problem, output)
                return idx, score

        tasks = [
            asyncio.create_task(_eval_one(i, p, o))
            for i, (p, o) in enumerate(problems)
        ]

        results_unordered = await asyncio.gather(*tasks, return_exceptions=True)

        # Re-order and propagate exceptions
        scores: list[JudgeScore | None] = [None] * len(problems)
        for result in results_unordered:
            if isinstance(result, BaseException):
                raise result
            idx, score = result
            scores[idx] = score

        return [s for s in scores if s is not None]

    # ── audit logging ───────────────────────────────────────────────────

    def _write_audit_entry(
        self,
        problem_id: str,
        model_output: str,
        user_prompt: str,
        score: JudgeScore,
    ) -> None:
        """Atomically append one audit record to the JSONL log."""
        entry: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "judge_model": self.model,
            "problem_id": problem_id,
            "model_output_excerpt": model_output[:500],
            "user_prompt_excerpt": user_prompt[:500],
            "score": score.model_dump(),
        }

        # Write to temp file first, then append for atomicity
        try:
            dir_path = self._audit_path.parent
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=dir_path,
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(json.dumps(entry, default=str) + "\n")
                tmp_path = Path(tmp.name)

            # Append temp content to the main audit file
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(tmp_path.read_text(encoding="utf-8"))
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to write audit entry for %s", problem_id)
