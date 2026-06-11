"""
Formatting Clerk: LLM-based generation of final benchmark problems.

Takes the 1,000-row mutation matrix and produces formatted
:class:`BenchmarkProblem` objects containing:
- Natural language scenario narratives (prompt_context)
- MCQ options (1 correct + 3 distractors)
- Ground-truth rubrics for open-generation evaluation
- Randomized correct-answer positions

Distractor strategy:
- Option type A: "Resource abundance" — assumes the protagonist can buy/order
- Option type B: "Brute force" — physically possible but exhausting/impractical
- Option type D: "Wrong mechanism" — uses correct materials but wrong physics

Output files:
- data/benchmark/jugaad_reasoning_1k_mcq.json
- data/benchmark/jugaad_reasoning_1k_opengen.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Literal

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from jugaad_bench.models import (
    BenchmarkProblem,
    MCQOptions,
    MutatedInstance,
    ProblemMetadata,
    SynthesisRubric,
)
from jugaad_bench.utils.config import find_project_root, get_api_key, load_config
from jugaad_bench.utils.rate_limiter import rate_limited_call

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# LLM response schema for benchmark problem generation
# --------------------------------------------------------------------------


class _MCQDistractors(BaseModel):
    """Schema for the three distractor options."""

    abundance_distractor: str = Field(
        ...,
        min_length=10,
        description=(
            "Distractor assuming resource abundance: suggests buying, "
            "ordering online, or using unavailable expensive resources."
        ),
    )
    brute_force_distractor: str = Field(
        ...,
        min_length=10,
        description=(
            "Brute force distractor: a physically possible but exhausting "
            "or wildly impractical approach."
        ),
    )
    wrong_mechanism_distractor: str = Field(
        ...,
        min_length=10,
        description=(
            "Wrong mechanism distractor: uses the correct available materials "
            "but applies them with an incorrect physical principle."
        ),
    )


class _BenchmarkFormattingResponse(BaseModel):
    """Full LLM response for formatting a single benchmark problem."""

    prompt_context: str = Field(
        ...,
        min_length=50,
        description=(
            "Natural language scenario narrative (4-6 sentences). "
            "Describes the protagonist, location, constraints, and the "
            "specific problem they face. Must NOT reveal the solution."
        ),
    )
    correct_option: str = Field(
        ...,
        min_length=10,
        description=(
            "The correct MCQ answer: describes the actual innovation approach "
            "using the available materials and correct physics mechanism."
        ),
    )
    distractors: _MCQDistractors
    essential_keywords: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Keywords that MUST appear in a correct open-generation response "
            "(e.g., specific material names, physics terms)."
        ),
    )
    forbidden_keywords: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Keywords that must NOT appear (e.g., 'buy', 'Amazon', 'motor', "
            "'electricity'). Presence triggers AUTO_FAIL."
        ),
    )
    required_physical_mechanism: str = Field(
        ...,
        min_length=10,
        description=(
            "Description of the physics the model must explain to earn "
            "the physical_viability point in open-gen evaluation."
        ),
    )


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

_FORMATTING_SYSTEM_PROMPT = """\
You are the Formatting Clerk for the JugaadReasoning-1K benchmark.
Your job is to convert a structured mutation instance into a polished
multiple-choice benchmark problem.

Given a MutatedInstance with:
- Target goal, physics mechanism, domain
- Applied constraints (budget, environment, infrastructure)
- Adjusted inventory (available materials)
- Context narrative (character, location, situation)

You must generate:

1. **prompt_context** (4-6 sentences): A vivid, realistic scenario narrative.
   - Use the character name and location from the context_narrative.
   - Clearly state the constraints (budget, weather, infrastructure).
   - List the available materials naturally within the narrative.
   - Describe the problem clearly, but do NOT hint at the solution.

2. **correct_option**: The actual innovative solution using:
   - The available inventory items
   - The correct physics mechanism
   - Written as a clear action plan (2-3 sentences)

3. **distractors** (three wrong options):
   a. *abundance_distractor*: Assumes resources the protagonist doesn't have.
      Uses words like "order online", "buy from market", "use a motor/pump".
   b. *brute_force_distractor*: Physically possible but absurdly impractical.
      E.g., "manually carry 500 buckets" or "dig a 2-km canal by hand".
   c. *wrong_mechanism_distractor*: Uses the listed materials but applies
      them with incorrect physics (e.g., using a pipe as a lever when
      it should be used for siphoning).

4. **Rubric fields**:
   - essential_keywords: 3-6 keywords that MUST appear in a correct response.
   - forbidden_keywords: 3-6 keywords that signal constraint violation.
   - required_physical_mechanism: The physics principle the model must explain.

RULES:
- All four options must be grammatically parallel and similar in length.
- Distractors must be plausible enough that an uninformed reader might choose them.
- The correct option must NOT be obviously the longest or most detailed.
- forbidden_keywords should include terms implying constraint violation
  (e.g., "electricity" under Grid Outage, "buy" under ₹0 budget).
"""

_FORMATTING_USER_TEMPLATE = """\
Format this mutation instance into a benchmark problem:

INSTANCE:
- Instance ID: {instance_id}
- Domain: {domain}
- Target Goal: {target_goal}
- Core Physics Mechanism: {core_physics_mechanism}

CONSTRAINTS:
- Budget: {budget}
- Environment: {environment}
- Infrastructure: {infrastructure}

AVAILABLE INVENTORY:
{inventory_list}

CONTEXT:
{context_narrative}

ORIGINAL MATERIALS (for reference, NOT available):
{original_materials}

Generate the prompt_context, correct_option, three distractors, and rubric.
"""


class FormattingClerk:
    """Generates final benchmark problems from the mutation matrix.

    Args:
        config: Pipeline configuration.
        model: OpenAI model for formatting.
        temperature: Sampling temperature.
        max_retries_per_problem: Max retries for failed validations.
    """

    def __init__(
        self,
        config: Any | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_retries_per_problem: int = 3,
    ) -> None:
        self._config = config or load_config()
        self.model = model
        self.temperature = temperature
        self.max_retries_per_problem = max_retries_per_problem
        self._project_root = find_project_root()

        api_key = get_api_key("openai")
        self._raw_client = AsyncOpenAI(api_key=api_key)
        self._client = instructor.from_openai(self._raw_client)

    # ------------------------------------------------------------------
    # Single problem formatting
    # ------------------------------------------------------------------

    async def _format_single(
        self, instance: MutatedInstance
    ) -> BenchmarkProblem | None:
        """Format a single MutatedInstance into a BenchmarkProblem.

        Retries up to max_retries_per_problem times on validation failures.
        """
        # Parse seed/mutation numbers from instance_id (MI_001_01)
        parts = instance.instance_id.split("_")
        seed_num = int(parts[1])
        mut_num = int(parts[2])
        problem_id = f"JR-1K-{seed_num:03d}-{mut_num:02d}"

        # Deterministic RNG for answer position randomization
        pos_hash = int(
            hashlib.sha256(problem_id.encode()).hexdigest(), 16
        )
        rng = random.Random(pos_hash)

        inventory_list = "\n".join(
            f"  - {item}" for item in instance.adjusted_inventory
        )
        original_materials = ", ".join(instance.original_materials)

        user_message = _FORMATTING_USER_TEMPLATE.format(
            instance_id=instance.instance_id,
            domain=instance.domain.value,
            target_goal=instance.target_goal,
            core_physics_mechanism=instance.core_physics_mechanism,
            budget=instance.applied_constraints.budget,
            environment=instance.applied_constraints.environment,
            infrastructure=instance.applied_constraints.infrastructure,
            inventory_list=inventory_list,
            context_narrative=instance.context_narrative,
            original_materials=original_materials,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries_per_problem + 1):
            try:
                response: _BenchmarkFormattingResponse = await rate_limited_call(
                    "openai",
                    self._client.chat.completions.create,
                    model=self.model,
                    response_model=_BenchmarkFormattingResponse,
                    messages=[
                        {"role": "system", "content": _FORMATTING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                    max_retries=2,
                )

                # Assemble the four options and randomize correct answer position
                options_map = self._randomize_options(
                    correct=response.correct_option,
                    abundance=response.distractors.abundance_distractor,
                    brute_force=response.distractors.brute_force_distractor,
                    wrong_mechanism=response.distractors.wrong_mechanism_distractor,
                    rng=rng,
                )

                mcq_options = MCQOptions(
                    A=options_map["A"],
                    B=options_map["B"],
                    C=options_map["C"],
                    D=options_map["D"],
                )

                rubric = SynthesisRubric(
                    essential_keywords=response.essential_keywords,
                    forbidden_keywords=response.forbidden_keywords,
                    required_physical_mechanism=response.required_physical_mechanism,
                )

                metadata = ProblemMetadata(
                    seed_source=instance.seed_id,
                    physics_principle=instance.core_physics_mechanism,
                )

                problem = BenchmarkProblem(
                    problem_id=problem_id,
                    domain=instance.domain,
                    metadata=metadata,
                    prompt_context=response.prompt_context,
                    applied_constraints=instance.applied_constraints,
                    available_inventory=instance.adjusted_inventory,
                    mcq_options=mcq_options,
                    ground_truth_option=options_map["correct_letter"],
                    ground_truth_synthesis_rubric=rubric,
                )

                return problem

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    self.max_retries_per_problem,
                    problem_id,
                    exc,
                )

        logger.error(
            "All %d attempts failed for %s: %s",
            self.max_retries_per_problem,
            problem_id,
            last_error,
        )
        return None

    @staticmethod
    def _randomize_options(
        correct: str,
        abundance: str,
        brute_force: str,
        wrong_mechanism: str,
        rng: random.Random,
    ) -> dict[str, str]:
        """Randomly assign option letters to the four choices.

        Returns a dict with keys 'A', 'B', 'C', 'D', and 'correct_letter'.
        """
        options = [
            ("correct", correct),
            ("abundance", abundance),
            ("brute_force", brute_force),
            ("wrong_mechanism", wrong_mechanism),
        ]
        rng.shuffle(options)

        letters = ["A", "B", "C", "D"]
        result: dict[str, str] = {}
        correct_letter = "A"

        for letter, (option_type, text) in zip(letters, options):
            result[letter] = text
            if option_type == "correct":
                correct_letter = letter

        result["correct_letter"] = correct_letter
        return result

    # ------------------------------------------------------------------
    # Full batch formatting
    # ------------------------------------------------------------------

    async def format_all(
        self,
        instances: list[MutatedInstance],
    ) -> list[BenchmarkProblem]:
        """Format all mutation instances into benchmark problems.

        Args:
            instances: The 1,000-row mutation matrix.

        Returns:
            List of validated BenchmarkProblem objects.
        """
        logger.info(
            "Formatting %d mutation instances into benchmark problems",
            len(instances),
        )

        problems: list[BenchmarkProblem] = []
        failed_count = 0

        for idx, instance in enumerate(instances):
            logger.info(
                "Formatting %d/%d: %s",
                idx + 1,
                len(instances),
                instance.instance_id,
            )

            problem = await self._format_single(instance)
            if problem is not None:
                problems.append(problem)
            else:
                failed_count += 1

        logger.info(
            "Formatting complete: %d problems (%d failed)",
            len(problems),
            failed_count,
        )

        return problems

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_benchmark(
        self,
        problems: list[BenchmarkProblem],
        mcq_path: Path | None = None,
        opengen_path: Path | None = None,
    ) -> tuple[Path, Path]:
        """Save the benchmark in both MCQ and Open-Gen formats.

        Args:
            problems: List of BenchmarkProblem objects.
            mcq_path: Override path for MCQ output.
            opengen_path: Override path for Open-Gen output.

        Returns:
            Tuple of (mcq_file_path, opengen_file_path).
        """
        benchmark_dir = self._project_root / "data" / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if mcq_path is None:
            mcq_path = benchmark_dir / "jugaad_reasoning_1k_mcq.json"
        if opengen_path is None:
            opengen_path = benchmark_dir / "jugaad_reasoning_1k_opengen.json"

        # MCQ format: full problem with options
        mcq_data = [p.model_dump(mode="json") for p in problems]
        mcq_path.write_text(
            json.dumps(mcq_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved MCQ benchmark (%d problems) to %s", len(problems), mcq_path)

        # Open-Gen format: problem without MCQ options (rubric-based evaluation)
        opengen_data = []
        for problem in problems:
            entry = problem.model_dump(mode="json")
            # Remove MCQ-specific fields for open-gen format
            entry.pop("mcq_options", None)
            entry.pop("ground_truth_option", None)
            opengen_data.append(entry)

        opengen_path.write_text(
            json.dumps(opengen_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved Open-Gen benchmark (%d problems) to %s",
            len(problems),
            opengen_path,
        )

        return mcq_path, opengen_path

    def load_benchmark(
        self,
        mcq_path: Path | None = None,
    ) -> list[BenchmarkProblem]:
        """Load a previously saved MCQ benchmark from JSON.

        Args:
            mcq_path: Path to the MCQ JSON file.

        Returns:
            List of validated BenchmarkProblem objects.
        """
        if mcq_path is None:
            mcq_path = (
                self._project_root
                / "data"
                / "benchmark"
                / "jugaad_reasoning_1k_mcq.json"
            )

        if not mcq_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {mcq_path}")

        raw_data: list[dict[str, Any]] = json.loads(
            mcq_path.read_text(encoding="utf-8")
        )

        problems: list[BenchmarkProblem] = []
        for entry in raw_data:
            problems.append(BenchmarkProblem.model_validate(entry))

        logger.info("Loaded %d benchmark problems from %s", len(problems), mcq_path)
        return problems
