"""
Combinatorial Mutation Engine for JugaadReasoning-1K.

Generates the 1,000-row mutation matrix by:
1. Loading the 27 possible constraint combinations from pipeline_config.yaml
2. For each of 100 seeds, selecting 10 unique constraint profiles via
   stratified sampling (guaranteeing coverage of all tiers)
3. Using an LLM (OpenAI + instructor) to adjust each seed's inventory for
   the applied constraints, producing MutatedInstance objects
4. Validating the final matrix has exactly 1,000 rows

Sampling guarantees per seed (10 slots):
- ≥1 from each financial tier  (3 tiers → 3 guaranteed)
- ≥1 from each environmental condition  (3 conditions → 3 guaranteed)
- ≥1 from each infrastructure constraint  (3 constraints → 3 guaranteed)
- Remaining 1 slot selected randomly from unused combinations
- Uses deterministic RNG seeded from the seed_id hash
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
from pathlib import Path
from typing import Any

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from jugaad_bench.models import (
    ConstraintProfile,
    Domain,
    MutatedInstance,
    SeedTuple,
)
from jugaad_bench.utils.config import (
    PipelineConfig,
    find_project_root,
    get_api_key,
    load_config,
)
from jugaad_bench.utils.rate_limiter import rate_limited_call

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# LLM response schema for inventory adjustment
# --------------------------------------------------------------------------


class _InventoryAdjustmentResponse(BaseModel):
    """Schema for the LLM's inventory adjustment output."""

    adjusted_inventory: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Adjusted list of materials available under the given constraints. "
            "Must still enable the core physics mechanism."
        ),
    )
    context_narrative: str = Field(
        ...,
        min_length=20,
        description=(
            "A brief narrative (2-3 sentences) describing the scenario: "
            "character name, specific location in India, and the immediate "
            "situation they face."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of what was added/removed and why.",
    )


# --------------------------------------------------------------------------
# System prompt for inventory adjustment
# --------------------------------------------------------------------------

_MUTATION_SYSTEM_PROMPT = """\
You are a materials engineering and constraint-reasoning expert for the
JugaadReasoning-1K benchmark. Your job is to adjust an innovation's material
inventory to fit specific real-world constraints.

Given:
- A seed innovation with its target goal, physics mechanism, and original materials
- A constraint profile (budget, environment, infrastructure)

You must:
1. KEEP materials essential to the core physics mechanism.
2. REMOVE materials that violate the constraints:
   - ₹0 budget → Remove anything that costs money; only scrounged/free items.
   - ₹50 budget → Allow very cheap items (twine, tape, basic nails).
   - ₹200 budget → Allow moderately priced items (basic tools, materials).
   - 45°C Heatwave → Remove items that melt, degrade, or are hazardous in heat.
   - Monsoon Flash Flood → Remove paper, cardboard, water-soluble materials.
   - Dust Storm/Arid Air → Remove electronics, precision instruments.
   - Total Grid Outage → Remove ANY electrically powered items.
   - No Internet/Cell Signal → Remove smart devices, IoT components.
   - No Motorized Transport → Remove heavy machinery, items needing transport.
3. ADD 2-5 contextually appropriate scavenged or low-cost substitutes that:
   - Are realistically available in rural/peri-urban India
   - Fit the budget tier
   - Survive the environmental condition
   - Work without the missing infrastructure
4. Generate a context_narrative: Create a realistic character (Indian name),
   a specific location (village/town in India), and their immediate situation.

CRITICAL: The adjusted inventory MUST still enable the core physics mechanism.
"""

_MUTATION_USER_TEMPLATE = """\
Adjust the inventory for this innovation under the given constraints.

SEED INNOVATION:
- Target Goal: {target_goal}
- Core Physics Mechanism: {core_physics_mechanism}
- Domain: {domain}
- Original Materials: {original_materials}

APPLIED CONSTRAINTS:
- Budget: {budget}
- Environment: {environment}
- Infrastructure: {infrastructure}

Generate:
1. adjusted_inventory: The modified materials list
2. context_narrative: A 2-3 sentence scenario with character name, location, situation
3. reasoning: Brief explanation of your material changes
"""


class MutationEngine:
    """Generates the 1,000-row mutation matrix from 100 seeds × 10 mutations.

    Args:
        config: Pipeline configuration (loaded from YAML).
        model: OpenAI model to use for inventory adjustment.
        temperature: Sampling temperature for the LLM.
        max_retries: Max retries per LLM call.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        self._config = config or load_config()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._project_root = find_project_root()

        api_key = get_api_key("openai")
        self._raw_client = AsyncOpenAI(api_key=api_key)
        self._client = instructor.from_openai(self._raw_client)

        # Build the 27-combination constraint space
        self._all_combinations = self._build_constraint_space()
        logger.info(
            "Constraint space: %d combinations", len(self._all_combinations)
        )

    # ------------------------------------------------------------------
    # Constraint space construction
    # ------------------------------------------------------------------

    def _build_constraint_space(self) -> list[ConstraintProfile]:
        """Generate all 27 constraint combinations from config.

        Uses itertools.product over the 3 financial × 3 environmental
        × 3 infrastructural constraint levels.
        """
        financial = self._config.constraints.financial
        environmental = self._config.constraints.environmental
        infrastructural = self._config.constraints.infrastructural

        combinations: list[ConstraintProfile] = []
        for fin, env, infra in itertools.product(
            financial, environmental, infrastructural
        ):
            combinations.append(
                ConstraintProfile(
                    budget=fin,
                    environment=env,
                    infrastructure=infra,
                )
            )

        assert len(combinations) == 27, (
            f"Expected 27 constraint combinations, got {len(combinations)}"
        )
        return combinations

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------

    def _select_constraint_profiles(
        self, seed_id: str, count: int = 10
    ) -> list[ConstraintProfile]:
        """Select constraint profiles for a seed using stratified sampling.

        Guarantees:
        - ≥1 from each of the 3 financial tiers
        - ≥1 from each of the 3 environmental conditions
        - ≥1 from each of the 3 infrastructure constraints
        - Remaining slot(s) filled randomly from unused combinations

        Uses a deterministic RNG seeded from the seed_id hash.
        """
        # Create deterministic RNG from seed_id
        seed_hash = int(hashlib.sha256(seed_id.encode()).hexdigest(), 16)
        rng = random.Random(seed_hash)

        financial_tiers = self._config.constraints.financial
        environmental_conditions = self._config.constraints.environmental
        infrastructure_constraints = self._config.constraints.infrastructural

        selected: list[ConstraintProfile] = []
        used_indices: set[int] = set()

        # Helper: find combinations matching a criterion and pick one
        def _pick_one_matching(
            field: str, value: str
        ) -> ConstraintProfile | None:
            matching = [
                (i, c)
                for i, c in enumerate(self._all_combinations)
                if i not in used_indices and getattr(c, field) == value
            ]
            if not matching:
                return None
            idx, combo = rng.choice(matching)
            used_indices.add(idx)
            return combo

        # Guarantee ≥1 from each financial tier
        for tier in financial_tiers:
            combo = _pick_one_matching("budget", tier)
            if combo:
                selected.append(combo)

        # Guarantee ≥1 from each environmental condition
        for condition in environmental_conditions:
            combo = _pick_one_matching("environment", condition)
            if combo:
                selected.append(combo)

        # Guarantee ≥1 from each infrastructure constraint
        for constraint in infrastructure_constraints:
            combo = _pick_one_matching("infrastructure", constraint)
            if combo:
                selected.append(combo)

        # Fill remaining slots randomly
        remaining_needed = count - len(selected)
        if remaining_needed > 0:
            available = [
                (i, c)
                for i, c in enumerate(self._all_combinations)
                if i not in used_indices
            ]
            if available:
                chosen = rng.sample(available, min(remaining_needed, len(available)))
                for idx, combo in chosen:
                    selected.append(combo)
                    used_indices.add(idx)

        # Shuffle to avoid predictable ordering
        rng.shuffle(selected)

        return selected[:count]

    # ------------------------------------------------------------------
    # Inventory mutation via LLM
    # ------------------------------------------------------------------

    async def _mutate_single(
        self,
        seed: SeedTuple,
        constraint: ConstraintProfile,
        mutation_num: int,
    ) -> MutatedInstance:
        """Generate a single MutatedInstance via LLM inventory adjustment.

        Args:
            seed: The parent seed tuple.
            constraint: The constraint profile to apply.
            mutation_num: The mutation number (1-based) within this seed.

        Returns:
            Validated MutatedInstance.
        """
        seed_num = int(seed.seed_id.split("_")[1])
        instance_id = f"MI_{seed_num:03d}_{mutation_num:02d}"

        user_message = _MUTATION_USER_TEMPLATE.format(
            target_goal=seed.target_goal,
            core_physics_mechanism=seed.core_physics_mechanism,
            domain=seed.domain.value,
            original_materials=", ".join(seed.historical_materials_used),
            budget=constraint.budget,
            environment=constraint.environment,
            infrastructure=constraint.infrastructure,
        )

        response: _InventoryAdjustmentResponse = await rate_limited_call(
            "openai",
            self._client.chat.completions.create,
            model=self.model,
            response_model=_InventoryAdjustmentResponse,
            messages=[
                {"role": "system", "content": _MUTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

        return MutatedInstance(
            instance_id=instance_id,
            seed_id=seed.seed_id,
            domain=seed.domain,
            target_goal=seed.target_goal,
            core_physics_mechanism=seed.core_physics_mechanism,
            applied_constraints=constraint,
            adjusted_inventory=response.adjusted_inventory,
            context_narrative=response.context_narrative,
            original_materials=seed.historical_materials_used,
        )

    # ------------------------------------------------------------------
    # Full matrix generation
    # ------------------------------------------------------------------

    async def generate_matrix(
        self,
        seeds: list[SeedTuple],
        mutations_per_seed: int = 10,
    ) -> list[MutatedInstance]:
        """Generate the complete 1,000-row mutation matrix.

        Args:
            seeds: List of 100 SeedTuples.
            mutations_per_seed: Number of mutations per seed (default 10).

        Returns:
            List of 1,000 MutatedInstance objects.

        Raises:
            ValueError: If the final matrix doesn't have the expected size.
        """
        expected_total = len(seeds) * mutations_per_seed
        logger.info(
            "Generating mutation matrix: %d seeds × %d mutations = %d instances",
            len(seeds),
            mutations_per_seed,
            expected_total,
        )

        all_instances: list[MutatedInstance] = []
        failed_count = 0

        for seed_idx, seed in enumerate(seeds):
            logger.info(
                "Processing seed %d/%d: %s (%s)",
                seed_idx + 1,
                len(seeds),
                seed.seed_id,
                seed.target_goal[:50],
            )

            # Select constraint profiles for this seed
            profiles = self._select_constraint_profiles(
                seed.seed_id, count=mutations_per_seed
            )

            for mut_idx, profile in enumerate(profiles, start=1):
                try:
                    instance = await self._mutate_single(seed, profile, mut_idx)
                    all_instances.append(instance)
                    logger.debug(
                        "  %s: %s / %s / %s → %d materials",
                        instance.instance_id,
                        profile.budget,
                        profile.environment,
                        profile.infrastructure,
                        len(instance.adjusted_inventory),
                    )
                except Exception:
                    logger.exception(
                        "  Failed mutation %d for %s", mut_idx, seed.seed_id
                    )
                    failed_count += 1

        logger.info(
            "Matrix generation complete: %d instances (%d failed)",
            len(all_instances),
            failed_count,
        )

        # Validate expected size
        if len(all_instances) != expected_total:
            logger.warning(
                "Matrix size mismatch: expected %d, got %d",
                expected_total,
                len(all_instances),
            )

        return all_instances

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_matrix(
        self,
        instances: list[MutatedInstance],
        output_path: Path | None = None,
    ) -> Path:
        """Save the mutation matrix to JSON.

        Args:
            instances: List of MutatedInstance objects.
            output_path: Override output file path.

        Returns:
            Path to the written JSON file.
        """
        if output_path is None:
            output_path = (
                self._project_root
                / "data"
                / "mutations"
                / "mutation_matrix_1000.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [inst.model_dump(mode="json") for inst in instances]
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved mutation matrix (%d instances) to %s",
            len(instances),
            output_path,
        )
        return output_path

    def load_matrix(
        self,
        input_path: Path | None = None,
    ) -> list[MutatedInstance]:
        """Load a previously saved mutation matrix from JSON.

        Args:
            input_path: Path to the matrix JSON file.

        Returns:
            List of validated MutatedInstance objects.
        """
        if input_path is None:
            input_path = (
                self._project_root
                / "data"
                / "mutations"
                / "mutation_matrix_1000.json"
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Mutation matrix not found: {input_path}")

        raw_data: list[dict[str, Any]] = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        instances: list[MutatedInstance] = []
        for entry in raw_data:
            instances.append(MutatedInstance.model_validate(entry))

        logger.info("Loaded %d instances from %s", len(instances), input_path)
        return instances
