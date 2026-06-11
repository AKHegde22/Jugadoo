"""
LLM-assisted extraction of SeedTuples from RawCases.

Uses OpenAI SDK with the ``instructor`` library to enforce structured output
matching the :class:`SeedTuple` Pydantic schema.  Each RawCase is sent to
GPT-4o with a system prompt that instructs the model to extract structured
innovation data (target goal, physics mechanism, materials, etc.).

Key features:
- Structured output via instructor + Pydantic validation
- Domain assignment with distribution awareness
- Sequential seed_id generation (SEED_001 … SEED_100)
- Deduplication by target_goal cosine similarity (basic Jaccard-based)
- Export to JSON for human review
- ``review_and_finalize()`` to load human-edited JSON back into validated models
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import instructor
from openai import AsyncOpenAI

from jugaad_bench.models import Domain, RawCase, SeedTuple
from jugaad_bench.utils.config import find_project_root, get_api_key
from jugaad_bench.utils.rate_limiter import rate_limited_call

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# System prompt for seed extraction
# --------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert analyst for the JugaadReasoning-1K benchmark project.
Your task is to extract structured innovation data from raw text describing
a real-world grassroots innovation from India.

You must produce output conforming EXACTLY to the SeedTuple schema:
- seed_id: Will be assigned later — use "SEED_000" as a placeholder.
- domain: One of: agriculture, healthcare, construction, street_vending.
  Choose the BEST-FIT domain based on the innovation's primary application.
- target_goal: A clear description of the PHYSICAL PROBLEM the innovation
  solves (NOT the solution itself). Must be ≥20 characters. Example:
  "Irrigate a 2-acre plot without electric pumps or canal access."
- core_physics_mechanism: The underlying physical principle making the
  innovation work. Must be ≥10 characters. Example: "Gravity-fed fluid
  dynamics regulated via mechanical constriction."
- historical_materials_used: The ACTUAL materials used in the documented
  solution. Each entry must be a specific, tangible item (≥3 characters).
  List at least 3 materials.
- source_reference: Cite the original source. Use the information provided.

RULES:
1. If the text doesn't describe a physical innovation with identifiable
   materials and mechanism, respond with an empty target_goal to signal
   "skip this entry."
2. The target_goal must describe a PROBLEM, not a solution.
3. The physics mechanism must be a scientifically accurate description.
4. Materials must be specific physical objects, not categories.
5. Domain assignment must reflect the PRIMARY use case of the innovation.
"""

_EXTRACTION_USER_TEMPLATE = """\
Extract the structured innovation data from the following raw text.
If this text does not describe a valid physical innovation with identifiable
materials and mechanism, set target_goal to an empty string.

Source: {source}
Path/URL: {url_or_path}

--- RAW TEXT ---
{raw_text}
--- END RAW TEXT ---

Additional context (if available):
- Title: {title}
- Innovator: {innovator_name}
- Location: {location}
- Category: {category}
"""

# Threshold for target_goal similarity (Jaccard) to consider duplicates
_SIMILARITY_THRESHOLD = 0.65


class _SeedExtractionResponse(SeedTuple):
    """Relaxed SeedTuple for LLM extraction (allows placeholder seed_id)."""

    class Config:
        """Allow placeholder seed_id during extraction."""

    seed_id: str = "SEED_000"


class SeedExtractor:
    """Extracts structured SeedTuples from raw innovation cases using GPT-4o.

    Args:
        model: OpenAI model identifier.
        temperature: Sampling temperature for extraction.
        max_retries: Maximum retries per extraction call.
        similarity_threshold: Jaccard similarity threshold for deduplication.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_retries: int = 3,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.similarity_threshold = similarity_threshold
        self._project_root = find_project_root()

        api_key = get_api_key("openai")
        self._raw_client = AsyncOpenAI(api_key=api_key)
        self._client = instructor.from_openai(self._raw_client)

    # ------------------------------------------------------------------
    # Main extraction workflow
    # ------------------------------------------------------------------

    async def extract_seeds(
        self,
        raw_cases: list[RawCase],
        target_count: int = 100,
        domain_distribution: dict[str, int] | None = None,
    ) -> list[SeedTuple]:
        """Extract SeedTuples from a list of RawCases.

        Args:
            raw_cases: Raw innovation cases from scrapers.
            target_count: Target number of seeds (default 100).
            domain_distribution: Optional target distribution per domain
                (e.g. {"agriculture": 30, "healthcare": 20, ...}).

        Returns:
            List of validated, deduplicated SeedTuple objects with
            sequential seed_ids.
        """
        logger.info(
            "Starting seed extraction from %d raw cases (target: %d)",
            len(raw_cases),
            target_count,
        )

        # Phase 1: Extract candidates from all raw cases
        candidates: list[SeedTuple] = []
        for idx, raw_case in enumerate(raw_cases):
            logger.info(
                "Extracting %d/%d: %s",
                idx + 1,
                len(raw_cases),
                raw_case.title or raw_case.source,
            )
            try:
                seed = await self._extract_single(raw_case)
                if seed is not None:
                    candidates.append(seed)
                    logger.info("  → Extracted: %s", seed.target_goal[:60])
                else:
                    logger.debug("  → Skipped (no valid innovation)")
            except Exception:
                logger.exception(
                    "  → Failed extraction for case from %s", raw_case.source
                )

        logger.info("Phase 1 complete: %d candidates from %d raw cases", len(candidates), len(raw_cases))

        # Phase 2: Deduplicate by target_goal similarity
        deduplicated = self._deduplicate(candidates)
        logger.info("After deduplication: %d candidates", len(deduplicated))

        # Phase 3: Select final seeds with domain distribution
        selected = self._select_with_distribution(
            deduplicated, target_count, domain_distribution
        )
        logger.info("Selected %d seeds with domain distribution", len(selected))

        # Phase 4: Assign sequential seed_ids
        final = self._assign_seed_ids(selected)
        logger.info("Assigned seed IDs: SEED_001 through SEED_%03d", len(final))

        return final

    # ------------------------------------------------------------------
    # Single case extraction
    # ------------------------------------------------------------------

    async def _extract_single(self, raw_case: RawCase) -> SeedTuple | None:
        """Extract a SeedTuple from a single RawCase using GPT-4o.

        Returns None if the raw case doesn't describe a valid innovation.
        """
        user_message = _EXTRACTION_USER_TEMPLATE.format(
            source=raw_case.source,
            url_or_path=raw_case.url_or_path,
            raw_text=raw_case.raw_text[:4000],  # Limit input to ~4k chars
            title=raw_case.title or "N/A",
            innovator_name=raw_case.innovator_name or "N/A",
            location=raw_case.location or "N/A",
            category=raw_case.category or "N/A",
        )

        try:
            response: _SeedExtractionResponse = await rate_limited_call(
                "openai",
                self._client.chat.completions.create,
                model=self.model,
                response_model=_SeedExtractionResponse,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_retries=self.max_retries,
            )
        except Exception:
            logger.warning("LLM extraction failed for: %s", raw_case.title or raw_case.source)
            return None

        # Check if the LLM signaled "skip" via empty target_goal
        if not response.target_goal or len(response.target_goal.strip()) < 20:
            return None

        # Validate domain
        try:
            domain = Domain(response.domain)
        except ValueError:
            domain = Domain.AGRICULTURE  # Default fallback

        # Build validated SeedTuple (placeholder seed_id for now)
        try:
            seed = SeedTuple(
                seed_id="SEED_000",
                domain=domain,
                target_goal=response.target_goal,
                core_physics_mechanism=response.core_physics_mechanism,
                historical_materials_used=response.historical_materials_used,
                source_reference=response.source_reference or f"{raw_case.source}: {raw_case.url_or_path}",
            )
            return seed
        except Exception as exc:
            logger.warning("Validation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, candidates: list[SeedTuple]) -> list[SeedTuple]:
        """Remove near-duplicate SeedTuples based on target_goal similarity.

        Uses a simple token-level Jaccard coefficient. Two seeds are
        considered duplicates if their Jaccard similarity exceeds the
        configured threshold.
        """
        if not candidates:
            return []

        unique: list[SeedTuple] = [candidates[0]]

        for candidate in candidates[1:]:
            is_duplicate = False
            candidate_tokens = self._tokenize(candidate.target_goal)

            for existing in unique:
                existing_tokens = self._tokenize(existing.target_goal)
                similarity = self._jaccard_similarity(candidate_tokens, existing_tokens)

                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        "Duplicate detected (sim=%.2f):\n  '%s'\n  ≈ '%s'",
                        similarity,
                        candidate.target_goal[:60],
                        existing.target_goal[:60],
                    )
                    break

            # Also check with SequenceMatcher for substring-level similarity
            if not is_duplicate:
                for existing in unique:
                    seq_ratio = SequenceMatcher(
                        None,
                        candidate.target_goal.lower(),
                        existing.target_goal.lower(),
                    ).ratio()
                    if seq_ratio >= self.similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(candidate)

        return unique

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of lowercase words."""
        return {
            word.lower().strip(".,;:!?\"'()[]{}") for word in text.split() if len(word) > 2
        }

    @staticmethod
    def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """Calculate Jaccard similarity between two token sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Domain-balanced selection
    # ------------------------------------------------------------------

    def _select_with_distribution(
        self,
        candidates: list[SeedTuple],
        target_count: int,
        distribution: dict[str, int] | None,
    ) -> list[SeedTuple]:
        """Select candidates to match a target domain distribution.

        If distribution is not specified or there aren't enough candidates
        per domain, fills proportionally.
        """
        if not distribution:
            return candidates[:target_count]

        # Group by domain
        by_domain: dict[str, list[SeedTuple]] = {d.value: [] for d in Domain}
        for seed in candidates:
            by_domain[seed.domain.value].append(seed)

        selected: list[SeedTuple] = []

        # First pass: fill each domain up to its target
        for domain_str, target in distribution.items():
            available = by_domain.get(domain_str, [])
            take = min(target, len(available))
            selected.extend(available[:take])
            # Remove selected from pool
            by_domain[domain_str] = available[take:]

        # Second pass: fill remaining slots from any domain
        remaining_needed = target_count - len(selected)
        if remaining_needed > 0:
            all_remaining = []
            for domain_list in by_domain.values():
                all_remaining.extend(domain_list)
            selected.extend(all_remaining[:remaining_needed])

        return selected[:target_count]

    # ------------------------------------------------------------------
    # Seed ID assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_seed_ids(seeds: list[SeedTuple]) -> list[SeedTuple]:
        """Assign sequential seed_ids (SEED_001, SEED_002, ...).

        Returns new SeedTuple instances with updated seed_ids.
        """
        result: list[SeedTuple] = []
        for idx, seed in enumerate(seeds, start=1):
            updated = seed.model_copy(update={"seed_id": f"SEED_{idx:03d}"})
            result.append(updated)
        return result

    # ------------------------------------------------------------------
    # Export and review
    # ------------------------------------------------------------------

    def export_candidates(
        self,
        seeds: list[SeedTuple],
        output_path: Path | None = None,
    ) -> Path:
        """Export seed candidates to JSON for human review.

        Args:
            seeds: List of SeedTuple objects to export.
            output_path: Override output file path. Defaults to
                data/seeds/seed_candidates.json.

        Returns:
            Path to the written JSON file.
        """
        if output_path is None:
            output_path = self._project_root / "data" / "seeds" / "seed_candidates.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [seed.model_dump(mode="json") for seed in seeds]
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Exported %d seed candidates to %s", len(seeds), output_path)
        return output_path

    def review_and_finalize(
        self,
        reviewed_path: Path | None = None,
    ) -> list[SeedTuple]:
        """Load human-reviewed seed JSON and validate.

        The human reviewer may have:
        - Removed unwanted entries
        - Edited target_goal, materials, mechanism text
        - Reassigned domains
        - Adjusted seed_ids (will be re-assigned sequentially)

        Args:
            reviewed_path: Path to the human-reviewed JSON file.
                Defaults to data/seeds/seed_candidates_reviewed.json.

        Returns:
            List of validated SeedTuple objects with sequential seed_ids.

        Raises:
            FileNotFoundError: If the reviewed file doesn't exist.
            ValueError: If validation fails on any entry.
        """
        if reviewed_path is None:
            reviewed_path = (
                self._project_root / "data" / "seeds" / "seed_candidates_reviewed.json"
            )

        if not reviewed_path.exists():
            raise FileNotFoundError(
                f"Reviewed seed file not found: {reviewed_path}\n"
                "Please review the candidates file and save it as "
                "seed_candidates_reviewed.json"
            )

        raw_data: list[dict[str, Any]] = json.loads(
            reviewed_path.read_text(encoding="utf-8")
        )

        seeds: list[SeedTuple] = []
        errors: list[str] = []

        for idx, entry in enumerate(raw_data):
            try:
                # Allow flexible seed_id during loading — we'll reassign
                entry["seed_id"] = "SEED_000"
                seed = SeedTuple.model_validate(entry)
                seeds.append(seed)
            except Exception as exc:
                errors.append(f"Entry {idx}: {exc}")

        if errors:
            error_msg = "\n".join(errors[:10])
            logger.error(
                "Validation errors in reviewed seeds (%d errors):\n%s",
                len(errors),
                error_msg,
            )
            if not seeds:
                raise ValueError(
                    f"All {len(errors)} entries failed validation:\n{error_msg}"
                )
            logger.warning(
                "Proceeding with %d valid seeds (%d failed)", len(seeds), len(errors)
            )

        # Re-assign sequential IDs
        finalized = self._assign_seed_ids(seeds)
        logger.info(
            "Finalized %d seeds from reviewed file: %s", len(finalized), reviewed_path
        )

        # Save the finalized seeds
        final_path = self._project_root / "data" / "seeds" / "seeds_final.json"
        self.export_candidates(finalized, final_path)

        return finalized
