"""
JugaadReasoning-1K: Constraint-Satisfying Resource-Substitution Benchmark

All Pydantic data models for the entire pipeline.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class Domain(str, enum.Enum):
    """The four target domains for seed distribution."""

    AGRICULTURE = "agriculture"
    HEALTHCARE = "healthcare"
    CONSTRUCTION = "construction"
    STREET_VENDING = "street_vending"


class FilterResult(str, enum.Enum):
    """Result of the keyword guard filter."""

    AUTO_FAIL = "auto_fail"
    CLEAN = "clean"


class FailureCategory(str, enum.Enum):
    """Categories of failure modes in open-generation evaluation."""

    CONSTRAINT_VIOLATION = "constraint_violation"
    PHYSICAL_HALLUCINATION = "physical_hallucination"
    TASK_ABANDONMENT = "task_abandonment"


# =============================================================================
# Phase 1: Seed Tuples (Raw Data Extraction)
# =============================================================================


class SeedTuple(BaseModel):
    """
    A single real-world innovation seed extracted from NIF, Honey Bee Network,
    SRISTI, or validated grassroots media.

    Schema defined in PRD Section 2.B.
    """

    seed_id: str = Field(
        ...,
        pattern=r"^SEED_\d{3}$",
        description="Unique identifier in the format SEED_001 through SEED_100.",
    )
    domain: Domain = Field(
        ...,
        description="One of the four target domains.",
    )
    target_goal: str = Field(
        ...,
        min_length=20,
        description=(
            "A clear description of the physical problem the innovation solves. "
            "Must describe the goal, not the solution."
        ),
    )
    core_physics_mechanism: str = Field(
        ...,
        min_length=10,
        description=(
            "The underlying physical principle or mechanism that makes the "
            "innovation work (e.g., 'Gravity-fed fluid dynamics regulated via "
            "mechanical constriction')."
        ),
    )
    historical_materials_used: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "The actual materials used in the documented real-world solution. "
            "Each entry should be a specific, tangible item."
        ),
    )
    source_reference: str = Field(
        ...,
        min_length=5,
        description=(
            "Citation to the original source (e.g., 'NIF India, 9th National "
            "Grassroots Innovation Awards, Case ID: AG-2017-09')."
        ),
    )

    @field_validator("historical_materials_used")
    @classmethod
    def validate_materials_are_specific(cls, v: list[str]) -> list[str]:
        """Ensure each material entry is specific enough to be physically meaningful."""
        for material in v:
            if len(material.strip()) < 3:
                raise ValueError(
                    f"Material entry too vague: '{material}'. "
                    "Each material must be a specific, tangible item."
                )
        return [m.strip() for m in v]


class RawCase(BaseModel):
    """A raw, unstructured innovation case extracted by a scraper before
    LLM-assisted structuring into a SeedTuple."""

    source: str = Field(..., description="Which scraper produced this (nif_pdf, nif_web, etc.).")
    url_or_path: str = Field(..., description="Source URL or file path.")
    raw_text: str = Field(..., min_length=20, description="Full extracted text of the innovation.")
    title: str | None = Field(None, description="Title if available.")
    innovator_name: str | None = Field(None, description="Innovator name if available.")
    location: str | None = Field(None, description="Location/state if available.")
    category: str | None = Field(None, description="Category tag if available.")


# =============================================================================
# Phase 2: Constraint Profiles & Mutation Matrix
# =============================================================================


class ConstraintProfile(BaseModel):
    """A specific combination of financial, environmental, and infrastructural
    constraints applied to a seed."""

    budget: str = Field(
        ...,
        description="Financial constraint (e.g., '₹0 budget', '₹50 budget', '₹200 budget').",
    )
    environment: str = Field(
        ...,
        description="Environmental constraint (e.g., '45°C Heatwave', 'Monsoon Flash Flood').",
    )
    infrastructure: str = Field(
        ...,
        description="Infrastructural constraint (e.g., 'Total Grid Outage').",
    )

    @property
    def budget_tier(self) -> int:
        """Extract numeric budget tier for analysis."""
        if "₹0" in self.budget:
            return 0
        elif "₹50" in self.budget:
            return 50
        elif "₹200" in self.budget:
            return 200
        return -1

    def __hash__(self) -> int:
        return hash((self.budget, self.environment, self.infrastructure))


class MutatedInstance(BaseModel):
    """A single row in the 1,000-row mutation matrix.
    Combines a seed with a specific constraint profile and adjusted inventory."""

    instance_id: str = Field(
        ...,
        pattern=r"^MI_\d{3}_\d{2}$",
        description="Format: MI_{seed_num}_{mutation_num}, e.g., MI_001_01",
    )
    seed_id: str = Field(..., description="Reference to the parent SeedTuple.")
    domain: Domain
    target_goal: str
    core_physics_mechanism: str
    applied_constraints: ConstraintProfile
    adjusted_inventory: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "The available materials adjusted for constraints. Must still "
            "enable the core physics mechanism."
        ),
    )
    context_narrative: str = Field(
        ...,
        min_length=20,
        description=(
            "A brief narrative fragment describing the scenario context "
            "(location, character, situation)."
        ),
    )
    original_materials: list[str] = Field(
        ...,
        description="The original seed's historical_materials_used for reference.",
    )


# =============================================================================
# Phase 3: Formatted Benchmark Problems
# =============================================================================


class SynthesisRubric(BaseModel):
    """Ground truth rubric for evaluating open-generation responses."""

    essential_keywords: list[str] = Field(
        ...,
        min_length=1,
        description="Keywords that MUST appear in a correct solution.",
    )
    forbidden_keywords: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Keywords that MUST NOT appear (e.g., 'buy', 'online', 'motor'). "
            "Presence triggers AUTO_FAIL."
        ),
    )
    required_physical_mechanism: str = Field(
        ...,
        min_length=10,
        description=(
            "Description of the physics the model must explain to earn "
            "the physical_viability point."
        ),
    )


class MCQOptions(BaseModel):
    """Four multiple-choice options for a discriminative problem."""

    A: str = Field(..., min_length=10)
    B: str = Field(..., min_length=10)
    C: str = Field(..., min_length=10)
    D: str = Field(..., min_length=10)


class BenchmarkProblem(BaseModel):
    """
    A single benchmark problem in the JugaadReasoning-1K dataset.
    Contains both MCQ and Open Generation formats.

    Schema defined in PRD Section 4.B.
    """

    problem_id: str = Field(
        ...,
        pattern=r"^JR-1K-\d{3}-\d{2}$",
        description="Format: JR-1K-{seed_num}-{mutation_num}",
    )
    domain: Domain
    metadata: ProblemMetadata
    prompt_context: str = Field(
        ...,
        min_length=50,
        description="Natural language scenario description.",
    )
    applied_constraints: ConstraintProfile
    available_inventory: list[str] = Field(
        ...,
        min_length=1,
        description="Enumerated list of materials available to the protagonist.",
    )

    # MCQ format
    mcq_options: MCQOptions
    ground_truth_option: Literal["A", "B", "C", "D"]

    # Open Generation rubric
    ground_truth_synthesis_rubric: SynthesisRubric

    @model_validator(mode="after")
    def validate_ground_truth_in_options(self) -> "BenchmarkProblem":
        """Ensure the ground truth option letter is valid."""
        gt = self.ground_truth_option
        if gt not in ("A", "B", "C", "D"):
            raise ValueError(f"ground_truth_option must be A, B, C, or D, got '{gt}'")
        return self


class ProblemMetadata(BaseModel):
    """Metadata linking a benchmark problem back to its seed."""

    seed_source: str = Field(..., description="Reference to SEED_xxx.")
    physics_principle: str = Field(..., description="Short summary of the core physics.")


# =============================================================================
# Phase 4: Evaluation Results
# =============================================================================


class CompletionResult(BaseModel):
    """Raw result from running a single model on a single problem."""

    problem_id: str
    model_name: str
    prompt_sent: str
    raw_output: str
    selected_option: str | None = Field(
        None,
        description="For MCQ: extracted letter (A/B/C/D). None for open-gen.",
    )
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JudgeScore(BaseModel):
    """
    Output of the LLM-as-a-Judge for a single open-generation response.

    Schema defined in PRD Section 6.C.
    """

    reasoning: str = Field(
        ...,
        description="Chain-of-thought reasoning BEFORE scoring.",
    )
    constraint_adherence: Literal[0, 1] = Field(
        ...,
        description="1 if model avoided unlisted materials / budget violations.",
    )
    inventory_utilization: Literal[0, 1] = Field(
        ...,
        description="1 if model correctly used the specific listed inventory items.",
    )
    physical_viability: Literal[0, 1] = Field(
        ...,
        description="1 if proposed mechanism matches the physics rubric.",
    )
    total_score: int = Field(
        ...,
        ge=0,
        le=3,
        description="Sum of the three binary checks (0-3).",
    )

    @model_validator(mode="after")
    def validate_total_is_sum(self) -> "JudgeScore":
        expected = self.constraint_adherence + self.inventory_utilization + self.physical_viability
        if self.total_score != expected:
            # Auto-correct the total to match the actual sum
            self.total_score = expected
        return self


class KeywordGuardResult(BaseModel):
    """Result of the keyword guard filter for a single response."""

    problem_id: str
    model_name: str
    filter_result: FilterResult
    triggered_keywords: list[str] = Field(
        default_factory=list,
        description="Which forbidden keywords were found (empty if CLEAN).",
    )


class DomainPerformance(BaseModel):
    """Per-domain performance metrics."""

    mcq: float = Field(..., ge=0.0, le=1.0, description="MCQ accuracy for this domain.")
    open_gen: float = Field(..., ge=0.0, le=3.0, description="Average open-gen score (0-3).")


class FailureModes(BaseModel):
    """Breakdown of failure modes across all problems for a model."""

    constraint_violations: int = Field(0, ge=0)
    physical_hallucinations: int = Field(0, ge=0)
    task_abandonment: int = Field(0, ge=0)

    @property
    def total(self) -> int:
        return self.constraint_violations + self.physical_hallucinations + self.task_abandonment


class EvalResult(BaseModel):
    """
    Complete evaluation result for a single model.

    Schema defined in PRD Section 7.A.
    """

    eval_timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_under_test: str
    mcq_global_accuracy: float = Field(..., ge=0.0, le=1.0)
    open_gen_global_average_score: float = Field(..., ge=0.0, le=3.0)
    domain_performance: dict[str, DomainPerformance] = Field(
        default_factory=dict,
        description="Keyed by domain name (agriculture, healthcare, etc.).",
    )
    failure_modes: FailureModes = Field(default_factory=FailureModes)

    # Additional analytics
    budget_tier_performance: dict[str, float] | None = Field(
        None,
        description="Average open-gen score keyed by budget tier (₹0, ₹50, ₹200).",
    )
    total_problems_evaluated: int = 0
    total_cost_usd: float = 0.0


# =============================================================================
# Configuration Models
# =============================================================================


class ModelConfig(BaseModel):
    """Configuration for a single model to evaluate."""

    name: str
    provider: str = Field(
        ...,
        description="Provider SDK: 'openai', 'anthropic', 'google', 'vllm'.",
    )
    model_id: str = Field(
        ...,
        description="Provider-specific model identifier.",
    )
    max_tokens: int = 1024
    api_base: str | None = Field(
        None,
        description="Custom API base URL (for OpenAI-compatible endpoints).",
    )
    api_key_env: str | None = Field(
        None,
        description="Environment variable name for the API key.",
    )


class PipelineConfig(BaseModel):
    """Top-level configuration parsed from pipeline_config.yaml."""

    data: DataConfig
    constraints: ConstraintConfig
    eval: EvalConfig
    models: ModelsConfig
    plots: PlotsConfig
    scrapers: ScrapersConfig | None = None


class DataConfig(BaseModel):
    seed_target_count: int = 100
    domain_distribution: dict[str, int]
    mutations_per_seed: int = 10
    total_benchmark_size: int = 1000
    paths: dict[str, str] = Field(default_factory=dict)


class ConstraintConfig(BaseModel):
    financial: list[str]
    environmental: list[str]
    infrastructural: list[str]


class EvalConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 5
    checkpoint_every: int = 10
    dry_run: bool = False
    judge: JudgeConfig = Field(default_factory=lambda: JudgeConfig())


class JudgeConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 512
    kappa_threshold: float = 0.80
    calibration_sample_size: int = 100


class ModelsConfig(BaseModel):
    frontier: list[ModelConfig] = Field(default_factory=list)
    open_weights: list[ModelConfig] = Field(default_factory=list)
    indic_native: list[ModelConfig] = Field(default_factory=list)

    @property
    def all_models(self) -> list[ModelConfig]:
        return self.frontier + self.open_weights + self.indic_native


class PlotsConfig(BaseModel):
    style: str = "science"
    dpi: int = 300
    formats: list[str] = Field(default_factory=lambda: ["pdf", "png"])
    output_dir: str = "plots"
    figsize_double_column: list[float] = Field(default_factory=lambda: [7.0, 4.0])
    figsize_single_column: list[float] = Field(default_factory=lambda: [3.5, 2.5])
    colorblind_safe: bool = True


class ScrapersConfig(BaseModel):
    nif_pdf: dict | None = None
    nif_web: dict | None = None
    honeybee: dict | None = None
    youtube: dict | None = None
