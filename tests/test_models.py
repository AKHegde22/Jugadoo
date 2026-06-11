"""
Tests for Pydantic data models.

Validates schema enforcement, field constraints, and model validators.
"""

import pytest
from pydantic import ValidationError

from jugaad_bench.models import (
    SeedTuple,
    Domain,
    RawCase,
    ConstraintProfile,
    MutatedInstance,
    BenchmarkProblem,
    ProblemMetadata,
    MCQOptions,
    SynthesisRubric,
    JudgeScore,
    KeywordGuardResult,
    FilterResult,
    EvalResult,
    DomainPerformance,
    FailureModes,
    CompletionResult,
)


# =============================================================================
# SeedTuple Tests
# =============================================================================


class TestSeedTuple:
    """Tests for the SeedTuple model."""

    def test_valid_seed_tuple(self):
        """A well-formed seed tuple should validate successfully."""
        seed = SeedTuple(
            seed_id="SEED_001",
            domain=Domain.AGRICULTURE,
            target_goal="Deliver consistent micro-targeted irrigation to young saplings under water scarcity.",
            core_physics_mechanism="Gravity-fed fluid dynamics regulated via mechanical constriction (roller clamp).",
            historical_materials_used=[
                "Discarded medical saline/IV bottles",
                "Bamboo sticks",
                "Plastic IV tubing",
            ],
            source_reference="NIF India, 9th National Grassroots Innovation Awards, Case ID: AG-2017-09",
        )
        assert seed.seed_id == "SEED_001"
        assert seed.domain == Domain.AGRICULTURE
        assert len(seed.historical_materials_used) == 3

    def test_invalid_seed_id_format(self):
        """seed_id must match SEED_XXX pattern."""
        with pytest.raises(ValidationError, match="seed_id"):
            SeedTuple(
                seed_id="INVALID_ID",
                domain=Domain.AGRICULTURE,
                target_goal="A" * 20,
                core_physics_mechanism="A" * 10,
                historical_materials_used=["item1"],
                source_reference="Some reference",
            )

    def test_target_goal_too_short(self):
        """target_goal must be at least 20 characters."""
        with pytest.raises(ValidationError, match="target_goal"):
            SeedTuple(
                seed_id="SEED_001",
                domain=Domain.AGRICULTURE,
                target_goal="Too short",
                core_physics_mechanism="A" * 10,
                historical_materials_used=["item1"],
                source_reference="Some reference",
            )

    def test_empty_materials_list(self):
        """historical_materials_used must have at least 1 item."""
        with pytest.raises(ValidationError):
            SeedTuple(
                seed_id="SEED_001",
                domain=Domain.AGRICULTURE,
                target_goal="A" * 20,
                core_physics_mechanism="A" * 10,
                historical_materials_used=[],
                source_reference="Some reference",
            )

    def test_vague_material_rejected(self):
        """Materials that are too vague (< 3 chars) should be rejected."""
        with pytest.raises(ValidationError, match="too vague"):
            SeedTuple(
                seed_id="SEED_001",
                domain=Domain.AGRICULTURE,
                target_goal="A" * 20,
                core_physics_mechanism="A" * 10,
                historical_materials_used=["ok material", "ab"],
                source_reference="Some reference",
            )

    def test_materials_are_stripped(self):
        """Materials should have whitespace stripped."""
        seed = SeedTuple(
            seed_id="SEED_001",
            domain=Domain.AGRICULTURE,
            target_goal="A" * 20,
            core_physics_mechanism="A" * 10,
            historical_materials_used=["  bamboo sticks  ", "  rope  "],
            source_reference="Some reference",
        )
        assert seed.historical_materials_used == ["bamboo sticks", "rope"]

    def test_all_domains(self):
        """All four domains should be valid."""
        for domain in Domain:
            seed = SeedTuple(
                seed_id="SEED_001",
                domain=domain,
                target_goal="A" * 20,
                core_physics_mechanism="A" * 10,
                historical_materials_used=["item"],
                source_reference="ref" * 5,
            )
            assert seed.domain == domain


# =============================================================================
# ConstraintProfile Tests
# =============================================================================


class TestConstraintProfile:
    """Tests for constraint profiles."""

    def test_budget_tier_extraction(self):
        """budget_tier should correctly extract numeric value."""
        assert ConstraintProfile(
            budget="₹0 budget", environment="45°C Heatwave", infrastructure="Total Grid Outage"
        ).budget_tier == 0
        assert ConstraintProfile(
            budget="₹50 budget", environment="45°C Heatwave", infrastructure="Total Grid Outage"
        ).budget_tier == 50
        assert ConstraintProfile(
            budget="₹200 budget", environment="45°C Heatwave", infrastructure="Total Grid Outage"
        ).budget_tier == 200

    def test_hashable(self):
        """ConstraintProfile should be hashable for set operations."""
        p1 = ConstraintProfile(
            budget="₹0 budget", environment="45°C Heatwave", infrastructure="Total Grid Outage"
        )
        p2 = ConstraintProfile(
            budget="₹0 budget", environment="45°C Heatwave", infrastructure="Total Grid Outage"
        )
        p3 = ConstraintProfile(
            budget="₹50 budget", environment="45°C Heatwave", infrastructure="Total Grid Outage"
        )
        assert hash(p1) == hash(p2)
        assert hash(p1) != hash(p3)
        assert len({p1, p2, p3}) == 2


# =============================================================================
# JudgeScore Tests
# =============================================================================


class TestJudgeScore:
    """Tests for judge scoring model."""

    def test_valid_score(self):
        """A valid judge score should validate."""
        score = JudgeScore(
            reasoning="The model correctly used the IV bottles...",
            constraint_adherence=1,
            inventory_utilization=1,
            physical_viability=0,
            total_score=2,
        )
        assert score.total_score == 2

    def test_total_auto_corrects(self):
        """total_score should auto-correct to match binary sum."""
        score = JudgeScore(
            reasoning="Test",
            constraint_adherence=1,
            inventory_utilization=1,
            physical_viability=1,
            total_score=2,  # Wrong — should be 3
        )
        assert score.total_score == 3

    def test_perfect_score(self):
        score = JudgeScore(
            reasoning="Perfect", constraint_adherence=1,
            inventory_utilization=1, physical_viability=1, total_score=3,
        )
        assert score.total_score == 3

    def test_zero_score(self):
        score = JudgeScore(
            reasoning="Fail", constraint_adherence=0,
            inventory_utilization=0, physical_viability=0, total_score=0,
        )
        assert score.total_score == 0


# =============================================================================
# BenchmarkProblem Tests
# =============================================================================


class TestBenchmarkProblem:
    """Tests for the complete benchmark problem schema."""

    def _make_problem(self, **overrides):
        defaults = dict(
            problem_id="JR-1K-001-01",
            domain=Domain.AGRICULTURE,
            metadata=ProblemMetadata(
                seed_source="SEED_001",
                physics_principle="Gravity-fed drip irrigation",
            ),
            prompt_context="A" * 60,
            applied_constraints=ConstraintProfile(
                budget="₹0 budget",
                environment="45°C Heatwave",
                infrastructure="Total Grid Outage",
            ),
            available_inventory=["IV bottles", "bamboo stakes"],
            mcq_options=MCQOptions(
                A="Order online" + " " * 10,
                B="Carry buckets" + " " * 10,
                C="Use IV drip system" + " " * 10,
                D="Crush bottles" + " " * 10,
            ),
            ground_truth_option="C",
            ground_truth_synthesis_rubric=SynthesisRubric(
                essential_keywords=["IV", "drip"],
                forbidden_keywords=["buy", "online"],
                required_physical_mechanism="Gravity-fed drip via roller clamp valve" + " " * 10,
            ),
        )
        defaults.update(overrides)
        return BenchmarkProblem(**defaults)

    def test_valid_problem(self):
        """A complete benchmark problem should validate."""
        problem = self._make_problem()
        assert problem.problem_id == "JR-1K-001-01"
        assert problem.ground_truth_option == "C"

    def test_invalid_problem_id(self):
        """problem_id must match JR-1K-XXX-XX pattern."""
        with pytest.raises(ValidationError, match="problem_id"):
            self._make_problem(problem_id="INVALID")

    def test_invalid_ground_truth(self):
        """ground_truth_option must be A, B, C, or D."""
        with pytest.raises(ValidationError):
            self._make_problem(ground_truth_option="E")


# =============================================================================
# EvalResult Tests
# =============================================================================


class TestEvalResult:
    """Tests for evaluation result schema."""

    def test_valid_eval_result(self):
        result = EvalResult(
            model_under_test="gpt-4o",
            mcq_global_accuracy=0.421,
            open_gen_global_average_score=1.12,
            domain_performance={
                "agriculture": DomainPerformance(mcq=0.45, open_gen=1.25),
                "healthcare": DomainPerformance(mcq=0.31, open_gen=0.85),
            },
            failure_modes=FailureModes(
                constraint_violations=412,
                physical_hallucinations=318,
                task_abandonment=149,
            ),
        )
        assert result.mcq_global_accuracy == 0.421
        assert result.failure_modes.total == 412 + 318 + 149

    def test_accuracy_bounds(self):
        """MCQ accuracy must be between 0 and 1."""
        with pytest.raises(ValidationError):
            EvalResult(
                model_under_test="test",
                mcq_global_accuracy=1.5,
                open_gen_global_average_score=1.0,
            )


# =============================================================================
# RawCase Tests
# =============================================================================


class TestRawCase:
    """Tests for raw case schema."""

    def test_valid_raw_case(self):
        case = RawCase(
            source="nif_pdf",
            url_or_path="/path/to/file.pdf",
            raw_text="A" * 30,
        )
        assert case.source == "nif_pdf"

    def test_raw_text_too_short(self):
        with pytest.raises(ValidationError, match="raw_text"):
            RawCase(
                source="nif_pdf",
                url_or_path="/path",
                raw_text="Short",
            )
