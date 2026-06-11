"""
Tests for the LLM-as-a-Judge.

Validates judge prompt construction, response parsing, and scoring logic.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from jugaad_bench.models import (
    JudgeScore,
    BenchmarkProblem,
    ProblemMetadata,
    MCQOptions,
    SynthesisRubric,
    ConstraintProfile,
    Domain,
)


def make_test_problem() -> BenchmarkProblem:
    """Create a test benchmark problem for judge testing."""
    return BenchmarkProblem(
        problem_id="JR-1K-001-01",
        domain=Domain.AGRICULTURE,
        metadata=ProblemMetadata(
            seed_source="SEED_001",
            physics_principle="Gravity-fed drip irrigation via roller clamp",
        ),
        prompt_context=(
            "A farmer in a remote village in Maharashtra needs to keep 50 fragile "
            "saplings alive through an unexpected dry spell. The local market is "
            "closed due to a regional strike, and the farmer has a budget of ₹0."
        ),
        applied_constraints=ConstraintProfile(
            budget="₹0 budget",
            environment="45°C Heatwave",
            infrastructure="Total Grid Outage",
        ),
        available_inventory=[
            "50 discarded plastic medical IV/saline bottles with tubing and roller clamps",
            "50 bamboo stakes",
            "1 roll of coconut fiber rope",
            "Standard well water",
        ],
        mcq_options=MCQOptions(
            A="Order a commercial PVC drip irrigation kit online.",
            B="Manually carry 20-liter metal buckets of water to every plant every hour.",
            C="Fill the discarded IV bottles with water, tie them upside down to bamboo stakes, and use roller clamps to tune the drip.",
            D="Crush the plastic IV bottles to form a synthetic mulch layer over the soil.",
        ),
        ground_truth_option="C",
        ground_truth_synthesis_rubric=SynthesisRubric(
            essential_keywords=["IV bottle", "saline", "clamp", "drip", "bamboo"],
            forbidden_keywords=["order", "online", "buy", "motor", "pump", "app"],
            required_physical_mechanism=(
                "The model must explain using the roller clamp or a minor puncture "
                "to regulate water flow rate from the inverted plastic container "
                "to simulate low-cost drip irrigation."
            ),
        ),
    )


class TestJudgePromptConstruction:
    """Tests for judge prompt construction."""

    def test_judge_prompt_contains_inventory(self):
        """The judge prompt should include the problem inventory."""
        from jugaad_bench.eval.judge import JugaadJudge, _build_user_prompt
    
        problem = make_test_problem()
        prompt = _build_user_prompt(problem, "test response")
        assert "IV/saline bottles" in prompt
        assert "bamboo stakes" in prompt

    def test_judge_prompt_contains_rubric(self):
        """The judge prompt should include the synthesis rubric."""
        from jugaad_bench.eval.judge import JugaadJudge, _build_user_prompt
    
        problem = make_test_problem()
        prompt = _build_user_prompt(problem, "test response")
        assert "roller clamp" in prompt
        assert "drip irrigation" in prompt

    def test_judge_prompt_contains_model_response(self):
        """The judge prompt should include the model's response to grade."""
        from jugaad_bench.eval.judge import JugaadJudge, _build_user_prompt
    
        problem = make_test_problem()
        model_output = "Use the IV bottles as drip irrigators by hanging them on bamboo."
        prompt = _build_user_prompt(problem, model_output)
        assert model_output in prompt


class TestJudgeScoreValidation:
    """Tests for judge score validation logic."""

    def test_valid_perfect_score(self):
        score = JudgeScore(
            reasoning="All criteria met.",
            constraint_adherence=1,
            inventory_utilization=1,
            physical_viability=1,
            total_score=3,
        )
        assert score.total_score == 3

    def test_total_auto_corrects_to_sum(self):
        """If total doesn't match sum, it should auto-correct."""
        score = JudgeScore(
            reasoning="Test",
            constraint_adherence=1,
            inventory_utilization=0,
            physical_viability=1,
            total_score=1,  # Wrong — should be 2
        )
        assert score.total_score == 2

    def test_all_zeros(self):
        score = JudgeScore(
            reasoning="Nothing correct.",
            constraint_adherence=0,
            inventory_utilization=0,
            physical_viability=0,
            total_score=0,
        )
        assert score.total_score == 0
