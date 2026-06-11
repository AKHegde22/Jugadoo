"""
Tests for evaluation metrics.

Validates MCQ accuracy, open-gen scoring, Cohen's Kappa, domain breakdown,
and failure mode analysis.
"""

import pytest

from jugaad_bench.models import (
    CompletionResult,
    JudgeScore,
    KeywordGuardResult,
    FilterResult,
    BenchmarkProblem,
    ProblemMetadata,
    MCQOptions,
    SynthesisRubric,
    ConstraintProfile,
    Domain,
    DomainPerformance,
    FailureModes,
)


def make_completion(problem_id: str, selected_option: str = "C") -> CompletionResult:
    return CompletionResult(
        problem_id=problem_id,
        model_name="test-model",
        prompt_sent="test prompt",
        raw_output=f"The answer is {selected_option}",
        selected_option=selected_option,
    )


def make_judge_score(ca: int = 1, iu: int = 1, pv: int = 1) -> JudgeScore:
    return JudgeScore(
        reasoning="Test evaluation",
        constraint_adherence=ca,
        inventory_utilization=iu,
        physical_viability=pv,
        total_score=ca + iu + pv,
    )


def make_problem(problem_id: str, domain: Domain = Domain.AGRICULTURE,
                 budget: str = "₹0 budget") -> BenchmarkProblem:
    return BenchmarkProblem(
        problem_id=problem_id,
        domain=domain,
        metadata=ProblemMetadata(seed_source="SEED_001", physics_principle="Test physics"),
        prompt_context="A" * 60,
        applied_constraints=ConstraintProfile(
            budget=budget, environment="45°C Heatwave", infrastructure="Total Grid Outage",
        ),
        available_inventory=["item1", "item2"],
        mcq_options=MCQOptions(
            A="A" * 15, B="B" * 15, C="C" * 15, D="D" * 15,
        ),
        ground_truth_option="C",
        ground_truth_synthesis_rubric=SynthesisRubric(
            essential_keywords=["keyword"],
            forbidden_keywords=["forbidden"],
            required_physical_mechanism="Test mechanism description here",
        ),
    )


class TestMCQAccuracy:
    """Tests for MCQ accuracy metric."""

    def test_perfect_accuracy(self):
        from jugaad_bench.eval.metrics import mcq_accuracy

        completions = [
            make_completion("JR-1K-001-01", "C"),
            make_completion("JR-1K-001-02", "A"),
            make_completion("JR-1K-001-03", "B"),
        ]
        ground_truth = {
            "JR-1K-001-01": "C",
            "JR-1K-001-02": "A",
            "JR-1K-001-03": "B",
        }
        assert mcq_accuracy(completions, ground_truth) == 1.0

    def test_zero_accuracy(self):
        from jugaad_bench.eval.metrics import mcq_accuracy

        completions = [
            make_completion("JR-1K-001-01", "A"),
            make_completion("JR-1K-001-02", "B"),
        ]
        ground_truth = {
            "JR-1K-001-01": "C",
            "JR-1K-001-02": "D",
        }
        assert mcq_accuracy(completions, ground_truth) == 0.0

    def test_partial_accuracy(self):
        from jugaad_bench.eval.metrics import mcq_accuracy

        completions = [
            make_completion("JR-1K-001-01", "C"),
            make_completion("JR-1K-001-02", "A"),
            make_completion("JR-1K-001-03", "D"),
            make_completion("JR-1K-001-04", "B"),
        ]
        ground_truth = {
            "JR-1K-001-01": "C",
            "JR-1K-001-02": "B",
            "JR-1K-001-03": "D",
            "JR-1K-001-04": "A",
        }
        assert mcq_accuracy(completions, ground_truth) == 0.5

    def test_empty_completions(self):
        from jugaad_bench.eval.metrics import mcq_accuracy
        assert mcq_accuracy([], {}) == 0.0


class TestOpenGenScore:
    """Tests for open-generation average score."""

    def test_perfect_score(self):
        from jugaad_bench.eval.metrics import open_gen_average_score

        scores = [make_judge_score(1, 1, 1), make_judge_score(1, 1, 1)]
        assert open_gen_average_score(scores) == 3.0

    def test_zero_score(self):
        from jugaad_bench.eval.metrics import open_gen_average_score

        scores = [make_judge_score(0, 0, 0), make_judge_score(0, 0, 0)]
        assert open_gen_average_score(scores) == 0.0

    def test_mixed_scores(self):
        from jugaad_bench.eval.metrics import open_gen_average_score

        scores = [
            make_judge_score(1, 1, 1),  # 3
            make_judge_score(1, 0, 0),  # 1
            make_judge_score(0, 0, 0),  # 0
        ]
        assert abs(open_gen_average_score(scores) - 4/3) < 0.001

    def test_empty_scores(self):
        from jugaad_bench.eval.metrics import open_gen_average_score
        assert open_gen_average_score([]) == 0.0


class TestCohensKappa:
    """Tests for Cohen's Kappa calculation."""

    def test_perfect_agreement(self):
        from jugaad_bench.eval.metrics import cohens_kappa

        human = [0, 1, 2, 3, 1, 2, 3, 0, 1, 2]
        llm   = [0, 1, 2, 3, 1, 2, 3, 0, 1, 2]
        assert cohens_kappa(human, llm) == 1.0

    def test_no_agreement(self):
        from jugaad_bench.eval.metrics import cohens_kappa

        # Systematically opposite scores
        human = [0, 0, 0, 0, 0]
        llm   = [3, 3, 3, 3, 3]
        kappa = cohens_kappa(human, llm)
        assert kappa <= 0.0  # Negative or zero kappa = worse than or equal to chance

    def test_moderate_agreement(self):
        from jugaad_bench.eval.metrics import cohens_kappa

        human = [0, 1, 2, 3, 1, 2, 0, 1, 2, 3]
        llm   = [0, 1, 2, 3, 2, 2, 0, 0, 2, 3]
        kappa = cohens_kappa(human, llm)
        assert 0.3 < kappa < 1.0


class TestFailureModeAnalysis:
    """Tests for failure mode breakdown."""

    def test_failure_modes(self):
        from jugaad_bench.eval.metrics import failure_mode_analysis

        scores = [
            make_judge_score(0, 1, 1),  # constraint violation
            make_judge_score(1, 0, 0),  # physical hallucination + inventory issue
            make_judge_score(1, 1, 0),  # physical hallucination
        ]
        guard_results = [
            KeywordGuardResult(
                problem_id=f"p{i}", model_name="test",
                filter_result=FilterResult.CLEAN, triggered_keywords=[],
            )
            for i in range(3)
        ]

        # Add an auto-fail (task abandonment counted separately)
        scores.append(make_judge_score(0, 0, 0))
        guard_results.append(
            KeywordGuardResult(
                problem_id="p3", model_name="test",
                filter_result=FilterResult.AUTO_FAIL,
                triggered_keywords=["buy"],
            )
        )

        modes = failure_mode_analysis(scores, guard_results)
        assert isinstance(modes, FailureModes)
        assert modes.constraint_violations >= 1
        assert modes.physical_hallucinations >= 1
