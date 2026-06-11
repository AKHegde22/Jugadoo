"""
Evaluation metrics for JugaadReasoning-1K.

Computes MCQ accuracy, open-gen average scores, Cohen's κ, domain breakdowns,
failure-mode analysis, budget-tier analysis, and aggregated ``EvalResult``.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sklearn.metrics import cohen_kappa_score

from jugaad_bench.models import (
    BenchmarkProblem,
    CompletionResult,
    DomainPerformance,
    EvalResult,
    FailureCategory,
    FailureModes,
    JudgeScore,
    KeywordGuardResult,
    FilterResult,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────


def mcq_accuracy(
    completions: list[CompletionResult],
    ground_truth: dict[str, str],
) -> float:
    """
    Compute MCQ accuracy.

    Args:
        completions: List of completion results with ``selected_option`` set.
        ground_truth: Mapping of ``problem_id → correct option letter``.

    Returns:
        Accuracy as a float in [0, 1].  Returns 0.0 if no valid completions.
    """
    if not completions:
        return 0.0

    correct = 0
    total = 0
    for c in completions:
        if c.problem_id not in ground_truth:
            continue
        total += 1
        if c.selected_option and c.selected_option.upper() == ground_truth[c.problem_id].upper():
            correct += 1

    return correct / total if total > 0 else 0.0


def open_gen_average_score(scores: list[JudgeScore]) -> float:
    """
    Compute the average total score across open-generation judgements.

    Args:
        scores: List of judge scores (each has ``total_score`` in [0, 3]).

    Returns:
        Mean score in [0, 3].  Returns 0.0 if no scores.
    """
    if not scores:
        return 0.0
    return sum(s.total_score for s in scores) / len(scores)


def cohens_kappa(
    human_labels: list[int],
    llm_labels: list[int],
) -> float:
    """
    Compute Cohen's κ between human and LLM judge labels.

    Args:
        human_labels: List of human scores (integer labels).
        llm_labels: List of LLM judge scores (same length as *human_labels*).

    Returns:
        Cohen's κ coefficient.

    Raises:
        ValueError: If the lists have different lengths or are empty.
    """
    if len(human_labels) != len(llm_labels):
        raise ValueError(
            f"Label lists must have the same length; got "
            f"{len(human_labels)} vs {len(llm_labels)}."
        )
    if not human_labels:
        raise ValueError("Cannot compute κ on empty label lists.")

    return float(cohen_kappa_score(human_labels, llm_labels))


# ─────────────────────────────────────────────────────────────────────────────
# Domain breakdown
# ─────────────────────────────────────────────────────────────────────────────


def domain_breakdown(
    completions: list[CompletionResult],
    ground_truth: dict[str, str],
    problems: list[BenchmarkProblem],
    scores: list[JudgeScore],
) -> dict[str, DomainPerformance]:
    """
    Compute per-domain MCQ accuracy and open-gen average score.

    Args:
        completions: MCQ completion results.
        ground_truth: ``problem_id → correct option`` mapping.
        problems: Full problem list (used to map IDs to domains).
        scores: Open-gen judge scores aligned with *problems*.

    Returns:
        Dict keyed by domain name → ``DomainPerformance``.
    """
    # Build problem → domain lookup
    id_to_domain: dict[str, str] = {p.problem_id: p.domain.value for p in problems}

    # Group MCQ completions by domain
    domain_mcq_correct: dict[str, int] = defaultdict(int)
    domain_mcq_total: dict[str, int] = defaultdict(int)
    for c in completions:
        domain = id_to_domain.get(c.problem_id)
        if domain is None:
            continue
        if c.problem_id in ground_truth:
            domain_mcq_total[domain] += 1
            if (
                c.selected_option
                and c.selected_option.upper() == ground_truth[c.problem_id].upper()
            ):
                domain_mcq_correct[domain] += 1

    # Map scores to problems by index for domain resolution
    # Build a problem_id → score mapping from aligned lists
    id_to_score: dict[str, JudgeScore] = {}
    for problem, score in zip(problems, scores):
        id_to_score[problem.problem_id] = score

    domain_og_scores: dict[str, list[float]] = defaultdict(list)
    for pid, score in id_to_score.items():
        domain = id_to_domain.get(pid)
        if domain:
            domain_og_scores[domain].append(float(score.total_score))

    # Build result
    all_domains = set(id_to_domain.values())
    result: dict[str, DomainPerformance] = {}
    for domain in sorted(all_domains):
        mcq_total = domain_mcq_total.get(domain, 0)
        mcq_correct = domain_mcq_correct.get(domain, 0)
        mcq_acc = mcq_correct / mcq_total if mcq_total > 0 else 0.0

        og_list = domain_og_scores.get(domain, [])
        og_avg = sum(og_list) / len(og_list) if og_list else 0.0

        result[domain] = DomainPerformance(mcq=mcq_acc, open_gen=og_avg)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode analysis
# ─────────────────────────────────────────────────────────────────────────────


def failure_mode_analysis(
    scores: list[JudgeScore],
    guard_results: list[KeywordGuardResult],
) -> FailureModes:
    """
    Classify failures into constraint violations, physical hallucinations,
    and task abandonment.

    Logic:
    - **Constraint Violation**: keyword guard triggered (AUTO_FAIL) OR
      ``constraint_adherence == 0``.
    - **Physical Hallucination**: ``physical_viability == 0`` while the
      model at least attempted a solution (not abandoned).
    - **Task Abandonment**: ``total_score == 0`` AND the response likely
      contains a refusal/inability statement (heuristic: very short or
      all three dimensions are 0 without keyword trigger).

    Args:
        scores: Judge scores.
        guard_results: Keyword guard results aligned with *scores*.

    Returns:
        Aggregated ``FailureModes``.
    """
    constraint_violations = 0
    physical_hallucinations = 0
    task_abandonment = 0

    # Build guard lookup
    guard_map: dict[str, KeywordGuardResult] = {
        g.problem_id: g for g in guard_results
    }

    for score, guard in zip(scores, guard_results):
        is_failure = score.total_score < 3

        if not is_failure:
            continue

        guard_triggered = guard.filter_result == FilterResult.AUTO_FAIL

        # Task abandonment: all three dimensions are 0
        if score.total_score == 0:
            task_abandonment += 1
        elif score.constraint_adherence == 0 or guard_triggered:
            constraint_violations += 1
        elif score.physical_viability == 0:
            physical_hallucinations += 1
        else:
            # inventory_utilization == 0 alone: classify as constraint violation
            constraint_violations += 1

    return FailureModes(
        constraint_violations=constraint_violations,
        physical_hallucinations=physical_hallucinations,
        task_abandonment=task_abandonment,
    )


def failure_mode_analysis_per_problem(
    scores: list[JudgeScore],
    guard_results: list[KeywordGuardResult],
    problems: list[BenchmarkProblem],
) -> FailureModes:
    """
    Per-problem failure classification using aligned problem/score/guard lists.

    This version uses problem IDs to correctly correlate guard results with scores.

    Args:
        scores: Judge scores, one per problem, aligned with *problems*.
        guard_results: Keyword guard results, one per problem, aligned with *problems*.
        problems: Benchmark problems.

    Returns:
        Aggregated ``FailureModes``.
    """
    constraint_violations = 0
    physical_hallucinations = 0
    task_abandonment = 0

    for problem, score, guard in zip(problems, scores, guard_results):
        if score.total_score == 3:
            continue  # No failure

        guard_triggered = guard.filter_result == FilterResult.AUTO_FAIL

        if score.total_score == 0:
            task_abandonment += 1
        elif guard_triggered or score.constraint_adherence == 0:
            constraint_violations += 1
        elif score.physical_viability == 0:
            physical_hallucinations += 1
        else:
            constraint_violations += 1

    return FailureModes(
        constraint_violations=constraint_violations,
        physical_hallucinations=physical_hallucinations,
        task_abandonment=task_abandonment,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Budget tier analysis
# ─────────────────────────────────────────────────────────────────────────────


def budget_tier_analysis(
    scores: list[JudgeScore],
    problems: list[BenchmarkProblem],
) -> dict[str, float]:
    """
    Compute average open-gen score per budget tier.

    Args:
        scores: Judge scores, aligned with *problems*.
        problems: Benchmark problems.

    Returns:
        Dict keyed by budget label (e.g., ``"₹0"``, ``"₹50"``, ``"₹200"``)
        → average score.
    """
    tier_scores: dict[str, list[float]] = defaultdict(list)

    for problem, score in zip(problems, scores):
        tier = problem.applied_constraints.budget_tier
        if tier == 0:
            label = "₹0"
        elif tier == 50:
            label = "₹50"
        elif tier == 200:
            label = "₹200"
        else:
            label = f"₹{tier}"
        tier_scores[label].append(float(score.total_score))

    return {
        label: sum(vals) / len(vals) if vals else 0.0
        for label, vals in sorted(tier_scores.items())
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_eval_result(
    model_name: str,
    mcq_completions: list[CompletionResult],
    opengen_completions: list[CompletionResult],
    judge_scores: list[JudgeScore],
    guard_results: list[KeywordGuardResult],
    problems: list[BenchmarkProblem],
) -> EvalResult:
    """
    Build a complete ``EvalResult`` for one model.

    Args:
        model_name: Name of the model under test.
        mcq_completions: MCQ completion results.
        opengen_completions: Open-generation completion results.
        judge_scores: Judge scores aligned with *problems*.
        guard_results: Keyword guard results aligned with *problems*.
        problems: Benchmark problems.

    Returns:
        Fully populated ``EvalResult``.
    """
    # Ground truth mapping
    gt_map = {p.problem_id: p.ground_truth_option for p in problems}

    # MCQ accuracy
    mcq_acc = mcq_accuracy(mcq_completions, gt_map)

    # Open-gen average
    og_avg = open_gen_average_score(judge_scores)

    # Domain breakdown
    domain_perf = domain_breakdown(mcq_completions, gt_map, problems, judge_scores)

    # Failure modes
    failures = failure_mode_analysis_per_problem(judge_scores, guard_results, problems)

    # Budget tier
    budget_perf = budget_tier_analysis(judge_scores, problems)

    # Total cost
    total_cost = sum(
        c.input_tokens * 0.001 + c.output_tokens * 0.002
        for c in mcq_completions + opengen_completions
    ) / 1000  # rough estimate

    return EvalResult(
        model_under_test=model_name,
        mcq_global_accuracy=mcq_acc,
        open_gen_global_average_score=og_avg,
        domain_performance=domain_perf,
        failure_modes=failures,
        budget_tier_performance=budget_perf,
        total_problems_evaluated=len(problems),
        total_cost_usd=total_cost,
    )
