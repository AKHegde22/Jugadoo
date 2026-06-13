#!/usr/bin/env python3
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import BenchmarkProblem, CompletionResult, JudgeScore, KeywordGuardResult, EvalResult
from jugaad_bench.eval.metrics import aggregate_eval_result
from jugaad_bench.analytics.result_logger import ResultLogger
from jugaad_bench.eval.keyword_guard import KeywordGuard

def main():
    with open("data/benchmark/jugaad_reasoning_1k_full.json", "r") as f:
        problems_data = json.load(f)
    problem_map = {p["problem_id"]: BenchmarkProblem.model_validate(p) for p in problems_data}

    # Load 107 judge scores
    judge_scores = []
    judged_ids = set()
    with open("eval_outputs/judge_audit.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if data["judge_model"] == "gpt-4o":
                score = JudgeScore.model_validate(data["score"])
                judge_scores.append(score)
                judged_ids.add(data["problem_id"])

    print(f"Loaded {len(judge_scores)} judge scores.")

    # Load MCQ completions (keep only the ones we judged for parity, or keep all)
    # Actually, metrics.py aggregate_eval_result assumes problems, judge_scores, guard_results are aligned!
    # Let's align them.
    aligned_problems = []
    aligned_scores = []
    
    # We also need open gen completions for the keyword guard.
    og_comps = {}
    with open("data/results/gpt-5.5_opengen_completions.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            c = CompletionResult.model_validate_json(line)
            og_comps[c.problem_id] = c
            
    mcq_comps = []
    with open("data/results/gpt-5.5_mcq_completions.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            c = CompletionResult.model_validate_json(line)
            if c.problem_id in judged_ids:
                mcq_comps.append(c)

    guard = KeywordGuard()
    guard_results = []
    
    # Let's sort the scored ids so it's deterministic
    for pid in sorted(list(judged_ids)):
        problem = problem_map[pid]
        aligned_problems.append(problem)
        
        # Find score
        # Note: audit log has problem_id
        score_data = next(s for line in open("eval_outputs/judge_audit.jsonl") if json.loads(line)["problem_id"] == pid for s in [JudgeScore.model_validate(json.loads(line)["score"])])
        aligned_scores.append(score_data)
        
        # Guard
        og = og_comps[pid]
        g_res = guard.check(og.raw_output, problem.ground_truth_synthesis_rubric.forbidden_keywords)
        g_res.problem_id = pid
        guard_results.append(g_res)
        
    print(f"Aligned {len(aligned_problems)} problems.")
    
    res = aggregate_eval_result(
        model_name="gpt-5.5",
        mcq_completions=mcq_comps,
        opengen_completions=[og_comps[p.problem_id] for p in aligned_problems],
        judge_scores=aligned_scores,
        guard_results=guard_results,
        problems=aligned_problems,
    )
    
    # Write to eval_log.jsonl
    logger = ResultLogger(output_path=Path("data/results/eval_log.jsonl"))
    logger.log_eval_result(res)
    print("Done!")

if __name__ == "__main__":
    main()
