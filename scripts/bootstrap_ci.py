#!/usr/bin/env python3
"""Bootstrap 95% CI for HMR and Δ HMR across the eval'd candidates.

For each candidate AC run we have per-topic correctness + confidence + nugget
counts. Bootstrap: resample 65 topics with replacement, recompute HMR, repeat
1000×, take the 2.5th and 97.5th percentiles.

For Δ HMR between two candidates, the bootstrap is paired (resample the same
topics for both candidates).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

EVAL_DIR = BASE / "data" / "eval" / "ac_runs"


def per_topic_results(eval_path: Path) -> list[QuestionResult]:
    d = json.loads(eval_path.read_text())
    out = []
    for qid, q in d["per_question"].items():
        out.append(QuestionResult(
            question_id=qid,
            correct=bool(q.get("correct", False)),
            confidence=float(q.get("confidence", 0)),
            nuggets_returned=int(q.get("n_returned", 0)),
            nuggets_relevant=int(q.get("n_relevant", 0)),
        ))
    return out


def bootstrap_ci(results: list[QuestionResult], n_iter: int = 1000,
                 seed: int = 42) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_lo, ci_hi) for HMR."""
    rng = random.Random(seed)
    base_hmr = compute_metrics(results).HMR
    samples = []
    n = len(results)
    for _ in range(n_iter):
        resampled = [rng.choice(results) for _ in range(n)]
        samples.append(compute_metrics(resampled).HMR)
    samples.sort()
    return base_hmr, samples[int(0.025 * n_iter)], samples[int(0.975 * n_iter)]


def paired_bootstrap_delta(results_a: list[QuestionResult],
                            results_b: list[QuestionResult],
                            n_iter: int = 1000,
                            seed: int = 42) -> tuple[float, float, float]:
    """Returns (point_estimate_delta, ci_lo, ci_hi) for HMR(B) - HMR(A).

    Resamples topic indices and applies the same indices to both A and B (paired).
    Requires same topic ordering in both.
    """
    rng = random.Random(seed)
    n = len(results_a)
    assert len(results_b) == n, "Paired bootstrap requires same topic count"
    base_a = compute_metrics(results_a).HMR
    base_b = compute_metrics(results_b).HMR
    base_delta = base_b - base_a
    samples = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        ra = [results_a[i] for i in idx]
        rb = [results_b[i] for i in idx]
        samples.append(compute_metrics(rb).HMR - compute_metrics(ra).HMR)
    samples.sort()
    return base_delta, samples[int(0.025 * n_iter)], samples[int(0.975 * n_iter)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--output", default=str(BASE / "data/eval/ac_runs/bootstrap_ci.json"))
    args = parser.parse_args()

    # Per-candidate CIs for the most important configurations
    candidates = [
        ("ensemble_top1", "ensemble_top1.json"),
        ("ensemble_top3", "ensemble_top3.json"),
        ("bitem_only_refined_sonnet_A", "bitem_only_refined_sonnet_A.json"),
        ("bitem_only_broad_A", "bitem_only_broad_A.json"),
        ("refined_sonnet_A", "refined_sonnet_A.json"),
        ("refined_ce_A", "refined_ce_A.json"),
        ("refined_lexical_A", "refined_lexical_A.json"),
        ("broad_A", "broad_A.json"),
        ("bitem_only_s1_haiku_A", "bitem_only_s1_haiku_A.json"),
        ("bitem_only_s4_haiku_A", "bitem_only_s4_haiku_A.json"),
        ("bitem_only_all_haiku_A", "bitem_only_all_haiku_A.json"),
        ("bitem_only_opus_A", "bitem_only_opus_A.json"),
        ("retrank_plus_error404_refined_sonnet_A", "retrank_plus_error404_refined_sonnet_A.json"),
    ]

    print(f"\n{'─'*80}")
    print(f"Bootstrap 95% CI per candidate ({args.n_iter} iterations)")
    print(f"{'─'*80}")
    print(f"{'Candidate':<42} {'HMR':>7}  {'95% CI':>15}")

    per_cand: dict[str, dict] = {}
    cached = {}
    for name, fname in candidates:
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"  SKIP {name}: {fname} not found")
            continue
        results = per_topic_results(path)
        cached[name] = results
        hmr, lo, hi = bootstrap_ci(results, n_iter=args.n_iter)
        per_cand[name] = {"hmr": hmr, "ci_lo": lo, "ci_hi": hi,
                           "ci_width": hi - lo}
        print(f"  {name:<42} {hmr:>7.4f}  [{lo:.3f}, {hi:.3f}]")

    # Paired Δ for key comparisons
    comparisons = [
        ("RQ2 (BITEM vs Tier-1, broad)",
         "bitem_only_broad_A", "broad_A"),
        ("RQ6 (Sonnet refinement vs broad on Tier-1)",
         "broad_A", "refined_sonnet_A"),
        ("RQ6b (CE vs broad on Tier-1)",
         "broad_A", "refined_ce_A"),
        ("RQ6b (lexical vs broad on Tier-1)",
         "broad_A", "refined_lexical_A"),
        ("RQ9 (top-1 ensemble vs BITEM baseline)",
         "bitem_only_refined_sonnet_A", "ensemble_top1"),
        ("RQ9 (top-3 ensemble vs BITEM baseline)",
         "bitem_only_refined_sonnet_A", "ensemble_top3"),
        ("RQ10 (s1-Haiku vs all-Sonnet)",
         "bitem_only_refined_sonnet_A", "bitem_only_s1_haiku_A"),
        ("RQ10 (s4-Haiku vs all-Sonnet)",
         "bitem_only_refined_sonnet_A", "bitem_only_s4_haiku_A"),
        ("RQ10 (all-Haiku vs all-Sonnet)",
         "bitem_only_refined_sonnet_A", "bitem_only_all_haiku_A"),
        ("RQ7-improvement #7 (Opus pipeline vs Sonnet)",
         "bitem_only_refined_sonnet_A", "bitem_only_opus_A"),
        ("CASE (retrank+E404+sonnet vs broad)",
         "retrank_plus_error404_broad_A", "retrank_plus_error404_refined_sonnet_A"),
    ]

    # Need the broad version for the case study
    case_extra = ("retrank_plus_error404_broad_A", "retrank_plus_error404_broad_A.json")
    if case_extra[0] not in cached:
        path = EVAL_DIR / case_extra[1]
        if path.exists():
            cached[case_extra[0]] = per_topic_results(path)

    print(f"\n{'─'*80}")
    print(f"Paired bootstrap 95% CI for Δ HMR ({args.n_iter} iterations)")
    print(f"{'─'*80}")
    print(f"{'Comparison':<48} {'Δ':>9}  {'95% CI':>17}  {'p≈0?':>5}")

    pair_results: dict[str, dict] = {}
    for label, a, b in comparisons:
        if a not in cached or b not in cached:
            print(f"  SKIP: {label} ({a} or {b} missing)")
            continue
        delta, lo, hi = paired_bootstrap_delta(cached[a], cached[b], n_iter=args.n_iter)
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        marker = "★" if sig == "YES" else " "
        print(f"  {marker} {label:<46} {delta:>+8.4f}  [{lo:>+5.3f}, {hi:>+5.3f}]  {sig:>5}")
        pair_results[label] = {"delta": delta, "ci_lo": lo, "ci_hi": hi,
                                "significant": sig == "YES",
                                "comparison": (a, b)}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "n_iter": args.n_iter,
        "per_candidate": per_cand,
        "pairwise": pair_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
