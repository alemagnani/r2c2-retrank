#!/usr/bin/env python3
"""
RQ7 — Abstention threshold sweep.

Hypothesis: there is an abstention rate that maximises HMR. Sweep refusal
thresholds over the cached Stage 4 + Stage B outputs and recompute metrics
with no new LLM calls.

Inputs (per candidate AC run):
  - Stage 4 output (has candidate_answer, c_self, verifier_answer, match_score,
    verifier_confidence, n_entailed_nuggets per topic)
  - Self-eval output from scripts/ac_eval.py (has per-question correctness +
    relevant_nugget count)

For each rule, build a QuestionResult list:
  - refused topic    → correct=False, conf=0.05, returned=0, relevant=0
  - non-refused      → use cached eval values; conf computed by Variant A
                       (cap by match_score, then divide by 100)

Output: a CSV-like table with one row per (candidate × rule) combination.

Usage:
    python scripts/ac_abstention_sweep.py \\
        --output data/eval/ac_runs/abstention_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402


# ─── Abstention rules ─────────────────────────────────────────────────────────


def _variant_a_confidence(c_self: int, match_score: int) -> int:
    """Variant A's cap-by-match-score formula. Returns 0-100 int."""
    if match_score == 0:
        cap = 25
    elif match_score == 1:
        cap = 60
    else:
        cap = 100
    return max(5, min(c_self, cap))


# Each rule takes a Stage 4 record and returns True if topic should be refused.
RULES: dict[str, callable] = {
    "R0_baseline_VariantA": lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                                        or not r.get("answer", "").strip()),
    "R1_refuse_match_score_0": lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                                           or r.get("match_score", 0) == 0),
    "R2_refuse_match_score_lt_2": lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                                              or r.get("match_score", 0) < 2),
    "R3_refuse_c_self_lt_50": lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                                          or int(r.get("confidence_a", 0) or 0) < 50),
    "R4_refuse_c_self_lt_70": lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                                          or int(r.get("confidence_a", 0) or 0) < 70),
    "R5_refuse_c_self_lt_80_AND_ms_lt_2":
        lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                   or (int(r.get("confidence_a", 0) or 0) < 80 and r.get("match_score", 0) < 2)),
    "R6_refuse_few_nuggets_lt_2": lambda r: (r.get("refused") or len(r.get("filtered_nuggets", [])) < 2),
    "R7_refuse_verifier_low_conf_50": lambda r: (r.get("refused") or not r.get("filtered_nuggets")
                                                  or int(r.get("verifier_confidence", 0) or 0) < 50),
}


def evaluate_rule(stage4: dict, eval_data: dict, rule_name: str) -> dict:
    """Apply a rule and compute metrics. Returns dict with metrics + abstention stats."""
    rule = RULES[rule_name]
    eval_topics = eval_data.get("per_question", {})
    s4_topics = stage4["topics"]

    results: list[QuestionResult] = []
    n_refused_total = 0
    n_refused_via_rule = 0  # excluding stage1 refusals
    answered = 0

    for qid, s4 in sorted(s4_topics.items()):
        # Determine if refused under this rule
        refuse = rule(s4)
        # Track distinction: stage1 refusals vs added by rule
        baseline_refuse = (s4.get("refused")
                           or not s4.get("filtered_nuggets")
                           or not s4.get("answer", "").strip())
        if refuse:
            n_refused_total += 1
            if not baseline_refuse:
                n_refused_via_rule += 1
            # refusal record
            results.append(QuestionResult(
                question_id=qid, correct=False, confidence=0.05,
                nuggets_returned=0, nuggets_relevant=0,
            ))
        else:
            # Use cached self-eval correctness; confidence via Variant A formula
            ev = eval_topics.get(qid, {})
            correct = bool(ev.get("correct", False))
            n_ret = int(ev.get("n_returned", 0))
            n_rel = int(ev.get("n_relevant", 0))
            conf = _variant_a_confidence(
                int(s4.get("confidence_a") or 0),
                int(s4.get("match_score") or 0)
            ) / 100.0
            results.append(QuestionResult(
                question_id=qid, correct=correct, confidence=conf,
                nuggets_returned=n_ret, nuggets_relevant=n_rel,
            ))
            answered += 1

    metrics = compute_metrics(results)
    return {
        "rule": rule_name,
        "n_refused_total": n_refused_total,
        "n_refused_via_rule_only": n_refused_via_rule,
        "n_answered": answered,
        "abstention_rate": n_refused_total / len(s4_topics) if s4_topics else 0,
        "metrics": metrics.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-dir", default=str(BASE / "data/processed"))
    parser.add_argument("--eval-dir", default=str(BASE / "data/eval/ac_runs"))
    parser.add_argument("--output", default=str(BASE / "data/eval/ac_runs/abstention_sweep.json"))
    args = parser.parse_args()

    # Tier-1 candidates
    candidates = []
    for variant in ["broad", "refined_ce", "refined_lexical", "refined_sonnet"]:
        s4 = Path(args.stage4_dir) / f"ac_stage4_tier1_{variant}.json"
        ev = Path(args.eval_dir) / f"{variant}_A.json"
        if s4.exists() and ev.exists():
            candidates.append((f"tier1_{variant}", s4, ev))

    # Ablation pool candidates
    for pool in ["bitem_only", "retrank_only", "retrank_plus_error404", "tier1plus2"]:
        for variant in ["broad", "refined_sonnet"]:
            s4 = Path(args.stage4_dir) / f"ac_stage4_{pool}_{variant}.json"
            ev = Path(args.eval_dir) / f"{pool}_{variant}_A.json"
            if s4.exists() and ev.exists():
                candidates.append((f"{pool}_{variant}", s4, ev))

    print(f"Found {len(candidates)} candidate runs to sweep")
    print(f"Rules: {list(RULES.keys())}\n")

    all_results = {}
    for name, s4_path, ev_path in candidates:
        stage4 = json.loads(s4_path.read_text())
        eval_data = json.loads(ev_path.read_text())
        print(f"═══ Candidate: {name} ═══")
        rule_results = {}
        for rule_name in RULES:
            r = evaluate_rule(stage4, eval_data, rule_name)
            rule_results[rule_name] = r
            m = r["metrics"]
            print(f"  {rule_name:<40}  abst={r['abstention_rate']*100:>4.1f}% "
                  f"acc={m['accuracy']:.3f}  MNP={m['mean_nugget_precision']:.3f}  "
                  f"R_O={m['R_O']:.3f}  HMR={m['HMR']:.3f}")
        all_results[name] = rule_results
        # Find best HMR rule for this candidate
        best = max(rule_results.values(), key=lambda r: r["metrics"]["HMR"])
        print(f"  → best: {best['rule']}  HMR={best['metrics']['HMR']:.3f}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nFull sweep saved to {args.output}")

    # Print global leaderboard sorted by HMR
    print(f"\n{'Candidate':<20} {'Rule':<40} {'AbstRate':>10} {'Acc':>6} {'MNP':>6} {'R_O':>6} {'HMR':>6}")
    print("─" * 100)
    flat = []
    for cand, rules in all_results.items():
        for rname, rdat in rules.items():
            m = rdat["metrics"]
            flat.append((cand, rname, rdat["abstention_rate"], m))
    flat.sort(key=lambda x: -x[3]["HMR"])
    for cand, rname, ar, m in flat[:15]:
        print(f"{cand:<20} {rname:<40} {ar*100:>9.1f}% {m['accuracy']:>6.3f} "
              f"{m['mean_nugget_precision']:>6.3f} {m['R_O']:>6.3f} {m['HMR']:>6.3f}")


if __name__ == "__main__":
    main()
