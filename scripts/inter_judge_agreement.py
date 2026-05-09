#!/usr/bin/env python3
"""Per-topic inter-judge agreement on Stage A (bogus) and Stage B (correct).

Measures Cohen's kappa-style agreement to address: "Does Sonnet judge agree
with Opus judge on the underlying verdicts, or do they reach the same HMR
by different routes?"
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE / "data" / "eval" / "ac_runs"


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n
    p_a_pos = sum(a) / n
    p_b_pos = sum(b) / n
    p_e = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def per_topic_correctness(eval_path: Path) -> dict[str, bool]:
    d = json.loads(eval_path.read_text())
    return {qid: bool(q.get("correct", False))
            for qid, q in d["per_question"].items()}


def main():
    candidates = [
        ("ensemble_top1", "ensemble top-1 (best)"),
        ("ensemble_top3", "ensemble top-3"),
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet baseline"),
        ("bitem_only_broad_A", "BITEM-broad"),
        ("bitem_only_opus_A", "BITEM-Opus pipeline"),
        ("refined_sonnet_A", "Tier-1 refined-Sonnet"),
        ("retrank_plus_error404_refined_sonnet_A", "retrank+E404+Sonnet (case study)"),
        ("refined_ce_A", "Tier-1 refined-CE"),
        ("broad_A", "Tier-1 broad"),
    ]

    print(f"\n{'─'*88}")
    print(f"Per-topic Stage B agreement: Sonnet judge vs Opus judge")
    print(f"{'─'*88}\n")
    print(f"{'Candidate':<48} {'Agree':>7} {'κ':>7} {'S+/O−':>6} {'S−/O+':>6}")
    print("─" * 88)

    total_kappa = []
    for name, label in candidates:
        s = EVAL_DIR / f"{name}.json"
        o = EVAL_DIR / f"{name}_opus_judge.json"
        if not (s.exists() and o.exists()):
            continue
        sm = per_topic_correctness(s)
        om = per_topic_correctness(o)
        common = sorted(set(sm) & set(om))
        s_v = [sm[q] for q in common]
        o_v = [om[q] for q in common]
        n = len(common)
        agree = sum(1 for x, y in zip(s_v, o_v) if x == y)
        kappa = cohen_kappa(s_v, o_v)
        s_pos_o_neg = sum(1 for x, y in zip(s_v, o_v) if x and not y)
        s_neg_o_pos = sum(1 for x, y in zip(s_v, o_v) if not x and y)
        total_kappa.append(kappa)
        print(f"  {label:<48} {agree}/{n} ({agree/n*100:.0f}%)  "
              f"{kappa:>6.3f}  {s_pos_o_neg:>5}  {s_neg_o_pos:>5}")

    if total_kappa:
        avg = sum(total_kappa) / len(total_kappa)
        print(f"\n  Mean Cohen's κ (Stage B correctness): {avg:.3f}")
        if avg > 0.81:
            print("  → 'Almost perfect' agreement (Landis-Koch)")
        elif avg > 0.61:
            print("  → 'Substantial' agreement (Landis-Koch)")
        elif avg > 0.41:
            print("  → 'Moderate' agreement (Landis-Koch)")
        else:
            print("  → Agreement below 'substantial'")


if __name__ == "__main__":
    main()
