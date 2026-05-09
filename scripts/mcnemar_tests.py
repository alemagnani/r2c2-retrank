#!/usr/bin/env python3
"""McNemar's exact test on per-topic correctness for headline comparisons.

McNemar tests whether two paired binary classifiers (A correct vs B correct
on the same items) differ. It only uses the discordant cells (A-correct/B-wrong
and A-wrong/B-correct), so it is more powerful than bootstrap on Δ-HMR
when only a few topics flip between systems.

Two-sided exact binomial p-value on min(b, c) given b+c.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"


def correctness(p: Path) -> dict[str, bool]:
    d = json.loads(p.read_text())
    return {qid: bool(q.get("correct")) for qid, q in d["per_question"].items()}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided binomial test on min(b,c) successes in n=b+c trials, p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    one_tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * one_tail)


def run(name_a: str, name_b: str, label: str) -> tuple[int, int, int, int, float]:
    a_path = EVAL / f"{name_a}.json"
    b_path = EVAL / f"{name_b}.json"
    if not (a_path.exists() and b_path.exists()):
        print(f"  [skip] {label}: missing {name_a} or {name_b}")
        return (0, 0, 0, 0, 1.0)
    a = correctness(a_path)
    b = correctness(b_path)
    qids = sorted(set(a) & set(b))
    n11 = sum(1 for q in qids if a[q] and b[q])      # both correct
    n10 = sum(1 for q in qids if a[q] and not b[q])  # A correct only
    n01 = sum(1 for q in qids if not a[q] and b[q])  # B correct only
    n00 = sum(1 for q in qids if not a[q] and not b[q])  # both wrong
    p = mcnemar_exact_p(n10, n01)
    flag = "*" if p < 0.05 else " "
    print(f"  {flag} {label:<48} A-only={n10:>2}  B-only={n01:>2}  "
          f"both-correct={n11}  both-wrong={n00}  p={p:.4f}")
    return (n11, n10, n01, n00, p)


def main():
    print(f"\n{'─'*92}")
    print(f"McNemar exact test on per-topic correctness (paired)")
    print(f"{'─'*92}\n")
    print(f"  Significance flag '*' = p<0.05.  A-only/B-only = discordant cells.\n")

    comparisons = [
        # (A, B, "A vs B")
        ("ensemble_top1", "bitem_only_refined_sonnet_A",
         "ensemble_top1 vs BITEM-Sonnet (main)"),
        ("ensemble_top3", "bitem_only_refined_sonnet_A",
         "ensemble_top3 vs BITEM-Sonnet"),
        ("bitem_only_refined_sonnet_A", "bitem_only_broad_A",
         "BITEM-Sonnet vs BITEM-broad (refinement)"),
        ("bitem_only_refined_sonnet_A", "refined_sonnet_A",
         "BITEM-Sonnet vs Tier-1 Sonnet (pool)"),
        ("bitem_only_refined_sonnet_A", "broad_A",
         "BITEM-Sonnet vs Tier-1 broad (pool×refinement)"),
        ("bitem_only_refined_sonnet_A", "bitem_only_opus_A",
         "BITEM-Sonnet vs BITEM-Opus pipeline"),
        ("bitem_only_refined_sonnet_A", "bitem_only_s1_haiku_A",
         "BITEM-Sonnet vs BITEM-S1=Haiku"),
        ("bitem_only_refined_sonnet_A", "bitem_only_s4_haiku_A",
         "BITEM-Sonnet vs BITEM-S4=Haiku"),
        ("bitem_only_refined_sonnet_A", "bitem_only_all_haiku_A",
         "BITEM-Sonnet vs BITEM-all-Haiku"),
        ("refined_sonnet_A", "broad_A",
         "Tier-1: Sonnet refinement vs broad"),
        ("bitem_only_refined_sonnet_A", "error404_only_refined_sonnet_A",
         "BITEM-only vs Error404-only (single-team pool)"),
        ("bitem_only_refined_sonnet_A", "waterlooclarke_only_refined_sonnet_A",
         "BITEM-only vs WaterlooClarke-only"),
    ]

    rows = []
    for a, b, label in comparisons:
        r = run(a, b, label)
        rows.append((label, *r))

    out = [{"label": r[0], "n_both_correct": r[1], "A_only": r[2],
            "B_only": r[3], "n_both_wrong": r[4], "p": r[5]}
           for r in rows]
    Path(EVAL / "mcnemar_tests.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"\n  saved {EVAL / 'mcnemar_tests.json'}")


if __name__ == "__main__":
    main()
