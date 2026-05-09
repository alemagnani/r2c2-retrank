#!/usr/bin/env python3
"""Judge-specific decomposition: recompute Acc, Acc|ans, Refuse,
Cnf-W, R_O under Opus judge for the candidates we have both files for.

Tests whether the R_O-bottleneck story is judge-invariant.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"


def is_refusal(q: dict) -> bool:
    a = (q.get("answer") or "").strip().lower()
    if not a or a in {"i don't know", "i dont know", "unknown", ""}:
        return True
    return float(q.get("confidence", 0.0)) <= 0.10


def decompose(p: Path) -> dict:
    d = json.loads(p.read_text())
    qs = list(d["per_question"].values())
    n = len(qs)
    refusals = [q for q in qs if is_refusal(q)]
    answered = [q for q in qs if not is_refusal(q)]
    n_correct = sum(1 for q in qs if q.get("correct"))
    n_correct_ans = sum(1 for q in answered if q.get("correct"))
    cnf_w = sum(1 for q in answered if not q.get("correct")
                and float(q.get("confidence", 0)) >= 0.5)
    return {
        "Acc": n_correct / n if n else 0.0,
        "Acc|ans": n_correct_ans / len(answered) if answered else 0.0,
        "Refuse%": 100 * len(refusals) / n if n else 0.0,
        "Cnf-W": cnf_w,
        "R_O": d["metrics"]["R_O"],
        "HMR": d["metrics"]["HMR"],
    }


def main():
    candidates = [
        ("ensemble_top1", "ensemble_top1"),
        ("ensemble_top3", "ensemble_top3"),
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet (main)"),
        ("bitem_only_broad_A", "BITEM-broad"),
        ("bitem_only_opus_A", "BITEM-Opus pipeline"),
        ("refined_sonnet_A", "Tier-1 Sonnet"),
        ("broad_A", "Tier-1 broad"),
        ("retrank_plus_error404_refined_sonnet_A", "retrank+E404+Sonnet"),
    ]

    print(f"\n{'─'*108}")
    print(f"Judge-specific decomposition: Sonnet vs Opus judge per top candidate")
    print(f"{'─'*108}\n")
    print(f"{'Candidate':<28} {'Judge':<8} {'Acc':>5} {'Acc|ans':>8} {'Refuse%':>8} "
          f"{'Cnf-W':>5} {'R_O':>5} {'HMR':>5}")
    print("─" * 108)

    out: list[dict] = []
    for stem, label in candidates:
        s = EVAL / f"{stem}.json"
        o = EVAL / f"{stem}_opus_judge.json"
        if not s.exists() or not o.exists():
            continue
        ds = decompose(s)
        do = decompose(o)
        print(f"  {label:<26} {'Sonnet':<8} {ds['Acc']:>5.3f} {ds['Acc|ans']:>8.3f} "
              f"{ds['Refuse%']:>7.1f}% {ds['Cnf-W']:>5} {ds['R_O']:>5.3f} {ds['HMR']:>5.3f}")
        print(f"  {'':<26} {'Opus':<8} {do['Acc']:>5.3f} {do['Acc|ans']:>8.3f} "
              f"{do['Refuse%']:>7.1f}% {do['Cnf-W']:>5} {do['R_O']:>5.3f} {do['HMR']:>5.3f}")
        d_cnf_w = do["Cnf-W"] - ds["Cnf-W"]
        d_acc_ans = do["Acc|ans"] - ds["Acc|ans"]
        flag = " " if abs(d_cnf_w) <= 1 and abs(d_acc_ans) <= 0.05 else "*"
        print(f"  {flag} {'':<24} {'Δ':<8} {do['Acc']-ds['Acc']:>+5.3f} "
              f"{d_acc_ans:>+8.3f} {do['Refuse%']-ds['Refuse%']:>+7.1f}% "
              f"{d_cnf_w:>+5d} {do['R_O']-ds['R_O']:>+5.3f} "
              f"{do['HMR']-ds['HMR']:>+5.3f}")
        print()
        out.append({
            "candidate": label,
            "sonnet": ds, "opus": do,
            "delta_cnf_w": d_cnf_w,
            "delta_acc_ans": d_acc_ans,
        })

    print("─" * 108)
    print(f"  '*' marks candidates where judge choice changes the decomposition non-trivially")
    print(f"     (|ΔCnf-W| > 1 OR |ΔAcc|ans| > 0.05).\n")

    # Cnf-W invariance summary
    invariants = [r for r in out if abs(r["delta_cnf_w"]) <= 1
                  and abs(r["delta_acc_ans"]) <= 0.05]
    print(f"  R_O bottleneck story is judge-invariant on {len(invariants)}/{len(out)} candidates.")

    Path(EVAL / "judge_specific_decomposition.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"  saved {EVAL / 'judge_specific_decomposition.json'}")


if __name__ == "__main__":
    main()
