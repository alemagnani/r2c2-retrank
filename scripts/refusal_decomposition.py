#!/usr/bin/env python3
"""Decompose HMR/accuracy into:
   - overall accuracy (gains from refusal vs from better answers?)
   - answered-only accuracy
   - refusal rate
   - confident-wrong count (conf >= 0.5 AND incorrect)
   - bogus rate among returned nuggets

Refusal here = empty answer string OR confidence <= 0.10.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"

REFUSAL_CONF = 0.10


def is_refusal(q: dict) -> bool:
    a = (q.get("answer") or "").strip().lower()
    if not a or a in {"i don't know", "i dont know", "unknown", ""}:
        return True
    return float(q.get("confidence", 0.0)) <= REFUSAL_CONF


def decompose(path: Path) -> dict:
    d = json.loads(path.read_text())
    qs = list(d["per_question"].values())
    n = len(qs)
    refusals = [q for q in qs if is_refusal(q)]
    answered = [q for q in qs if not is_refusal(q)]
    correct_all = sum(1 for q in qs if q.get("correct"))
    correct_answered = sum(1 for q in answered if q.get("correct"))
    confident_wrong = sum(
        1 for q in answered
        if not q.get("correct") and float(q.get("confidence", 0)) >= 0.5
    )
    n_returned = sum(q.get("n_returned", 0) for q in qs)
    n_entailed = sum(q.get("n_entailed", 0) for q in qs)
    bogus_rate = (n_returned - n_entailed) / n_returned if n_returned else 0.0
    m = d["metrics"]
    return {
        "n": n,
        "accuracy_all": correct_all / n,
        "refusal_rate": len(refusals) / n,
        "answered_n": len(answered),
        "accuracy_answered": correct_answered / len(answered) if answered else 0.0,
        "confident_wrong": confident_wrong,
        "bogus_rate": bogus_rate,
        "HMR": m["HMR"],
        "R_O": m["R_O"],
        "R_U": m["R_U"],
        "MNP": m["mean_nugget_precision"],
    }


def main():
    candidates = [
        ("ensemble_top1", "ensemble_top1 (best)"),
        ("ensemble_top3", "ensemble_top3"),
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet"),
        ("bitem_only_opus_A", "BITEM-Opus pipeline"),
        ("bitem_only_broad_A", "BITEM-broad"),
        ("refined_sonnet_A", "Tier-1 refined-Sonnet"),
        ("broad_A", "Tier-1 broad"),
        ("error404_only_refined_sonnet_A", "Error404-only Sonnet"),
        ("waterlooclarke_only_refined_sonnet_A", "Waterloo-only Sonnet"),
        ("retrank_only_refined_sonnet_A", "retrank-only Sonnet"),
    ]

    print(f"\n{'─'*100}")
    print(f"{'Candidate':<28} {'Acc':>5} {'Acc|ans':>7} {'Ref%':>5} {'Cnf-W':>5} "
          f"{'Bogus':>6} {'R_O':>5} {'R_U':>5} {'HMR':>5}")
    print("─" * 100)
    rows = []
    for stem, label in candidates:
        p = EVAL / f"{stem}.json"
        if not p.exists():
            continue
        r = decompose(p)
        rows.append((label, r))
        print(f"  {label:<26} {r['accuracy_all']:>5.3f} {r['accuracy_answered']:>7.3f} "
              f"{r['refusal_rate']*100:>4.0f}% {r['confident_wrong']:>5} "
              f"{r['bogus_rate']*100:>5.0f}% {r['R_O']:>5.3f} {r['R_U']:>5.3f} "
              f"{r['HMR']:>5.3f}")

    out = {label: r for label, r in rows}
    Path(EVAL / "refusal_decomposition.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"\n  Saved to {EVAL / 'refusal_decomposition.json'}")


if __name__ == "__main__":
    main()
