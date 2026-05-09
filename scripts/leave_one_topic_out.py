#!/usr/bin/env python3
"""Leave-one-topic-out HMR robustness check.

For each pair of candidate runs, recompute Δ HMR after dropping each
topic in turn. If the sign of Δ flips for many leave-out sets, the
ranking is fragile.

Reports for each headline comparison: how many of 65 leave-outs
preserve the sign of Δ HMR (i.e., the "winner" remains the winner).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

EVAL = BASE / "data" / "eval" / "ac_runs"


def load_results(path: Path) -> list[QuestionResult]:
    d = json.loads(path.read_text())
    out = []
    for qid, q in d["per_question"].items():
        out.append(QuestionResult(
            question_id=qid,
            correct=bool(q.get("correct")),
            confidence=float(q.get("confidence", 0)),
            nuggets_returned=int(q.get("n_returned", 0)),
            nuggets_relevant=int(q.get("n_relevant", 0)),
        ))
    return out


def hmr_full(results: list[QuestionResult]) -> float:
    return compute_metrics(results).HMR


def loo_stability(a_path: Path, b_path: Path) -> dict:
    a = load_results(a_path)
    b = load_results(b_path)
    qids = sorted({r.question_id for r in a} & {r.question_id for r in b})
    a_by = {r.question_id: r for r in a if r.question_id in qids}
    b_by = {r.question_id: r for r in b if r.question_id in qids}
    full_delta = hmr_full([a_by[q] for q in qids]) - hmr_full([b_by[q] for q in qids])
    sign_full = 1 if full_delta > 0 else (-1 if full_delta < 0 else 0)
    flips = 0
    deltas = []
    for skip in qids:
        keep = [q for q in qids if q != skip]
        d = hmr_full([a_by[q] for q in keep]) - hmr_full([b_by[q] for q in keep])
        deltas.append(d)
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        if s != sign_full:
            flips += 1
    return {
        "full_delta": full_delta,
        "preserve_sign": len(qids) - flips,
        "n": len(qids),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def main():
    pairs = [
        ("bitem_only_refined_sonnet_A", "bitem_only_broad_A",
         "BITEM Sonnet refinement vs BITEM broad"),
        ("bitem_only_refined_sonnet_A", "refined_sonnet_A",
         "BITEM-only vs Tier-1 (both Sonnet refined)"),
        ("bitem_only_refined_sonnet_A", "broad_A",
         "BITEM-Sonnet vs Tier-1 broad"),
        ("ensemble_top1", "bitem_only_refined_sonnet_A",
         "ensemble_top1 vs BITEM-Sonnet (calibration artifact)"),
        ("bitem_only_refined_sonnet_A", "bitem_only_s1_haiku_A",
         "BITEM-Sonnet vs S1=Haiku"),
        ("bitem_only_refined_sonnet_A", "bitem_only_s4_haiku_A",
         "BITEM-Sonnet vs S4=Haiku"),
        ("bitem_only_refined_sonnet_A", "bitem_only_all_haiku_A",
         "BITEM-Sonnet vs all-Haiku"),
        ("bitem_only_refined_sonnet_A", "bitem_only_opus_A",
         "BITEM-Sonnet vs BITEM-Opus pipeline"),
        ("bitem_only_refined_sonnet_A", "error404_only_refined_sonnet_A",
         "BITEM-only vs Error404-only"),
    ]

    print(f"\n{'─'*108}")
    print(f"Leave-one-topic-out HMR ranking stability")
    print(f"{'─'*108}\n")
    print(f"{'Comparison':<54} {'Δ HMR':>8} {'min':>8} {'max':>8} {'Sign-stable':>13}")
    print("─" * 108)

    out = []
    for a, b, label in pairs:
        ap = EVAL / f"{a}.json"; bp = EVAL / f"{b}.json"
        if not ap.exists() or not bp.exists():
            print(f"  [skip] {label}"); continue
        r = loo_stability(ap, bp)
        flag = "✓" if r["preserve_sign"] == r["n"] else \
               (" " if r["preserve_sign"] >= 0.95 * r["n"] else "✗")
        print(f"  {flag} {label:<52} {r['full_delta']:>+8.4f} "
              f"{r['min_delta']:>+8.4f} {r['max_delta']:>+8.4f} "
              f"{r['preserve_sign']}/{r['n']}")
        out.append({"label": label, **r})

    print(f"\n  ✓ = sign of Δ preserved on every leave-out (fully robust)")
    print(f"  blank = stable on >= 95% of leave-outs")
    print(f"  ✗ = sign flips on > 5% of leave-outs (fragile)\n")

    Path(EVAL / "loo_stability.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"  saved {EVAL / 'loo_stability.json'}")


if __name__ == "__main__":
    main()
