#!/usr/bin/env python3
"""
Improvement #1 — Per-topic pool selection.

For each of 65 topics, choose the candidate AC pipeline whose Stage 4 output
gives the strongest signal that its candidate answer is correct. Then use
that pipeline's answer + nuggets for the submission.

Strength score per candidate per topic:
    score = (match_score × 100) + n_entailed_nuggets + (c_self / 10)

Refuses when no candidate has match_score≥1 OR all are refusals.

Inputs: Stage 4 outputs for the 12 base candidates (Tier-1 × {broad,
ce, lexical, sonnet} × Variant A only here, plus 4 ablation pools ×
{broad, sonnet}).

Output: a Stage-5-shaped JSON ready for ac_stage6_format.py.

Usage:
    python scripts/ac_per_topic_select.py \\
        --output data/processed/ac_stage5_per_topic_best.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Same candidate set as ac_ensemble.py (16 base candidates)
CANDIDATES = [
    ("tier1_broad_A",         "ac_stage4_tier1_broad.json"),
    ("tier1_ce_A",            "ac_stage4_tier1_refined_ce.json"),
    ("tier1_lexical_A",       "ac_stage4_tier1_refined_lexical.json"),
    ("tier1_sonnet_A",        "ac_stage4_tier1_refined_sonnet.json"),
    ("bitem_broad_A",         "ac_stage4_bitem_only_broad.json"),
    ("bitem_sonnet_A",        "ac_stage4_bitem_only_refined_sonnet.json"),
    ("retrank_broad_A",       "ac_stage4_retrank_only_broad.json"),
    ("retrank_sonnet_A",      "ac_stage4_retrank_only_refined_sonnet.json"),
    ("rE404_broad_A",         "ac_stage4_retrank_plus_error404_broad.json"),
    ("rE404_sonnet_A",        "ac_stage4_retrank_plus_error404_refined_sonnet.json"),
    ("tier1p2_broad_A",       "ac_stage4_tier1plus2_broad.json"),
    ("tier1p2_sonnet_A",      "ac_stage4_tier1plus2_refined_sonnet.json"),
]

# Pipelines we trust (skip retrank-only because of catastrophic Stage 1 refusals)
TRUSTED = [c for c in CANDIDATES if "retrank_broad" not in c[0] and "retrank_sonnet_A" != c[0]]


def candidate_strength(rec: dict) -> float:
    """Higher = stronger signal that this candidate is correct on this topic."""
    if rec.get("refused"):
        return -1e9
    if not rec.get("filtered_nuggets"):
        return -1e9
    if not (rec.get("answer") or "").strip():
        return -1e9
    ms = int(rec.get("match_score", 0) or 0)
    n_nug = len(rec.get("filtered_nuggets", []))
    c_self = int(rec.get("confidence_a", 0) or 0)
    # match_score is the dominant term
    return ms * 100.0 + n_nug + c_self / 10.0


def variant_a_confidence(c_self: int, ms: int) -> int:
    if ms == 0: cap = 25
    elif ms == 1: cap = 60
    else: cap = 100
    return max(5, min(c_self, cap))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-dir", default=str(BASE / "data/processed"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-trusted-only", action="store_true",
                        help="Skip the retrank-only candidate (catastrophic Stage 1 refusals)")
    parser.add_argument("--max-confidence", action="store_true",
                        help="Set confidence=100 on all selected (à la top-1 ensemble)")
    args = parser.parse_args()

    cand_set = TRUSTED if args.use_trusted_only else CANDIDATES
    print(f"Using {len(cand_set)} candidate pipelines")

    loaded = {}
    for tag, fname in cand_set:
        p = Path(args.stage4_dir) / fname
        if not p.exists():
            print(f"  ⚠ missing: {fname}")
            continue
        loaded[tag] = json.loads(p.read_text())["topics"]
    print(f"Loaded {len(loaded)} candidates")

    # Iterate topics (assume all candidates have the same topic set)
    topics = sorted(set().union(*[set(v.keys()) for v in loaded.values()]))
    out_topics = {}
    selection_counts = {}
    refusals = 0
    n_match_2 = 0
    n_match_1 = 0
    n_match_0 = 0

    for qid in topics:
        # Score all candidates and pick the best
        best_tag = None
        best_score = -1e9
        best_rec = None
        for tag, run in loaded.items():
            rec = run.get(qid, {})
            s = candidate_strength(rec)
            if s > best_score:
                best_score = s
                best_tag = tag
                best_rec = rec

        if best_rec is None or best_score < 0:
            # All refused / no viable answer
            out_topics[qid] = {
                "answer": "", "confidence": 5,
                "calibration_reason": "no candidate produced a viable answer",
                "filtered_nuggets": [],
                "selected_pipeline": "(none)",
            }
            refusals += 1
            continue

        ms = int(best_rec.get("match_score", 0) or 0)
        c_self = int(best_rec.get("confidence_a", 0) or 0)
        if args.max_confidence:
            confidence = 100 if ms >= 1 else 5
        else:
            confidence = variant_a_confidence(c_self, ms)
        if ms == 2: n_match_2 += 1
        elif ms == 1: n_match_1 += 1
        else: n_match_0 += 1

        out_topics[qid] = {
            "answer": best_rec["answer"],
            "confidence": confidence,
            "calibration_reason": f"per-topic best from {best_tag} (ms={ms})",
            "filtered_nuggets": best_rec.get("filtered_nuggets", []),
            "selected_pipeline": best_tag,
            "candidate_answer": best_rec.get("answer"),
            "candidate_confidence": c_self,
            "verifier_answer": best_rec.get("verifier_answer", ""),
            "verifier_confidence": best_rec.get("verifier_confidence", 0),
            "match_score": ms,
            "n_entailed_nuggets": len(best_rec.get("filtered_nuggets", [])),
        }
        selection_counts[best_tag] = selection_counts.get(best_tag, 0) + 1

    output = {
        "_meta": {
            "n_candidates": len(loaded),
            "candidates": list(loaded.keys()),
            "n_topics": len(topics),
            "n_refusals": refusals,
            "n_match_2": n_match_2,
            "n_match_1": n_match_1,
            "n_match_0": n_match_0,
            "max_confidence_mode": args.max_confidence,
            "selection_counts": selection_counts,
        },
        "topics": out_topics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n╔═══════════════════════════════════════════════════════════════════╗")
    print(f"║  Per-Topic Pool Selection                                          ║")
    print(f"╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║  Topics:                {len(topics):<43}║")
    print(f"║  Match=2 selections:    {n_match_2:<43}║")
    print(f"║  Match=1 selections:    {n_match_1:<43}║")
    print(f"║  Match=0 selections:    {n_match_0:<43}║")
    print(f"║  Refusals:              {refusals:<43}║")
    print(f"║  Max-confidence mode:   {str(args.max_confidence):<43}║")
    print(f"╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║  Pipeline selection counts (top-K):                                ║")
    for tag, c in sorted(selection_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"║    {tag:<28} {c:>3}{'':<32}║")
    print(f"╚═══════════════════════════════════════════════════════════════════╝")
    print(f"\nFull output: {args.output}")


if __name__ == "__main__":
    main()
