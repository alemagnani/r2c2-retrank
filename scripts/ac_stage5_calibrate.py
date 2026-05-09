#!/usr/bin/env python3
"""
Stage 5 — Confidence calibration & refusal.

Variant A (refusal-aware self-report):
  - If no entailed nuggets OR candidate answer empty → refuse (answer="", conf=5)
  - If verifier match_score == 0  → cap confidence at 25
  - If verifier match_score == 1  → cap confidence at 60
  - If verifier match_score == 2  → use c_self unchanged
  - confidence is then clipped to [5, 100] (no zero — avoids HMR edge cases)

Variant B (ensemble): requires sampled re-runs of stages 1-4.
  Implemented separately — needs `--sampled-stage4` argument.

Output: ready-to-format records with (answer, confidence, nuggets) per topic.

Usage:
    python scripts/ac_stage5_calibrate.py \\
        --stage4 data/processed/ac_stage4_tier1_broad.json \\
        --variant A \\
        --output data/processed/ac_stage5_tier1_broad_A.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def variant_a(rec: dict) -> tuple[str, int, str]:
    """Returns (final_answer, final_confidence, reason)."""
    if rec.get("refused"):
        return "", 5, "stage1 refused — no answer to give"
    nuggets = rec.get("filtered_nuggets", [])
    cand_ans = (rec.get("answer") or "").strip()
    if not nuggets or not cand_ans:
        return "", 5, "no entailed nuggets after Stage 3 filter"
    c_self = int(rec.get("confidence_a") or rec.get("confidence") or 0)
    ms = int(rec.get("match_score") or 0)
    if ms == 0:
        cap = 25
        reason = "verifier disagrees with candidate"
    elif ms == 1:
        cap = 60
        reason = "verifier partially agrees"
    else:
        cap = 100
        reason = "verifier confirms candidate"
    final_conf = max(5, min(c_self, cap))
    return cand_ans, final_conf, reason


def variant_b(rec: dict, sampled_rec: dict | None) -> tuple[str, int, str]:
    """Ensemble agreement: confidence from agreement between deterministic and
    sampled verifications.

    Each "vote" = (Stage 4 verifier_answer, derived from filtered nuggets). We
    have 2 votes here: deterministic (Stage 1 temp=0) and sampled (Stage 1 temp=0.7).
    """
    if rec.get("refused"):
        return "", 5, "stage1 refused"
    if not rec.get("filtered_nuggets"):
        return "", 5, "no entailed nuggets"
    cand_ans = (rec.get("answer") or "").strip()
    if not cand_ans:
        return "", 5, "no candidate answer"

    votes = []
    # vote 1: candidate (Stage 1)
    votes.append(cand_ans)
    # vote 2: verifier (Stage 4)
    v_ans = (rec.get("verifier_answer") or "").strip()
    if v_ans:
        votes.append(v_ans)
    # vote 3 (if available): sampled candidate
    if sampled_rec is not None:
        s_ans = (sampled_rec.get("answer") or "").strip()
        if s_ans:
            votes.append(s_ans)

    # Agreement = fraction of votes matching candidate
    norm = lambda s: s.lower().strip()
    nc = norm(cand_ans)
    n_agree = sum(1 for v in votes if norm(v) == nc or nc in norm(v) or norm(v) in nc)
    agreement = n_agree / len(votes)

    if agreement < 0.4:
        return "", 5, f"ensemble disagreement ({n_agree}/{len(votes)} agree)"

    confidence = round(100 * agreement)
    confidence = max(5, min(100, confidence))
    return cand_ans, confidence, f"ensemble agreement {n_agree}/{len(votes)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4", required=True)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--sampled-stage1", default=None,
                        help="(Variant B) Stage 1 output from a sampled rerun")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stage4 = json.loads(Path(args.stage4).read_text())
    sampled_s1 = None
    if args.sampled_stage1:
        sampled_s1 = json.loads(Path(args.sampled_stage1).read_text())["topics"]

    out_topics: dict[str, dict] = {}
    n_refusals = 0
    n_full_conf = 0
    n_capped = 0

    for qid, rec in sorted(stage4["topics"].items()):
        if args.variant == "A":
            ans, conf, reason = variant_a(rec)
        else:
            sampled = sampled_s1.get(qid) if sampled_s1 else None
            ans, conf, reason = variant_b(rec, sampled)
        if not ans:
            n_refusals += 1
        elif conf >= 90:
            n_full_conf += 1
        else:
            n_capped += 1
        out_topics[qid] = {
            "answer": ans,
            "confidence": conf,
            "calibration_reason": reason,
            "candidate_answer": rec.get("answer", ""),
            "candidate_confidence": rec.get("confidence_a", 0),
            "verifier_answer": rec.get("verifier_answer", ""),
            "verifier_confidence": rec.get("verifier_confidence", 0),
            "match_score": rec.get("match_score", 0),
            "n_entailed_nuggets": len(rec.get("filtered_nuggets", [])),
            "filtered_nuggets": rec.get("filtered_nuggets", []),
        }

    output = {
        "_meta": {
            "stage4_file": Path(args.stage4).name,
            "variant": args.variant,
            "n_topics": len(out_topics),
            "n_refusals": n_refusals,
            "n_full_confidence": n_full_conf,
            "n_capped": n_capped,
        },
        "topics": out_topics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    confs = [r["confidence"] for r in out_topics.values()]

    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  Stage 5 — Calibration (Variant {args.variant})                              ║")
    print(f"║  Source: {Path(args.stage4).name:<55}║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    print(f"║  Refusals (conf=5):       {n_refusals:>3}/65{'':<35}║")
    print(f"║  Full conf (≥90):         {n_full_conf:>3}/65{'':<35}║")
    print(f"║  Capped (5<conf<90):      {n_capped:>3}/65{'':<35}║")
    print(f"║  Mean confidence:         {sum(confs)/len(confs):>5.1f}{'':<37}║")
    print(f"║  Median confidence:       {sorted(confs)[len(confs)//2]:>3}{'':<41}║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")
    print(f"\nFull output: {output_path}")


if __name__ == "__main__":
    main()
