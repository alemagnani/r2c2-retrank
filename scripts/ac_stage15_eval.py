#!/usr/bin/env python3
"""
Stage 1.5 — Evaluate refined pool quality vs broad pool.

Two cheap evals:

  E1.5-1 — Pool-overlap ratio: |refined ∩ broad_top_k| / |refined|.
           Low overlap means refinement is doing real work; high overlap means
           the answer doesn't change passage ranking much (probably an "easy" topic
           where the question alone already retrieved the right passages).

  E1.5-2 — Top-3 oracle-entity lift: count topics where the top-3 refined
           passages contain the oracle's key_entities, vs top-3 of the broad pool.
           Tests whether refinement surfaces the answer-bearing doc more often.

Both evals are zero-cost.

Usage:
    python scripts/ac_stage15_eval.py \\
        --refined data/processed/ac_pool_tier1_refined_ce.json \\
        --broad data/processed/ac_pool_tier1.json \\
        --oracle data/eval/ac/oracle_answers.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def passage_set(passages: list[dict], k: int) -> set[tuple[str, str]]:
    """Identifies passages by (doc_id, text-prefix) for stable comparison."""
    return {(p["doc_id"], p["text"][:120]) for p in passages[:k]}


def has_oracle_entity(passages: list[dict], oracle_entry: dict, k: int) -> bool:
    entities = oracle_entry.get("key_entities") or []
    if not entities and oracle_entry.get("answer"):
        entities = [oracle_entry["answer"]]
    if not entities:
        return False
    for p in passages[:k]:
        nt = normalise(p["text"])
        if any(e and normalise(e) in nt for e in entities):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined", required=True)
    parser.add_argument("--broad", required=True)
    parser.add_argument("--oracle", default=str(BASE / "data/eval/ac/oracle_answers.json"))
    parser.add_argument("--top-k-overlap", type=int, default=30,
                        help="K to use when comparing top-K refined vs top-K broad")
    parser.add_argument("--top-k-oracle", type=int, default=3,
                        help="K for E1.5-2 oracle-lift check (top-3 by default)")
    args = parser.parse_args()

    refined_data = json.loads(Path(args.refined).read_text())
    refined_meta = refined_data.get("_meta", {})
    refined = refined_data["topics"]

    broad_data = json.loads(Path(args.broad).read_text())
    broad = broad_data["topics"]

    oracle = json.loads(Path(args.oracle).read_text())

    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  Stage 1.5 Refined-Pool Quality                                    ║")
    print(f"║  Refined:  {Path(args.refined).name:<53} ║")
    print(f"║  Broad:    {Path(args.broad).name:<53} ║")
    print(f"║  Algo:     {refined_meta.get('algo', '?'):<53} ║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")

    # ─── E1.5-1: pool-overlap ─────────────────────────────────────────────
    overlaps = []
    skipped = 0
    for qid, r_entry in refined.items():
        if not r_entry.get("refined"):
            skipped += 1
            continue
        ref_set = passage_set(r_entry["passages"], args.top_k_overlap)
        broad_set = passage_set(broad.get(qid, []), args.top_k_overlap)
        if not ref_set:
            continue
        overlap = len(ref_set & broad_set) / len(ref_set)
        overlaps.append((qid, overlap))

    if overlaps:
        ratios = [o for _, o in overlaps]
        print(f"\n─── E1.5-1 Pool-overlap ──────────────────────────────────────")
        print(f"  Refinement done on:        {len(overlaps)}/{len(refined)} topics  ({skipped} skipped due to refusal)")
        print(f"  Mean overlap (refined∩broad)/refined: {sum(ratios)/len(ratios)*100:.1f}%")
        sorted_o = sorted(ratios)
        n = len(sorted_o)
        print(f"  median: {sorted_o[n//2]*100:.1f}%   p25: {sorted_o[n//4]*100:.1f}%   p75: {sorted_o[3*n//4]*100:.1f}%")
        # Show topics where refinement reordered the most (low overlap) and least
        sorted_topics = sorted(overlaps, key=lambda x: x[1])
        print(f"\n  Top 5 most-reordered (lowest overlap → refinement made big difference):")
        for qid, o in sorted_topics[:5]:
            print(f"    {qid}: {o*100:.1f}% overlap")
        print(f"  Top 5 least-reordered (highest overlap → answer didn't change ranking):")
        for qid, o in sorted_topics[-5:]:
            print(f"    {qid}: {o*100:.1f}% overlap")

    # ─── E1.5-2: top-3 oracle lift ────────────────────────────────────────
    print(f"\n─── E1.5-2 Top-{args.top_k_oracle} oracle-entity lift ───────────────────────────")
    refined_hits = 0
    broad_hits = 0
    only_refined = []
    only_broad = []
    n_assessable = 0
    for qid, r_entry in refined.items():
        o = oracle.get(qid)
        if not o or not o.get("answer"):
            continue  # no oracle reference
        n_assessable += 1
        ref_passages = r_entry["passages"] if r_entry.get("refined") else broad.get(qid, [])
        broad_passages = broad.get(qid, [])
        ref_hit = has_oracle_entity(ref_passages, o, args.top_k_oracle)
        broad_hit = has_oracle_entity(broad_passages, o, args.top_k_oracle)
        if ref_hit:
            refined_hits += 1
        if broad_hit:
            broad_hits += 1
        if ref_hit and not broad_hit:
            only_refined.append(qid)
        elif broad_hit and not ref_hit:
            only_broad.append(qid)

    print(f"  Topics assessable: {n_assessable}")
    print(f"  Top-{args.top_k_oracle} contains oracle entity:")
    print(f"    refined: {refined_hits}/{n_assessable} ({refined_hits/max(n_assessable,1)*100:.1f}%)")
    print(f"    broad:   {broad_hits}/{n_assessable} ({broad_hits/max(n_assessable,1)*100:.1f}%)")
    delta = refined_hits - broad_hits
    print(f"  Lift (refined - broad): {delta:+d} topics")
    if only_refined:
        print(f"\n  Topics where refinement helped (only refined surfaces oracle in top-{args.top_k_oracle}):")
        for qid in only_refined[:10]:
            print(f"    {qid}: oracle answer = {(oracle[qid].get('answer','') or '')[:60]}")
    if only_broad:
        print(f"\n  Topics where refinement HURT (broad had it, refined doesn't):")
        for qid in only_broad[:10]:
            print(f"    {qid}: oracle answer = {(oracle[qid].get('answer','') or '')[:60]}")

    # Summary
    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  SUMMARY ({refined_meta.get('algo', '?'):>10}):                                       ║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    if overlaps:
        m = sum(r for _, r in overlaps) / len(overlaps)
        print(f"║  E1.5-1 Mean pool-overlap:       {m*100:>5.1f}%                          ║")
    print(f"║  E1.5-2 Top-{args.top_k_oracle} oracle hit (refined):  {refined_hits}/{n_assessable} ({refined_hits/max(n_assessable,1)*100:.1f}%)              ║")
    print(f"║  E1.5-2 Top-{args.top_k_oracle} oracle hit (broad):    {broad_hits}/{n_assessable} ({broad_hits/max(n_assessable,1)*100:.1f}%)              ║")
    print(f"║  E1.5-2 Lift (refined - broad):  {delta:+d}                                ║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")

    output = {
        "algo": refined_meta.get("algo"),
        "e15_1_mean_overlap": sum(r for _, r in overlaps) / max(len(overlaps), 1),
        "e15_1_overlaps": dict(overlaps),
        "e15_2_refined_hits": refined_hits,
        "e15_2_broad_hits": broad_hits,
        "e15_2_n_assessable": n_assessable,
        "e15_2_lift": delta,
        "e15_2_only_refined": only_refined,
        "e15_2_only_broad": only_broad,
    }
    out_path = Path(args.refined).with_suffix(".eval.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
