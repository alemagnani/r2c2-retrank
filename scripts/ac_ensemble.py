#!/usr/bin/env python3
"""
RQ9 — Cross-pool answer ensembling.

For each of the 65 topics, collect every candidate AC pipeline's
`(answer, confidence, filtered_nuggets)`, group answers by semantic equivalence,
pick the most-supported answer, output ensemble confidence proportional to
agreement, and select nuggets from the strongest pipeline that agrees.

Tests RQ9: does diversity at the answer level help, even though diversity at
the passage level hurts (RQ2)?

Inputs: all data/processed/ac_stage5_*_{A,B}.json files (16 candidate runs).
Output: a single ensembled AC run JSON (Stage-5 shape) ready for Stage 6
formatting and self-eval.

Usage:
    python scripts/ac_ensemble.py \\
        --output data/processed/ac_stage5_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

# Which candidate runs to ensemble. We use all 12 clearly-tagged candidates
# (4 Tier-1 with A calibration + 4 ablation pools × broad/sonnet), plus the 4
# Tier-1 B-calibration variants for extra diversity. Total: 16.
CANDIDATES_ALL = [
    # Tier-1 × {broad, refined_ce, refined_lexical, refined_sonnet} × {A, B}
    ("tier1_broad_A", "ac_stage5_tier1_broad_A.json"),
    ("tier1_broad_B", "ac_stage5_tier1_broad_B.json"),
    ("tier1_ce_A", "ac_stage5_tier1_refined_ce_A.json"),
    ("tier1_ce_B", "ac_stage5_tier1_refined_ce_B.json"),
    ("tier1_lexical_A", "ac_stage5_tier1_refined_lexical_A.json"),
    ("tier1_lexical_B", "ac_stage5_tier1_refined_lexical_B.json"),
    ("tier1_sonnet_A", "ac_stage5_tier1_refined_sonnet_A.json"),
    ("tier1_sonnet_B", "ac_stage5_tier1_refined_sonnet_B.json"),
    # Ablation pools × {broad, refined_sonnet}, A calibration
    ("bitem_broad_A", "ac_stage5_bitem_only_broad_A.json"),
    ("bitem_sonnet_A", "ac_stage5_bitem_only_refined_sonnet_A.json"),
    ("retrank_broad_A", "ac_stage5_retrank_only_broad_A.json"),
    ("retrank_sonnet_A", "ac_stage5_retrank_only_refined_sonnet_A.json"),
    ("rE404_broad_A", "ac_stage5_retrank_plus_error404_broad_A.json"),
    ("rE404_sonnet_A", "ac_stage5_retrank_plus_error404_refined_sonnet_A.json"),
    ("tier1p2_broad_A", "ac_stage5_tier1plus2_broad_A.json"),
    ("tier1p2_sonnet_A", "ac_stage5_tier1plus2_refined_sonnet_A.json"),
]

# Top-5 pipelines by self-eval HMR (filtering out the noisy/weak ones)
CANDIDATES_TOP5 = [
    ("bitem_sonnet_A", "ac_stage5_bitem_only_refined_sonnet_A.json"),  # HMR 0.963
    ("bitem_broad_A", "ac_stage5_bitem_only_broad_A.json"),            # HMR 0.937
    ("tier1_sonnet_A", "ac_stage5_tier1_refined_sonnet_A.json"),       # HMR 0.911
    ("rE404_broad_A", "ac_stage5_retrank_plus_error404_broad_A.json"), # HMR 0.910
    ("tier1p2_sonnet_A", "ac_stage5_tier1plus2_refined_sonnet_A.json"),# HMR 0.874
]

# Top-K subsets ranked by self-eval HMR
CANDIDATES_RANKED = [
    ("bitem_sonnet_A", "ac_stage5_bitem_only_refined_sonnet_A.json"),  # HMR 0.963
    ("bitem_broad_A", "ac_stage5_bitem_only_broad_A.json"),            # HMR 0.937
    ("tier1_sonnet_A", "ac_stage5_tier1_refined_sonnet_A.json"),       # HMR 0.911
    ("rE404_broad_A", "ac_stage5_retrank_plus_error404_broad_A.json"), # HMR 0.910
    ("tier1p2_sonnet_A", "ac_stage5_tier1plus2_refined_sonnet_A.json"),# HMR 0.874
    ("retrank_broad_A", "ac_stage5_retrank_only_broad_A.json"),        # HMR 0.858
    ("tier1p2_broad_A", "ac_stage5_tier1plus2_broad_A.json"),          # HMR 0.840
    ("retrank_sonnet_A", "ac_stage5_retrank_only_refined_sonnet_A.json"),# HMR 0.838
    ("tier1_lexical_A", "ac_stage5_tier1_refined_lexical_A.json"),     # HMR 0.833
    ("tier1_broad_A", "ac_stage5_tier1_broad_A.json"),                 # HMR 0.826
    ("tier1_ce_A", "ac_stage5_tier1_refined_ce_A.json"),               # HMR 0.802
    ("rE404_sonnet_A", "ac_stage5_retrank_plus_error404_refined_sonnet_A.json"),  # HMR 0.738
]

CANDIDATE_SETS = {
    "all": CANDIDATES_ALL,
    "top12": CANDIDATES_RANKED[:12],
    "top9": CANDIDATES_RANKED[:9],
    "top7": CANDIDATES_RANKED[:7],
    "top5": CANDIDATES_RANKED[:5],
    "top4": CANDIDATES_RANKED[:4],
    "top3": CANDIDATES_RANKED[:3],
    "top2": CANDIDATES_RANKED[:2],
    "top1": CANDIDATES_RANKED[:1],
}


def normalise(s: str) -> str:
    """Loose normalisation for clustering: lowercase, strip punctuation/articles."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]", "", s)  # keep word chars, whitespace, hyphens
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s.strip()


def lexical_cluster(answers: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """First-pass clustering by string similarity.

    answers: list of (cand_name, raw_answer)
    Returns: dict of {representative_answer: [(cand, raw), ...]}
    """
    clusters: dict[str, list[tuple[str, str]]] = {}
    for cand, raw in answers:
        if not raw.strip():
            continue
        norm = normalise(raw)
        if not norm:
            continue
        # Match against existing clusters: exact, substring, or shared all words
        match_key = None
        for key in clusters:
            if norm == key or norm in key or key in norm:
                match_key = key
                break
        if match_key is not None:
            clusters[match_key].append((cand, raw))
        else:
            clusters[norm] = [(cand, raw)]
    return clusters


CLUSTER_PROMPT = """For a movie question, group these candidate answers by semantic equivalence.
Two answers are equivalent if they refer to the same fact (same person, same number, same title — paraphrasing/formatting OK).

Question: {question}

Candidate answers (numbered):
{ans_block}

Output JSON only — a list of groups, each group is a list of answer numbers that are equivalent:
{{"groups": [[1, 3], [2], [4, 5, 6]]}}"""


def call_haiku(client, prompt: str, retries: int = 4, min_interval: float = 0.6) -> dict:
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            dt = time.monotonic() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)
            text = r.content[0].text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            return json.loads(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError("Haiku clustering failed")


def haiku_cluster(client, question: str, answers: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Use Haiku to cluster. answers = [(cand, raw_answer), ...]"""
    if not answers:
        return {}
    ans_block = "\n".join(f"[{i+1}] {a}" for i, (_, a) in enumerate(answers))
    try:
        result = call_haiku(client, CLUSTER_PROMPT.format(question=question, ans_block=ans_block))
        groups = result.get("groups", [])
    except Exception as e:
        print(f"  Haiku clustering failed: {e}; falling back to lexical", file=sys.stderr)
        return lexical_cluster(answers)

    clusters: dict[str, list[tuple[str, str]]] = {}
    for grp in groups:
        if not grp:
            continue
        # Validate indices and pick the longest answer in the group as representative
        members = []
        for idx in grp:
            if isinstance(idx, int) and 1 <= idx <= len(answers):
                members.append(answers[idx - 1])
        if not members:
            continue
        rep = max(members, key=lambda x: len(x[1]))[1]  # longest answer as label
        clusters[normalise(rep)] = members
    return clusters


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        return {m.group(1).strip(): m.group(2).strip()
                for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL)}
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5-dir", default=str(BASE / "data/processed"))
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-with-haiku", action="store_true",
                        help="Use Haiku for fuzzy clustering when answers differ in spelling. ~$1 per pool of topics.")
    parser.add_argument("--cluster-cache", default=str(BASE / "data/eval/ac_cache/ensemble_clusters.json"))
    parser.add_argument("--min-support", type=float, default=0.50,
                        help="Refuse if no cluster covers this fraction of voters")
    parser.add_argument("--candidate-set", choices=list(CANDIDATE_SETS.keys()), default="all",
                        help="Which subset of candidates to ensemble (all / top5 / top3)")
    args = parser.parse_args()

    topics = load_topics(Path(args.topics))

    candidate_list = CANDIDATE_SETS[args.candidate_set]
    print(f"Using candidate set '{args.candidate_set}' ({len(candidate_list)} pipelines)")

    # Load candidates
    loaded = {}
    for tag, fname in candidate_list:
        p = Path(args.stage5_dir) / fname
        if not p.exists():
            print(f"  ⚠ missing: {fname}", file=sys.stderr)
            continue
        loaded[tag] = json.loads(p.read_text())["topics"]
    print(f"Loaded {len(loaded)}/{len(candidate_list)} candidate runs")

    cluster_cache_path = Path(args.cluster_cache)
    cluster_cache = (json.loads(cluster_cache_path.read_text())
                     if cluster_cache_path.exists() else {})

    client = anthropic.Anthropic() if args.cluster_with_haiku else None
    new_haiku_calls = 0

    out_topics: dict[str, dict] = {}
    n_consensus = 0
    n_refused = 0
    support_rates = []
    pipelines_per_consensus: list[int] = []

    for qid in sorted(topics.keys()):
        question = topics[qid]
        answers: list[tuple[str, str]] = []
        for tag, run in loaded.items():
            rec = run.get(qid, {})
            ans = (rec.get("answer") or "").strip()
            answers.append((tag, ans))

        # Clustering
        non_empty = [(t, a) for t, a in answers if a]
        if not non_empty:
            # Everyone refused → ensemble refuses too
            out_topics[qid] = {"answer": "", "confidence": 5,
                                "calibration_reason": "all candidates refused",
                                "filtered_nuggets": [],
                                "support": 0, "n_voters": len(answers),
                                "consensus_pipelines": []}
            n_refused += 1
            continue

        cache_key = f"{qid}|{'|'.join(sorted(set(a for _, a in non_empty)))}"
        if args.cluster_with_haiku:
            if cache_key in cluster_cache:
                clusters_idx = cluster_cache[cache_key]
                # Translate index-groups back to (cand, raw) members
                clusters = {}
                for rep_idx, grp in clusters_idx.items():
                    members = [non_empty[i] for i in grp if 0 <= i < len(non_empty)]
                    if members:
                        clusters[normalise(members[0][1])] = members
            else:
                clusters = haiku_cluster(client, question, non_empty)
                # Save groups as index lists for caching
                idx_groups = {}
                for rep, members in clusters.items():
                    idx_groups[rep] = [non_empty.index(m) for m in members if m in non_empty]
                cluster_cache[cache_key] = idx_groups
                new_haiku_calls += 1
                if new_haiku_calls % 10 == 0:
                    cluster_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cluster_cache_path.write_text(json.dumps(cluster_cache, indent=1, ensure_ascii=False))
        else:
            clusters = lexical_cluster(non_empty)

        if not clusters:
            out_topics[qid] = {"answer": "", "confidence": 5,
                                "calibration_reason": "no valid clusters",
                                "filtered_nuggets": [],
                                "support": 0, "n_voters": len(answers),
                                "consensus_pipelines": []}
            n_refused += 1
            continue

        # Pick the largest cluster
        best_key, best_members = max(clusters.items(), key=lambda kv: len(kv[1]))
        support = len(best_members)
        n_voters = len(answers)
        support_rate = support / n_voters

        if support_rate < args.min_support:
            out_topics[qid] = {"answer": "", "confidence": 5,
                                "calibration_reason": f"no clear majority ({support}/{n_voters})",
                                "filtered_nuggets": [],
                                "support": support, "n_voters": n_voters,
                                "consensus_pipelines": [m[0] for m in best_members]}
            n_refused += 1
            continue

        # Pick a representative answer (the longest one in the cluster — usually most informative)
        rep_answer = max(best_members, key=lambda m: len(m[1]))[1]

        # Confidence proportional to support_rate (clamped 5..100)
        confidence = max(5, min(100, round(support_rate * 100)))

        # Pick nuggets from the cluster member with highest match_score / nugget count
        # Look up each contributor's filtered_nuggets and choose the best
        best_pipeline = None
        best_score = -1
        for tag, _ in best_members:
            rec = loaded[tag].get(qid, {})
            ms = int(rec.get("match_score", 0) or 0)
            n_nug = len(rec.get("filtered_nuggets", []))
            score = ms * 100 + n_nug   # prefer higher match, more nuggets
            if score > best_score:
                best_score = score
                best_pipeline = tag

        chosen_nuggets = loaded[best_pipeline].get(qid, {}).get("filtered_nuggets", []) if best_pipeline else []

        out_topics[qid] = {
            "answer": rep_answer,
            "confidence": confidence,
            "calibration_reason": f"ensemble {support}/{n_voters} pipelines agree (from {best_pipeline})",
            "filtered_nuggets": chosen_nuggets,
            "support": support,
            "n_voters": n_voters,
            "consensus_pipelines": [m[0] for m in best_members],
            "all_clusters": {k: [m[0] for m in mems] for k, mems in clusters.items()},
        }
        n_consensus += 1
        support_rates.append(support_rate)
        pipelines_per_consensus.append(support)

    if args.cluster_with_haiku:
        cluster_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_cache_path.write_text(json.dumps(cluster_cache, indent=1, ensure_ascii=False))

    output = {
        "_meta": {
            "n_candidates": len(loaded),
            "candidates": list(loaded.keys()),
            "n_topics": len(topics),
            "n_consensus": n_consensus,
            "n_refused": n_refused,
            "min_support_threshold": args.min_support,
            "cluster_method": "haiku" if args.cluster_with_haiku else "lexical",
            "new_haiku_calls": new_haiku_calls,
        },
        "topics": out_topics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  RQ9 — Cross-Pool Answer Ensembler                                 ║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    print(f"║  Candidates:     {len(loaded):<46}║")
    print(f"║  Topics:         {len(topics):<46}║")
    print(f"║  Consensus:      {n_consensus:<46}║")
    print(f"║  Refused:        {n_refused:<46}║")
    if support_rates:
        print(f"║  Mean support:   {sum(support_rates)/len(support_rates)*100:>4.1f}%{'':<41}║")
        print(f"║  Median support: {sorted(support_rates)[len(support_rates)//2]*100:>4.1f}%{'':<41}║")
    print(f"║  Cluster method: {output['_meta']['cluster_method']:<46}║")
    print(f"║  New Haiku calls:{new_haiku_calls:<46}║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")
    print(f"\nFull output: {args.output}")


if __name__ == "__main__":
    main()
