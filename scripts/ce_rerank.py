#!/usr/bin/env python3
"""
CE reranker: takes a pool file (top-K passages), scores with cross-encoder,
applies diversity selection, outputs top-20 in submission format.

Pool format (same as gen_pools.py output):
    topic_id;rank;doc_id;passage_text

Output format:
    topic_id;rank;doc_id;passage_text  (top-20 per topic)

Selection modes:
  topk     — straight top-20 by CE score (no diversity)
  overlap  — Jaccard overlap dedup (known best from experiments)
  mmr      — Maximal Marginal Relevance (lambda-weighted CE vs similarity to selected)

Usage:
    python scripts/ce_rerank.py \\
        --topics data/raw/r2c2topics.txt \\
        --pool data/runs/pool_bm25_100_own.txt \\
        --output data/runs/own_bm25_100_topk.txt \\
        --selection topk

    python scripts/ce_rerank.py \\
        --topics-file data/synthetic_val/synthetic_topics_val250.json \\
        --pool data/runs/pool_bm25_100_own.txt \\
        --output data/runs/own_bm25_100_overlap.txt \\
        --selection overlap --max-per-doc 3
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

BASE = Path(__file__).resolve().parent.parent


# ── Topic / pool loading ──────────────────────────────────────────────────────

def load_topics(topics_path: Path, topics_file: Path | None) -> dict[str, str]:
    if topics_file and topics_file.exists():
        data = json.loads(topics_file.read_text())
        topics = {}
        for rec in data:
            tid = rec.get("loop_topic_id") or rec.get("topic_id", "")
            q = rec.get("question", "")
            if tid and q:
                topics[tid] = q
        return topics
    content = topics_path.read_text(encoding="utf-8")
    pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
    return {qid.strip(): q.strip() for qid, q in pairs}


def load_pool(pool_path: Path, top_k: int | None = None) -> dict[str, list[tuple]]:
    """Returns {topic_id: [(rank, doc_id, text), ...]} sorted by rank."""
    pool = defaultdict(list)
    with pool_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", 3)
            if len(parts) != 4:
                continue
            tid, rank, doc_id, text = parts
            rank = int(rank)
            if top_k is None or rank <= top_k:
                pool[tid].append((rank, doc_id, text))
    return {tid: sorted(v) for tid, v in pool.items()}


# ── Selection strategies ──────────────────────────────────────────────────────

def select_topk(passages: list[tuple], scores: list[float],
                final_k: int) -> list[tuple[float, str, str]]:
    """Straight top-k by CE score."""
    ranked = sorted(zip(scores, [d for _, d, _ in passages],
                        [t for _, _, t in passages]), reverse=True)
    return [(s, d, t) for s, d, t in ranked[:final_k]]


def select_overlap(passages: list[tuple], scores: list[float],
                   final_k: int, max_per_doc: int,
                   overlap_threshold: float = 0.5) -> list[tuple[float, str, str]]:
    """Jaccard overlap diversity — fast, no extra CE calls."""
    ranked = sorted(zip(scores, [d for _, d, _ in passages],
                        [t for _, _, t in passages]), reverse=True)
    selected: list[tuple[float, str, str]] = []
    selected_words: list[set] = []
    doc_counts: dict[str, int] = defaultdict(int)

    for score, doc_id, text in ranked:
        if len(selected) >= final_k:
            break
        if doc_counts[doc_id] >= max_per_doc:
            continue
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        if any(len(words & prev) / max(len(words | prev), 1) > overlap_threshold
               for prev in selected_words):
            continue
        selected.append((score, doc_id, text))
        selected_words.append(words)
        doc_counts[doc_id] += 1

    return selected


def select_mmr(passages: list[tuple], scores: list[float],
               final_k: int, max_per_doc: int,
               lambda_mmr: float = 0.5) -> list[tuple[float, str, str]]:
    """
    Maximal Marginal Relevance: balance relevance vs diversity.
    Uses word-overlap as similarity proxy (no extra CE calls needed).
    lambda_mmr=1.0 → pure relevance; lambda_mmr=0.0 → pure diversity.
    """
    items = [(s, d, t) for s, d, t in
             zip(scores, [d for _, d, _ in passages], [t for _, _, t in passages])]

    if not items:
        return []

    # Normalise scores to [0, 1]
    min_s, max_s = min(s for s, _, _ in items), max(s for s, _, _ in items)
    span = max_s - min_s or 1.0
    norm = [(( s - min_s) / span, d, t) for s, d, t in items]

    selected: list[tuple[float, str, str]] = []
    selected_words: list[set] = []
    doc_counts: dict[str, int] = defaultdict(int)
    remaining = list(norm)

    while remaining and len(selected) < final_k:
        best_score = -1.0
        best_idx = 0

        for i, (rel, doc_id, text) in enumerate(remaining):
            if doc_counts[doc_id] >= max_per_doc:
                continue
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            if selected_words:
                max_sim = max(
                    len(words & prev) / max(len(words | prev), 1)
                    for prev in selected_words
                )
            else:
                max_sim = 0.0
            mmr_score = lambda_mmr * rel - (1 - lambda_mmr) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        rel, doc_id, text = remaining.pop(best_idx)
        actual_score = rel * span + min_s
        selected.append((actual_score, doc_id, text))
        selected_words.append(set(re.findall(r"[a-z0-9]+", text.lower())))
        doc_counts[doc_id] += 1

    return selected


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_passages(ce: CrossEncoder, query: str,
                   passages: list[tuple], batch_size: int = 128) -> list[float]:
    texts = [text for _, _, text in passages]
    pairs = [[query, t] for t in texts]
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return scores.tolist() if hasattr(scores, "tolist") else list(scores)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--topics-file", default=None)
    parser.add_argument("--pool", required=True, help="Pool file to rerank")
    parser.add_argument("--output", required=True, help="Output run file")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Use only top-K from pool (default: all)")
    parser.add_argument("--final-k", type=int, default=20, help="Passages per topic")
    parser.add_argument("--selection", default="overlap",
                        choices=["topk", "overlap", "mmr"],
                        help="Selection strategy after CE scoring")
    parser.add_argument("--max-per-doc", type=int, default=3,
                        help="Max passages per doc (overlap/mmr modes)")
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--lambda-mmr", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    topics = load_topics(Path(args.topics),
                         Path(args.topics_file) if args.topics_file else None)
    print(f"Loaded {len(topics)} topics", flush=True)

    pool = load_pool(Path(args.pool), args.pool_size)
    print(f"Loaded pool: {len(pool)} topics, "
          f"avg {sum(len(v) for v in pool.values())/max(len(pool),1):.0f} passages/topic",
          flush=True)

    print(f"Loading CE model: {args.ce_model}", flush=True)
    ce = CrossEncoder(args.ce_model, max_length=512)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    total = len(pool)

    for i, (tid, passages) in enumerate(sorted(pool.items())):
        question = topics.get(tid, "")
        if not question:
            continue

        scores = score_passages(ce, question, passages, args.batch_size)

        if args.selection == "topk":
            selected = select_topk(passages, scores, args.final_k)
        elif args.selection == "overlap":
            selected = select_overlap(passages, scores, args.final_k,
                                      args.max_per_doc, args.overlap_threshold)
        else:  # mmr
            selected = select_mmr(passages, scores, args.final_k,
                                  args.max_per_doc, args.lambda_mmr)

        results[tid] = selected
        top_score = selected[0][0] if selected else 0
        print(f"  [{i+1}/{total}] {tid}: {len(passages)} → {len(selected)} "
              f"({args.selection}), top={top_score:.3f}", flush=True)

    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        for tid in sorted(results.keys()):
            for rank, (score, doc_id, text) in enumerate(results[tid], 1):
                text = text[:200].replace("\n", " ").replace(";", ",")
                f.write(f"{tid};{rank};{doc_id};{text}\n")

    covered = sum(1 for v in results.values() if v)
    print(f"\nWritten to {output_path}: {covered}/{total} topics covered")


if __name__ == "__main__":
    main()
