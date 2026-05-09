#!/usr/bin/env python3
"""
PG (own passages) extractor: given first-stage retrieval results, loads full
document text from passages_meta.pkl (pre-segmented corpus), then extracts
200-char sliding-window passages, scores with cross-encoder, and selects
top-20 with intra-doc diversity.

Uses passages_meta.pkl for fast doc text reconstruction (no tar.gz scanning).
Document text = sorted 200-char chunks joined together.

Intra-doc diversity (union test):
  After CE-scoring all passages within a doc, greedily select passages:
  start with highest-scoring p1, then for each candidate pi check if
  CE(query, p1_text + " " + pi_text) > CE(query, p1_text) + delta.
  If not, pi adds no new information → skip it.

Usage:
    python scripts/pg_extractor.py \\
        --topics data/raw/r2c2topics.txt \\
        --run data/runs/pool_bm25_500_val65.txt \\
        --output data/runs/pg_bm25_td20_mpd3_s100_d005.txt \\
        --top-docs 20 --max-per-doc 3 --stride 100 --delta 0.05
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

BASE = Path(__file__).resolve().parent.parent
META_FILE = BASE / "data/processed/passages_meta.pkl"


# ── Doc text loading from passages_meta ──────────────────────────────────────

_doc_texts: dict | None = None

def load_all_doc_texts() -> dict[str, str]:
    """
    Load passages_meta.pkl and reconstruct full document text per doc_id
    by joining sorted 200-char chunks. Fast: single pkl load, no archive scan.
    """
    global _doc_texts
    if _doc_texts is not None:
        return _doc_texts
    print(f"Loading passages_meta from {META_FILE} ...")
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages, grouping by doc_id ...")
    by_doc: dict[str, list] = defaultdict(list)
    for m in meta:
        by_doc[m["doc_id"]].append(m)
    # Sort chunks by chunk_id (e.g. Q123__00001, Q123__00002, ...) and join
    _doc_texts = {}
    for doc_id, chunks in by_doc.items():
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_id"])
        _doc_texts[doc_id] = " ".join(c["text_snippet"] for c in sorted_chunks)
    print(f"  {len(_doc_texts):,} unique documents")
    return _doc_texts


def get_doc_text(doc_id: str) -> str:
    return load_all_doc_texts().get(doc_id, "")


# ── Sliding window passage extraction ────────────────────────────────────────

def sliding_window_passages(text: str, max_len: int = 200, stride: int = 100,
                             min_len: int = 50) -> list[str]:
    """Extract overlapping passages of ≤max_len chars, word-boundary aligned."""
    if not text:
        return []
    words = text.split()
    passages = []
    i = 0
    while i < len(words):
        chunk = []
        length = 0
        j = i
        while j < len(words) and length + len(words[j]) + 1 <= max_len:
            chunk.append(words[j])
            length += len(words[j]) + 1
            j += 1
        if j == i:  # single word longer than max_len
            j = i + 1
        p = " ".join(chunk).strip()
        if len(p) >= min_len:
            passages.append(p)
        # Advance by stride words (approximate, ~6 chars/word average)
        stride_words = max(1, stride // 6)
        i += stride_words
    # Deduplicate adjacent identical passages
    return list(dict.fromkeys(passages))


# ── Cross-encoder scoring + intra-doc diversity ───────────────────────────────

def score_passages(ce: CrossEncoder, query: str,
                   passages: list[str], batch_size: int = 64) -> list[float]:
    """Score (query, passage) pairs with CE. Returns scores list."""
    if not passages:
        return []
    pairs = [[query, p] for p in passages]
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return scores.tolist() if hasattr(scores, "tolist") else list(scores)


def intra_doc_select_overlap(passages: list[str], scores: list[float],
                              max_select: int,
                              overlap_threshold: float = 0.5) -> list[tuple[float, str]]:
    """
    Fast Jaccard-overlap diversity selection within one document.
    Much cheaper than CE-based union test: no extra CE calls needed.
    """
    if not passages:
        return []
    ranked = sorted(zip(scores, passages), reverse=True)
    selected: list[tuple[float, str]] = []
    selected_words: list[set] = []

    for score, passage in ranked:
        if len(selected) >= max_select:
            break
        words = set(re.findall(r"[a-z0-9]+", passage.lower()))
        if not any(
            len(words & prev) / max(len(words | prev), 1) > overlap_threshold
            for prev in selected_words
        ):
            selected.append((score, passage))
            selected_words.append(words)

    return selected


def intra_doc_select(ce: CrossEncoder, query: str, passages: list[str],
                     scores: list[float], max_select: int,
                     delta: float = 0.05) -> list[tuple[float, str]]:
    """
    Greedy CE-based diversity selection within one document.
    For each candidate, test if CE(query, context+candidate) > best+delta.
    Use diversity_mode='overlap' for the fast path (no extra CE calls).
    """
    if not passages:
        return []

    ranked = sorted(zip(scores, passages), reverse=True)
    selected: list[tuple[float, str]] = []
    context = ""

    for score, passage in ranked:
        if len(selected) >= max_select:
            break
        if not selected:
            selected.append((score, passage))
            context = passage
            continue

        union_text = context + " " + passage
        union_score = ce.predict([[query, union_text[:400]]],
                                  show_progress_bar=False)[0]
        baseline = selected[0][0]
        if union_score > baseline + delta:
            selected.append((score, passage))
            context = union_text

    return selected


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_run(run_path: Path, top_k: int = 500) -> dict[str, list[tuple[int, str, str]]]:
    """Load run → {topic_id: [(rank, doc_id, text), ...]}."""
    run = defaultdict(list)
    with run_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", 3)
            if len(parts) != 4:
                continue
            topic_id, rank, doc_id, text = parts
            rank = int(rank)
            if rank <= top_k:
                run[topic_id].append((rank, doc_id, text))
    return {tid: sorted(v) for tid, v in run.items()}


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


def run_pg_pipeline(topics: dict, run: dict, ce: CrossEncoder,
                    top_docs: int, max_per_doc: int, stride: int,
                    delta: float, final_k: int = 20,
                    diversity_mode: str = "overlap") -> dict[str, list[tuple[float, str, str]]]:
    """
    Full PG pipeline for all topics.
    Returns {topic_id: [(score, doc_id, passage_text), ...]} top final_k.
    """
    # Trigger doc loading once upfront (not per topic)
    doc_texts = load_all_doc_texts()

    results = {}
    total = len(run)

    for i, (tid, passages_ranked) in enumerate(sorted(run.items())):
        question = topics.get(tid, "")
        if not question:
            continue

        # Get top-N unique docs by first appearance in ranking
        seen_docs = []
        seen_set: set[str] = set()
        for rank, doc_id, _ in passages_ranked:
            if doc_id not in seen_set:
                seen_set.add(doc_id)
                seen_docs.append(doc_id)
            if len(seen_docs) >= top_docs:
                break

        # Extract sliding window passages per doc
        all_candidates: list[tuple[str, str]] = []  # (doc_id, passage)
        for doc_id in seen_docs:
            text = doc_texts.get(doc_id, "")
            if not text:
                continue
            passages = sliding_window_passages(text, max_len=200, stride=stride)
            for p in passages:
                all_candidates.append((doc_id, p))

        if not all_candidates:
            results[tid] = []
            continue

        # Score all candidates with CE
        candidate_texts = [p for _, p in all_candidates]
        scores = score_passages(ce, question, candidate_texts)

        # Group by doc
        by_doc: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for (doc_id, passage), score in zip(all_candidates, scores):
            by_doc[doc_id].append((score, passage))

        # Intra-doc diversity selection
        doc_selected: list[tuple[float, str, str]] = []
        for doc_id in seen_docs:
            if doc_id not in by_doc:
                continue
            doc_passages = [p for _, p in by_doc[doc_id]]
            doc_scores = [s for s, _ in by_doc[doc_id]]
            if diversity_mode == "ce":
                selected = intra_doc_select(ce, question, doc_passages, doc_scores,
                                            max_select=max_per_doc, delta=delta)
            elif diversity_mode == "none":
                ranked = sorted(zip(doc_scores, doc_passages), reverse=True)
                selected = ranked[:max_per_doc]
            else:  # "overlap" (default) — fast Jaccard-based dedup
                selected = intra_doc_select_overlap(doc_passages, doc_scores,
                                                    max_select=max_per_doc)
            for score, passage in selected:
                doc_selected.append((score, doc_id, passage))

        # Sort globally by score, take top final_k
        doc_selected.sort(key=lambda x: -x[0])
        results[tid] = doc_selected[:final_k]

        n_passages = len(doc_selected)
        top_score = doc_selected[0][0] if doc_selected else 0
        print(f"  [{i+1}/{total}] {tid}: {len(seen_docs)} docs, "
              f"{len(all_candidates)} candidates → {n_passages} selected, "
              f"top_score={top_score:.3f}", flush=True)

    return results


def write_run_file(results: dict, output_path: Path):
    """Write results to submission format: topic_id;rank;doc_id;passage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for tid in sorted(results.keys()):
            for rank, (score, doc_id, passage) in enumerate(results[tid], 1):
                passage = passage[:200].replace("\n", " ").replace(";", ",")
                f.write(f"{tid};{rank};{doc_id};{passage}\n")
    print(f"Run written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--topics-file", default=None)
    parser.add_argument("--run", required=True, help="First-stage run file (pool format)")
    parser.add_argument("--output", required=True, help="Output run file path")
    parser.add_argument("--meta", default=str(META_FILE), help="passages_meta.pkl path")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--top-docs",    type=int,   default=20,   help="Unique docs to extract from")
    parser.add_argument("--max-per-doc", type=int,   default=3,    help="Max passages per doc after diversity")
    parser.add_argument("--stride",      type=int,   default=100,  help="Sliding window stride in chars")
    parser.add_argument("--delta",       type=float, default=0.05, help="Union test delta for CE diversity")
    parser.add_argument("--final-k",     type=int,   default=20,   help="Final passages per topic")
    parser.add_argument("--diversity-mode", default="overlap",
                        choices=["overlap", "none", "ce"],
                        help="Intra-doc diversity: overlap=Jaccard (fast), none=top-k, ce=CE union test (slow)")
    args = parser.parse_args()

    topics = load_topics(Path(args.topics),
                         Path(args.topics_file) if args.topics_file else None)
    print(f"Loaded {len(topics)} topics", flush=True)

    run = load_run(Path(args.run))
    print(f"Loaded run: {len(run)} topics", flush=True)

    print(f"Loading CE model: {args.ce_model}", flush=True)
    ce = CrossEncoder(args.ce_model, max_length=512)

    print(f"\nPG extraction: top_docs={args.top_docs}, max_per_doc={args.max_per_doc}, "
          f"stride={args.stride}, diversity={args.diversity_mode}", flush=True)
    results = run_pg_pipeline(topics, run, ce,
                              top_docs=args.top_docs,
                              max_per_doc=args.max_per_doc,
                              stride=args.stride,
                              delta=args.delta,
                              final_k=args.final_k,
                              diversity_mode=args.diversity_mode)

    write_run_file(results, Path(args.output))
    covered = sum(1 for v in results.values() if v)
    print(f"Done: {covered}/{len(results)} topics have passages")


if __name__ == "__main__":
    main()
