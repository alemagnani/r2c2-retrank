#!/usr/bin/env python3
"""
Cross-encoder reranker for R2C2.

Takes a first-stage BM25 run (top-k candidates) and reranks with a
cross-encoder model.

Usage:
    python scripts/run_reranker.py \
        --run data/runs/alemagnani-PO-1.txt \
        --topics data/raw/r2c2topics.txt \
        --output data/runs/alemagnani-PO-2.txt \
        --top-k 20 \
        --candidates 100

    # Increase --candidates to rerank more BM25 results (better recall, slower).
"""

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent

MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def load_topics(path: Path) -> dict[str, str]:
    """Return {topic_id: question}."""
    content = path.read_text(encoding="utf-8")
    first_line = next(l for l in content.splitlines() if l.strip())
    if first_line.strip().startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    # TSV/CSV fallback
    import csv
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    topics = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or row.get("query") or "").strip()
            if tid and q:
                topics[tid] = q
    return topics


def load_run(path: Path, max_candidates: int) -> dict[str, list[tuple[str, str]]]:
    """Return {topic_id: [(doc_id, text), ...]} keeping top-max_candidates per topic."""
    run = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";", 3)
            if len(parts) != 4:
                continue
            topic_id, rank, doc_id, text = parts
            if len(run[topic_id]) < max_candidates:
                run[topic_id].append((doc_id, text))
    return dict(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(BASE / "data/runs/alemagnani-PO-1.txt"),
                        help="First-stage run file")
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", default=str(BASE / "data/runs/alemagnani-PO-2.txt"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--top-k", type=int, default=20,
                        help="Passages to keep in output per topic")
    parser.add_argument("--candidates", type=int, default=100,
                        help="BM25 candidates to rerank per topic")
    args = parser.parse_args()

    print(f"Loading topics from {args.topics} ...")
    topics = load_topics(Path(args.topics))
    print(f"  {len(topics)} topics")

    print(f"Loading first-stage run from {args.run} ...")
    run = load_run(Path(args.run), args.candidates)
    print(f"  {sum(len(v) for v in run.values())} candidate passages across {len(run)} topics")

    print(f"Loading cross-encoder: {args.model} ...")
    model = CrossEncoder(args.model, max_length=512)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reranking (top-{args.candidates} → top-{args.top_k}) ...")
    with output_path.open("w", encoding="utf-8") as fout:
        for topic_id, candidates in tqdm(run.items(), desc="Topics"):
            question = topics.get(topic_id, "")
            if not question:
                continue

            pairs = [(question, text) for _, text in candidates]
            scores = model.predict(pairs, show_progress_bar=False)

            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

            for rank, (score, (doc_id, text)) in enumerate(ranked[: args.top_k], start=1):
                fout.write(f"{topic_id};{rank};{doc_id};{text}\n")

    print(f"Reranked run saved to {args.output}")


if __name__ == "__main__":
    main()
