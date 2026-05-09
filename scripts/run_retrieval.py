#!/usr/bin/env python3
"""
BM25 retrieval over segmented passages.

Topics file:  CSV/TSV with columns topic_id, question
Index:        data/processed/bm25_index.pkl  (built by index_bm25.py)
Output:       data/runs/bm25_run.txt

Run file format (one line per passage):
    TopicID;PassageRank;DocumentID;PassageText
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

from tqdm import tqdm


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_topics(path: Path) -> list[tuple[str, str]]:
    """Return list of (topic_id, question). Auto-detects CSV vs TSV."""
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    topics = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            topic_id = row.get("topic_id") or row.get("TopicID") or row.get("qid")
            question = row.get("question") or row.get("Question") or row.get("query")
            if topic_id is None or question is None:
                raise ValueError(
                    f"Cannot find topic_id/question columns. Found: {list(row.keys())}"
                )
            topics.append((topic_id.strip(), question.strip()))
    return topics


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval for R2C2 topics")
    parser.add_argument("topics", help="Topics file (CSV/TSV: topic_id, question)")
    parser.add_argument("--index", default="data/processed/bm25_index.pkl",
                        help="BM25 index pickle (default: data/processed/bm25_index.pkl)")
    parser.add_argument("--output", default="data/runs/bm25_run.txt",
                        help="Output run file (default: data/runs/bm25_run.txt)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of passages to retrieve per topic (default: 100)")
    args = parser.parse_args()

    topics_path = Path(args.topics)
    index_path = Path(args.index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading BM25 index from {index_path} ...")
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]
    records = payload["records"]
    print(f"Index contains {len(records):,} passages.")

    print(f"Loading topics from {topics_path} ...")
    topics = load_topics(topics_path)
    print(f"Loaded {len(topics):,} topics.")

    print(f"Running BM25 retrieval (top-{args.top_k}) ...")
    with output_path.open("w", encoding="utf-8") as fout:
        for topic_id, question in tqdm(topics, desc="Topics"):
            query_tokens = tokenize(question)
            scores = bm25.get_scores(query_tokens)

            # Rank indices by descending score, take top-k
            ranked_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: args.top_k]

            for rank, idx in enumerate(ranked_indices, start=1):
                rec = records[idx]
                # Escape semicolons in passage text to avoid breaking the format
                passage_text = rec["passage_text"].replace(";", ",")
                line = f"{topic_id};{rank};{rec['doc_id']};{passage_text}\n"
                fout.write(line)

    print(f"Run saved to {output_path}")


if __name__ == "__main__":
    main()
