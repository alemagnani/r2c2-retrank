#!/usr/bin/env python3
"""
Build a BM25 index from segmented passages.

Input JSONL:  {doc_id, url, passage_rank, passage_text}
Output:       data/processed/bm25_index.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from segmented passages")
    parser.add_argument("passages", help="Input JSONL passages file")
    parser.add_argument("--output", default="data/processed/bm25_index.pkl",
                        help="Path for the saved index (default: data/processed/bm25_index.pkl)")
    args = parser.parse_args()

    passages_path = Path(args.passages)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from {passages_path} ...")
    records = []
    with passages_path.open() as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records):,} passages. Tokenising ...")
    corpus_tokens = [tokenize(r["passage_text"]) for r in tqdm(records, desc="Tokenising")]

    print("Building BM25 index ...")
    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "records": records,          # parallel list: records[i] ↔ corpus_tokens[i]
    }

    print(f"Saving index to {output_path} ...")
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":
    main()
