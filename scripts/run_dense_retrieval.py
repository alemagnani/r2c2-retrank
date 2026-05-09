#!/usr/bin/env python3
"""
Dense retrieval for R2C2 topics using a bi-encoder + FAISS index.

Usage:
    python scripts/run_dense_retrieval.py data/raw/r2c2topics.txt \
        --output data/runs/alemagnani-PO-3.txt --top-k 20

    # For Run 4 input (more candidates for reranker):
    python scripts/run_dense_retrieval.py data/raw/r2c2topics.txt \
        --output data/runs/dense_top100.txt --top-k 100
"""

import argparse
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
META_FILE = BASE / "data/processed/passages_meta.pkl"
FAISS_INDEX = BASE / "data/processed/faiss_index.bin"
DEFAULT_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"


def load_topics(path: Path) -> list[tuple[str, str]]:
    content = path.read_text(encoding="utf-8")
    first_line = next(l for l in content.splitlines() if l.strip())
    if first_line.strip().startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return [(qid.strip(), q.strip()) for qid, q in pairs]
    import csv
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    topics = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or row.get("query") or "").strip()
            if tid and q:
                topics.append((tid, q))
    return topics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", help="Topics file (XML or TSV/CSV)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--index", default=str(FAISS_INDEX))
    parser.add_argument("--meta", default=str(META_FILE))
    parser.add_argument("--output", default=str(BASE / "data/runs/alemagnani-PO-3.txt"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hyde", default=None,
                        help="Path to HyDE queries JSON. Uses hypothetical passages as queries.")
    args = parser.parse_args()

    print(f"Loading passage metadata from {args.meta} ...")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages")

    print(f"Loading FAISS index from {args.index} ...")
    index = faiss.read_index(args.index)
    print(f"  {index.ntotal:,} vectors")

    print(f"Loading bi-encoder: {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"Loading topics from {args.topics} ...")
    topics = load_topics(Path(args.topics))
    print(f"  {len(topics)} topics")

    hyde_queries = {}
    if args.hyde:
        import json
        with open(args.hyde) as f:
            raw = json.load(f)
        hyde_queries = {tid: v["hypothesis"] for tid, v in raw.items()}
        print(f"  HyDE: using hypothetical passages for {len(hyde_queries)} topics")

    queries = [hyde_queries.get(tid, q) for tid, q in topics]
    print("Encoding queries ...")
    query_embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print(f"Searching (top-{args.top_k}) ...")
    scores, indices = index.search(query_embeddings, args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for (topic_id, _), top_indices in zip(topics, indices):
            for rank, idx in enumerate(top_indices, start=1):
                rec = meta[idx]
                fout.write(f"{topic_id};{rank};{rec['doc_id']};{rec['text_snippet']}\n")

    print(f"Run saved to {args.output}")


if __name__ == "__main__":
    main()
