#!/usr/bin/env python3
"""
BM25S retrieval for R2C2 topics.

Topics file format (auto-detected XML or TSV/CSV):
  XML: <topic><num>0001</num><title>What movie...</title></topic>
  TSV/CSV: topic_id [tab/comma] question

Output format (one line per passage):
  TopicID;PassageRank;DocumentID;PassageText

Usage:
    python scripts/run_bm25_retrieval.py data/raw/topics.xml
    python scripts/run_bm25_retrieval.py data/raw/topics.tsv --top-k 20 --output data/runs/team-PO-1.txt
"""

import argparse
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import bm25s
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE / "data/processed/bm25s_index"
META_FILE = BASE / "data/processed/passages_meta.pkl"


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_topics_xml(path: Path) -> list[tuple[str, str]]:
    """Parse R2C2/TREC XML topics with regex (tolerates unescaped & and other bad XML)."""
    content = path.read_text(encoding="utf-8")
    # R2C2 format: <qID>0001</qID> ... <q>...</q>
    r2c2 = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
    if r2c2:
        return [(qid.strip(), q.strip()) for qid, q in r2c2]
    # TREC format: <num>...</num> ... <title>...</title>
    trec = re.findall(r"<num>\s*(.*?)\s*</num>.*?<title>\s*(.*?)\s*</title>", content, re.DOTALL)
    return [(num.strip(), title.strip()) for num, title in trec]


def load_topics_tsv(path: Path) -> list[tuple[str, str]]:
    """Parse TSV/CSV topics with auto-detected delimiter."""
    import csv
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    topics = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            tid = (row.get("topic_id") or row.get("TopicID") or
                   row.get("qid") or row.get("id") or "").strip()
            q = (row.get("question") or row.get("title") or
                 row.get("query") or row.get("Question") or "").strip()
            if tid and q:
                topics.append((tid, q))
    return topics


def load_topics(path: Path) -> list[tuple[str, str]]:
    # Sniff first non-empty line to detect XML vs tabular
    first_line = next(l for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    if first_line.strip().startswith("<"):
        return load_topics_xml(path)
    return load_topics_tsv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", help="Topics file (XML or TSV/CSV)")
    parser.add_argument("--index", default=str(INDEX_DIR))
    parser.add_argument("--meta", default=str(META_FILE))
    parser.add_argument("--output", default=str(BASE / "data/runs/bm25s_run.txt"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hyde", default=None,
                        help="Path to HyDE queries JSON (data/processed/hyde_queries.json). "
                             "Uses hypothetical passages as BM25 queries instead of raw questions.")
    args = parser.parse_args()

    print(f"Loading BM25S index from {args.index} ...")
    retriever = bm25s.BM25.load(args.index, load_corpus=False)

    print(f"Loading passage metadata from {args.meta} ...")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages")

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running retrieval (top-{args.top_k}) ...")
    with output_path.open("w", encoding="utf-8") as fout:
        for topic_id, question in tqdm(topics, desc="Topics"):
            query_text = hyde_queries.get(topic_id, question)
            tokenized_q = bm25s.tokenize([query_text], stopwords="en", stemmer=None)
            results, scores = retriever.retrieve(tokenized_q, k=args.top_k)

            for rank, idx in enumerate(results[0], start=1):
                rec = meta[idx]
                text = rec["text_snippet"]  # already ≤200 chars
                fout.write(f"{topic_id};{rank};{rec['doc_id']};{text}\n")

    print(f"Run saved to {output_path}")


if __name__ == "__main__":
    main()
