#!/usr/bin/env python3
"""
Build BM25S index from organizer passage archives.

Streams directly from baseline_passages/*.tar.gz without extracting to disk.
Saves:
  data/processed/bm25s_index/   — BM25S index (bm25s native format)
  data/processed/passages_meta.pkl — list of {chunk_id, doc_id, text_snippet}

Usage:
    python scripts/build_bm25_index.py
    python scripts/build_bm25_index.py --wiki-only   # skip Wookieepedia
    python scripts/build_bm25_index.py --max-docs N  # limit for testing
"""

import argparse
import gzip
import io
import json
import pickle
import re
import tarfile
from pathlib import Path

import bm25s
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
ARCHIVES = {
    "wikipedia": BASE / "data/raw/baseline_passages/wikipedia_passages.tar.gz",
    "wookieepedia": BASE / "data/raw/baseline_passages/wookieepedia_passages.tar.gz",
}
INDEX_DIR = BASE / "data/processed/bm25s_index"
META_FILE = BASE / "data/processed/passages_meta.pkl"


def simple_tokenize(text: str) -> list[str]:
    """Fast whitespace+punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def stream_passages(archive_path: Path, max_docs: int | None = None):
    """Yield (chunk_id, doc_id, title, text) from a tar.gz of .jsonl.gz files."""
    docs_seen = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".jsonl.gz"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            with gzip.open(f, "rt", encoding="utf-8") as gz:
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    yield (
                        obj["chunk_id"],
                        obj["doc_id"],
                        obj.get("title", ""),
                        obj["text"],
                    )
            docs_seen += 1
            if max_docs and docs_seen >= max_docs:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-only", action="store_true",
                        help="Index only Wikipedia (skip Wookieepedia)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit documents per archive (for testing)")
    args = parser.parse_args()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

    archives = [("wikipedia", ARCHIVES["wikipedia"])]
    if not args.wiki_only:
        archives.append(("wookieepedia", ARCHIVES["wookieepedia"]))

    print("=== Pass 1: stream passages to build corpus ===")
    meta = []        # list of dicts: {chunk_id, doc_id, text_snippet}
    corpus_texts = []  # full texts for tokenization

    for source, path in archives:
        print(f"Reading {source} from {path.name} ...")
        for chunk_id, doc_id, title, text in tqdm(
            stream_passages(path, args.max_docs),
            desc=source,
            unit="passages",
        ):
            # Store only first 200 chars as snippet for submission output
            snippet = text[:200].replace(";", ",").replace("\n", " ").strip()
            meta.append({"chunk_id": chunk_id, "doc_id": doc_id, "text_snippet": snippet})
            # Use title + text for BM25 to boost title terms
            corpus_texts.append(f"{title} {text}" if title else text)

    print(f"\nTotal passages: {len(meta):,}")

    print("\n=== Pass 2: tokenize ===")
    tokenized = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=None, show_progress=True)
    del corpus_texts  # free memory

    print("\n=== Pass 3: build BM25S index ===")
    retriever = bm25s.BM25(method="bm25+", k1=1.5, b=0.75)
    retriever.index(tokenized, show_progress=True)

    print(f"\nSaving index to {INDEX_DIR} ...")
    retriever.save(str(INDEX_DIR))

    print(f"Saving metadata to {META_FILE} ...")
    with META_FILE.open("wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDone. {len(meta):,} passages indexed.")


if __name__ == "__main__":
    main()
