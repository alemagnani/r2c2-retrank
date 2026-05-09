#!/usr/bin/env python3
"""
BM25 retrieval with Pseudo-Relevance Feedback (PRF) query expansion.

Algorithm (RM3-style):
  1. First-pass BM25 retrieval with original query (top-fb_docs passages)
  2. Extract top-fb_terms expansion terms from those passages by
     P(t|R) * log(P(t|R)/P(t|C)) — term probability in feedback set
     weighted by inverse corpus frequency (like RM3 but simplified)
  3. Form expanded query: original terms (weight alpha) + expansion terms
  4. Second-pass BM25 retrieval with expanded query

No API calls, no models — runs in seconds.

Usage:
    python scripts/run_prf_retrieval.py data/raw/r2c2topics.txt \
        --output data/runs/bm25_prf.txt --top-k 20

    # Tune PRF parameters:
    python scripts/run_prf_retrieval.py data/raw/r2c2topics.txt \
        --fb-docs 10 --fb-terms 20 --alpha 0.5
"""

import argparse
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import bm25s
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE / "data/processed/bm25s_index"
META_FILE = BASE / "data/processed/passages_meta.pkl"

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","that","this","these",
    "those","it","its","from","by","as","not","what","who","which","how","when",
    "where","why","whose","whom","than","then","there","their","they","we","i",
    "he","she","you","me","him","her","us","them","my","your","his","our","its",
}


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


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 2]


def extract_expansion_terms(passages: list[str], query_tokens: set[str],
                             fb_terms: int, vocab_idf: dict[str, float]) -> list[str]:
    """
    Extract top-fb_terms expansion terms from feedback passages.
    Score each term by: tf_in_feedback * idf_in_corpus
    Excludes query terms already present.
    """
    term_freq = Counter()
    total_tokens = 0
    for text in passages:
        tokens = tokenize(text)
        term_freq.update(tokens)
        total_tokens += len(tokens)

    if total_tokens == 0:
        return []

    # Score = normalised TF in feedback docs * IDF in corpus
    scored = {}
    for term, freq in term_freq.items():
        if term in query_tokens:
            continue
        tf = freq / total_tokens
        idf = vocab_idf.get(term, 0.0)
        scored[term] = tf * idf

    return [t for t, _ in sorted(scored.items(), key=lambda x: -x[1])[:fb_terms]]


def build_vocab_idf(meta: list[dict]) -> dict[str, float]:
    """Compute IDF over corpus for PRF scoring. Uses passage text_snippets."""
    print("Building vocab IDF for PRF scoring ...")
    df = Counter()
    N = len(meta)
    for rec in meta:
        for tok in set(tokenize(rec["text_snippet"])):
            df[tok] += 1
    return {term: math.log((N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", help="Topics file (XML or TSV/CSV)")
    parser.add_argument("--index", default=str(INDEX_DIR))
    parser.add_argument("--meta", default=str(META_FILE))
    parser.add_argument("--output", default=str(BASE / "data/runs/bm25_prf.txt"))
    parser.add_argument("--top-k", type=int, default=20, help="Final passages to return")
    parser.add_argument("--fb-docs", type=int, default=10,
                        help="Number of top passages used as feedback (default: 10)")
    parser.add_argument("--fb-terms", type=int, default=15,
                        help="Number of expansion terms to add (default: 15)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of original query terms vs expansion terms (default: 0.5)")
    args = parser.parse_args()

    print(f"Loading BM25S index from {args.index} ...")
    retriever = bm25s.BM25.load(args.index, load_corpus=False)

    print(f"Loading passage metadata from {args.meta} ...")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages")

    vocab_idf = build_vocab_idf(meta)

    print(f"Loading topics from {args.topics} ...")
    topics = load_topics(Path(args.topics))
    print(f"  {len(topics)} topics")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running PRF retrieval "
          f"(fb_docs={args.fb_docs}, fb_terms={args.fb_terms}, alpha={args.alpha}) ...")

    with output_path.open("w", encoding="utf-8") as fout:
        for topic_id, question in tqdm(topics, desc="Topics"):
            query_tokens = tokenize(question)
            query_token_set = set(query_tokens)

            # Pass 1: retrieve feedback documents
            tok1 = bm25s.tokenize([question], stopwords="en", stemmer=None)
            fb_results, _ = retriever.retrieve(tok1, k=args.fb_docs)
            fb_texts = [meta[idx]["text_snippet"] for idx in fb_results[0]]

            # Extract expansion terms
            expansion = extract_expansion_terms(
                fb_texts, query_token_set, args.fb_terms, vocab_idf
            )

            # Build expanded query: weight original terms by alpha, expansions by (1-alpha)
            original_part = " ".join(query_tokens * max(1, round(args.alpha * 10)))
            expansion_part = " ".join(expansion * max(1, round((1 - args.alpha) * 10)))
            expanded_query = f"{original_part} {expansion_part}".strip()

            # Pass 2: retrieve with expanded query
            tok2 = bm25s.tokenize([expanded_query], stopwords="en", stemmer=None)
            results, _ = retriever.retrieve(tok2, k=args.top_k)

            for rank, idx in enumerate(results[0], start=1):
                rec = meta[idx]
                fout.write(f"{topic_id};{rank};{rec['doc_id']};{rec['text_snippet']}\n")

    print(f"Run saved to {output_path}")


if __name__ == "__main__":
    main()
