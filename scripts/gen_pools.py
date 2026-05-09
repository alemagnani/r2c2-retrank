#!/usr/bin/env python3
"""
Generate pre-rerank pool files for PG extraction.
Produces raw BM25/dense/hybrid retrieval results WITHOUT CE reranking,
so pg_extractor can access many more unique document IDs.

Usage:
    python scripts/gen_pools.py \
        --topics-file data/synthetic_val/synthetic_topics_val65.json \
        --output-dir data/runs
"""

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import Stemmer as PyStemmer
    _PORTER = PyStemmer.Stemmer("english")
    def porter_stem(tokens): return [_PORTER.stemWord(t) for t in tokens]
except ImportError:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    def porter_stem(tokens): return [_ps.stem(t) for t in tokens]

BASE = Path(__file__).resolve().parent.parent


def load_topics(topics_file: Path) -> dict[str, str]:
    content = topics_file.read_text(encoding="utf-8")
    # XML format: <qID>0001</qID> ... <q>question text</q>
    if content.strip().startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    # JSON format
    data = json.loads(content)
    topics = {}
    for rec in data:
        tid = rec.get("loop_topic_id") or rec.get("topic_id", "")
        q = rec.get("question", "")
        if tid and q:
            topics[tid] = q
    return topics


def tokenize(text, retriever, stem=False):
    raw = re.findall(r"[a-z0-9]+", text.lower())
    raw = [t for t in raw if len(t) > 1]
    if stem:
        raw = porter_stem(raw)
    return bm25s.tokenize([" ".join(raw)], stopwords="en", stemmer=None)


def bm25_retrieve(query, retriever_obj, meta, k):
    tok = tokenize(query, retriever_obj)
    results, _ = retriever_obj.retrieve(tok, k=min(k, len(meta)))
    return [(i + 1, meta[idx]["doc_id"], meta[idx]["text_snippet"])
            for i, idx in enumerate(results[0])]


def dense_retrieve(query, faiss_index, dense_model, meta, k):
    emb = dense_model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
    scores, indices = faiss_index.search(emb, k)
    return [(i + 1, meta[idx]["doc_id"], meta[idx]["text_snippet"])
            for i, idx in enumerate(indices[0]) if idx < len(meta)]


def rrf_merge(lists, k=60):
    scores = defaultdict(float)
    for lst in lists:
        for rank, doc_id, text in lst:
            scores[(doc_id, text)] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(i + 1, doc_id, text) for i, ((doc_id, text), _) in enumerate(ranked)]


def write_pool(results, output_path, pool_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for tid in sorted(results.keys()):
            passages = results[tid][:pool_size]
            for rank, (_, doc_id, text) in enumerate(passages, 1):
                text = text[:200].replace("\n", " ").replace(";", ",")
                f.write(f"{tid};{rank};{doc_id};{text}\n")
    n_unique_docs = len({doc_id for lst in results.values() for _, doc_id, _ in lst[:pool_size]})
    print(f"  Written {output_path.name}: "
          f"{sum(min(len(v), pool_size) for v in results.values())} passages, "
          f"~{n_unique_docs} unique doc IDs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file", default=None, help="JSON topics file")
    parser.add_argument("--topics", default=None, help="XML topics file (r2c2topics.txt)")
    parser.add_argument("--output-dir", default=str(BASE / "data/runs"))
    parser.add_argument("--bm25-index", default=str(BASE / "data/processed/bm25s_index"))
    parser.add_argument("--meta", default=str(BASE / "data/processed/passages_meta.pkl"))
    parser.add_argument("--faiss-index", default=str(BASE / "data/processed/faiss_index.bin"))
    parser.add_argument("--dense-model", default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    parser.add_argument("--bm25-pool", type=int, default=500, help="BM25 pool size")
    parser.add_argument("--dense-pool", type=int, default=200, help="Dense pool size")
    parser.add_argument("--hybrid-pool", type=int, default=500, help="Hybrid pool size (after RRF)")
    args = parser.parse_args()

    topics_path = Path(args.topics_file) if args.topics_file else Path(args.topics)
    topics = load_topics(topics_path)
    print(f"Loaded {len(topics)} topics")

    output_dir = Path(args.output_dir)

    # Load BM25
    print("Loading BM25 index...")
    retriever_obj = bm25s.BM25.load(args.bm25_index, load_corpus=False)
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages")

    # Load dense
    faiss_index = None
    dense_model = None
    if Path(args.faiss_index).exists():
        print("Loading FAISS index...")
        faiss_index = faiss.read_index(args.faiss_index)
        print(f"Loading dense model: {args.dense_model}...")
        dense_model = SentenceTransformer(args.dense_model)
        print("  Dense ready")
    else:
        print("  No FAISS index found, skipping dense pools")

    # Generate BM25 pool (top-500, no CE)
    print(f"\nGenerating BM25 pool (top-{args.bm25_pool}, no CE)...")
    bm25_results = {}
    for tid, question in sorted(topics.items()):
        passages = bm25_retrieve(question, retriever_obj, meta, args.bm25_pool)
        bm25_results[tid] = passages
    tag = "val65" if (args.topics_file and "val65" in args.topics_file) else "real"
    fname = f"pool_bm25_{args.bm25_pool}_{tag}.txt"
    write_pool(bm25_results, output_dir / fname, args.bm25_pool)

    if faiss_index is not None:
        # Generate dense pool (top-200, no CE)
        print(f"\nGenerating dense pool (top-{args.dense_pool}, no CE)...")
        dense_results = {}
        for tid, question in sorted(topics.items()):
            passages = dense_retrieve(question, faiss_index, dense_model, meta, args.dense_pool)
            dense_results[tid] = passages
        fname = f"pool_dense_{args.dense_pool}_{tag}.txt"
        write_pool(dense_results, output_dir / fname, args.dense_pool)

        # Generate hybrid pool (BM25 top-500 + dense top-200 via RRF, no CE)
        print(f"\nGenerating hybrid pool (BM25-{args.bm25_pool} + dense-{args.dense_pool} RRF, no CE)...")
        hybrid_results = {}
        for tid in sorted(topics.keys()):
            merged = rrf_merge([bm25_results[tid], dense_results[tid]])
            hybrid_results[tid] = merged
        fname = f"pool_hybrid_{args.hybrid_pool}_{tag}.txt"
        write_pool(hybrid_results, output_dir / fname, args.hybrid_pool)

    print("\nDone.")


if __name__ == "__main__":
    main()
