#!/usr/bin/env python3
"""
Build BM25 + FAISS index over own sliding-window passages (200 chars, stride 100).

Reconstructs full document text from passages_meta.pkl, applies sliding-window
segmentation, and indexes the resulting ~3M passages. This replaces the two-phase
PG pipeline: retrieval now operates directly on our own passages.

Output:
  data/processed/own_passages/passages_meta.pkl  — own passage metadata
  data/processed/own_passages/bm25s_index/       — BM25S index
  data/processed/own_passages/faiss_index.bin    — FAISS IVF-PQ index (--dense only)

Usage:
    python scripts/build_own_index.py                  # BM25 only (fast)
    python scripts/build_own_index.py --dense          # BM25 + FAISS
    python scripts/build_own_index.py --dense-only     # FAISS only (if BM25 already built)
"""

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import bm25s
import numpy as np
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
SRC_META = BASE / "data/processed/passages_meta.pkl"
OUT_DIR = BASE / "data/processed/own_passages"

MAX_LEN = 200
STRIDE = 100
MIN_LEN = 50


try:
    import Stemmer as PyStemmer
    _PORTER = PyStemmer.Stemmer("english")
    def porter_stem(tokens): return [_PORTER.stemWord(t) for t in tokens]
except ImportError:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    def porter_stem(tokens): return [_ps.stem(t) for t in tokens]


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_window_passages(text: str, max_len: int = MAX_LEN,
                             stride: int = STRIDE, min_len: int = MIN_LEN) -> list[str]:
    if not text:
        return []
    words = text.split()
    passages = []
    stride_words = max(1, stride // 6)
    i = 0
    while i < len(words):
        chunk, length, j = [], 0, i
        while j < len(words) and length + len(words[j]) + 1 <= max_len:
            chunk.append(words[j])
            length += len(words[j]) + 1
            j += 1
        if j == i:
            j = i + 1
        p = " ".join(chunk).strip()
        if len(p) >= min_len:
            passages.append(p)
        i += stride_words
    return list(dict.fromkeys(passages))  # deduplicate adjacent identical


# ── Doc text reconstruction ───────────────────────────────────────────────────

def build_doc_texts(src_meta: Path) -> dict[str, str]:
    print(f"Loading {src_meta} ...")
    with open(src_meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} organizer passages")

    by_doc: dict[str, list] = defaultdict(list)
    for m in meta:
        by_doc[m["doc_id"]].append(m)

    doc_texts = {}
    for doc_id, chunks in by_doc.items():
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_id"])
        doc_texts[doc_id] = " ".join(c["text_snippet"] for c in sorted_chunks)

    print(f"  {len(doc_texts):,} unique documents")
    return doc_texts


# ── Extract own passages ──────────────────────────────────────────────────────

def extract_own_passages(doc_texts: dict[str, str]) -> list[dict]:
    own = []
    print(f"Extracting sliding-window passages (max={MAX_LEN}, stride={STRIDE}) ...")
    for doc_id, text in tqdm(doc_texts.items(), unit="doc"):
        passages = sliding_window_passages(text)
        for i, p in enumerate(passages):
            own.append({
                "chunk_id": f"{doc_id}__sw_{i:05d}",
                "doc_id": doc_id,
                "text_snippet": p,
            })
    print(f"  {len(own):,} own passages from {len(doc_texts):,} docs")
    return own


# ── BM25 index ────────────────────────────────────────────────────────────────

def build_bm25(passages: list[dict], out_dir: Path):
    print("Tokenizing for BM25 ...")
    texts = [p["text_snippet"] for p in passages]

    # Batch tokenize
    batch_size = 50_000
    all_tokens = []
    for start in tqdm(range(0, len(texts), batch_size), unit="batch"):
        batch = texts[start:start + batch_size]
        raw = [re.findall(r"[a-z0-9]+", t.lower()) for t in batch]
        raw = [[w for w in toks if len(w) > 1] for toks in raw]
        joined = [" ".join(toks) for toks in raw]
        tok = bm25s.tokenize(joined, stopwords="en", stemmer=None)
        all_tokens.extend(tok.ids[0] if hasattr(tok, 'ids') else tok)

    # Actually bm25s.tokenize returns a tokenized object — use it directly
    print("Building BM25S index ...")
    all_joined = [" ".join(re.findall(r"[a-z0-9]+", t.lower())) for t in texts]
    # Filter single-char tokens
    all_joined = [
        " ".join(w for w in s.split() if len(w) > 1)
        for s in all_joined
    ]

    corpus_tokens = bm25s.tokenize(all_joined, stopwords="en", stemmer=None)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    bm25_dir = out_dir / "bm25s_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(bm25_dir))
    print(f"  BM25 index saved to {bm25_dir}")


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss(passages: list[dict], out_dir: Path, model_name: str, batch_size: int):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(16)

    print(f"Loading bi-encoder: {model_name} ...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    n = len(passages)
    print(f"  dim={dim}, passages={n:,}")

    texts = [p["text_snippet"] for p in passages]

    print(f"Encoding {n:,} passages in batches of {batch_size} ...")
    all_embs = []
    for start in tqdm(range(0, n, batch_size), unit="batch"):
        batch = texts[start:start + batch_size]
        emb = model.encode(batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(emb.astype(np.float32))
    embeddings = np.vstack(all_embs)
    print(f"  Encoded: {embeddings.shape}")

    # IVF-PQ: ~0.2 GB for 3M passages vs 4.6 GB flat
    nlist = 8192
    m_pq = 64        # subvectors (384 / 64 = 6 dims each)
    nbits = 8

    print(f"Building FAISS IVF-PQ index (nlist={nlist}, m={m_pq}, nbits={nbits}) ...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits)

    # Train on up to 500K random samples
    n_train = min(500_000, n)
    idx_train = np.random.choice(n, n_train, replace=False)
    print(f"  Training on {n_train:,} samples ...")
    index.train(embeddings[idx_train])

    print(f"  Adding {n:,} vectors ...")
    chunk = 100_000
    for start in tqdm(range(0, n, chunk), unit="chunk"):
        index.add(embeddings[start:start + chunk])

    index.nprobe = 64  # search 64 of 8192 clusters (good recall/speed tradeoff)

    out_path = out_dir / "faiss_index.bin"
    faiss.write_index(index, str(out_path))
    size_gb = out_path.stat().st_size / 1e9
    print(f"  FAISS index saved to {out_path} ({size_gb:.2f} GB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-meta", default=str(SRC_META))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--dense", action="store_true", help="Also build FAISS index")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25, only build FAISS")
    parser.add_argument("--model", default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = out_dir / "passages_meta.pkl"

    # Step 1: Load or build own passage list
    if meta_out.exists():
        print(f"Loading existing own passages from {meta_out} ...")
        with open(meta_out, "rb") as f:
            own_passages = pickle.load(f)
        print(f"  {len(own_passages):,} passages")
    else:
        doc_texts = build_doc_texts(Path(args.src_meta))
        own_passages = extract_own_passages(doc_texts)
        print(f"Saving own passages metadata ...")
        with open(meta_out, "wb") as f:
            pickle.dump(own_passages, f)
        print(f"  Saved {len(own_passages):,} passages to {meta_out}")

    # Step 2: BM25
    if not args.dense_only:
        bm25_dir = out_dir / "bm25s_index"
        if bm25_dir.exists():
            print(f"BM25 index already exists at {bm25_dir}, skipping.")
        else:
            build_bm25(own_passages, out_dir)

    # Step 3: FAISS
    if args.dense or args.dense_only:
        faiss_path = out_dir / "faiss_index.bin"
        if faiss_path.exists():
            print(f"FAISS index already exists at {faiss_path}, skipping.")
        else:
            build_faiss(own_passages, out_dir, args.model, args.batch_size)

    print("\nDone.")
    print(f"Output dir: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*"))
        print(f"  {p.name}: {size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
