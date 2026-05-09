#!/usr/bin/env python3
"""
Build a FAISS dense index from passage metadata using a bi-encoder.

Embeds the stored 200-char passage snippets (sufficient for MiniLM-scale models).
Saves:
  data/processed/faiss_index.bin  — FAISS flat inner-product index
  (reuses data/processed/passages_meta.pkl for doc_id / text_snippet)

Usage:
    python scripts/build_dense_index.py
    python scripts/build_dense_index.py --model sentence-transformers/msmarco-MiniLM-L6-cos-v5
    python scripts/build_dense_index.py --batch-size 512
"""

import argparse
import pickle
from pathlib import Path

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Use all available CPU cores for inference
torch.set_num_threads(16)

BASE = Path(__file__).resolve().parent.parent
META_FILE = BASE / "data/processed/passages_meta.pkl"
FAISS_INDEX = BASE / "data/processed/faiss_index.bin"

DEFAULT_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--meta", default=str(META_FILE))
    parser.add_argument("--output", default=str(FAISS_INDEX))
    args = parser.parse_args()

    print(f"Loading passage metadata from {args.meta} ...")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages")

    texts = [m["text_snippet"] for m in meta]

    print(f"Loading bi-encoder: {args.model} ...")
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")

    print(f"Encoding {len(texts):,} passages (batch_size={args.batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine sim via inner product
    )
    embeddings = embeddings.astype(np.float32)

    print("Building FAISS index (flat inner product) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal:,} vectors")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"Index saved to {output_path}")


if __name__ == "__main__":
    main()
