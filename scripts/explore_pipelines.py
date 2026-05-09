#!/usr/bin/env python3
"""
Pipeline combination explorer for R2C2 passage retrieval.

Systematically explores ~50 pipeline configurations on val65 synthetic topics
to identify the 2-3 best retrieval combinations.

Each configuration is a complete pipeline spec:
  • query_variants : which query versions to use (original / reform / multi_query / prf)
  • bm25_params    : k1, b, stem, lemma
  • use_dense      : include FAISS bi-encoder results
  • pool_size      : candidates per retrieval source before reranking
  • fusion         : rrf | concat
  • rerank         : none | ce

Exploration strategy:
  • Configs  1-15 : systematic seed grid (single components, then 2-way combos)
  • Configs 16-50 : LLM hypothesis-guided — after every 5 results Claude proposes
                    the next 5 configs based on what's working

Usage:
    python scripts/explore_pipelines.py \\
        --topics-file  data/synthetic_val/synthetic_topics_val65.json \\
        --loop-state   data/synthetic_autoresearch_val65/loop_state.json \\
        --qrels        data/synthetic_autoresearch_val65/qrels_loop_claude_haiku_4_5_20251001.json \\
        --output-dir   data/pipeline_exploration \\
        --model        claude-haiku-4-5-20251001 \\
        --max-configs  50
"""

import argparse
import json
import math
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import anthropic
import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import Stemmer as PyStemmer
    _PORTER = PyStemmer.Stemmer("english")
    def porter_stem(tokens): return [_PORTER.stemWord(t) for t in tokens]
except ImportError:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    def porter_stem(tokens): return [_ps.stem(t) for t in tokens]

try:
    from nltk.stem import WordNetLemmatizer
    _lemma = WordNetLemmatizer()
    def wordnet_lemmatize(tokens): return [_lemma.lemmatize(t) for t in tokens]
except Exception:
    def wordnet_lemmatize(tokens): return tokens

BASE = Path(__file__).resolve().parent.parent


# ── Pipeline config ────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Complete pipeline specification. All fields are hashable primitives."""
    name: str
    hypothesis: str
    # Query variants to generate and retrieve for
    use_original:    bool = True
    use_reform:      bool = False   # disabled: LLM query rewriting excluded (generative retrieval bias)
    use_multi_query: bool = False   # disabled: LLM sub-query decomposition excluded
    use_prf:         bool = False   # pseudo-relevance feedback expansion (term-based, no LLM)
    # BM25 parameters
    bm25_k1:   float = 1.5
    bm25_b:    float = 0.75
    bm25_stem: bool  = False
    bm25_lemma: bool = False
    # Dense retrieval
    use_dense: bool = False
    # Candidate pool per retrieval source
    pool_size: int = 100
    # Fusion method when merging multiple retrieval lists
    fusion: str = "rrf"    # "rrf" | "concat"
    # Cross-encoder reranking
    rerank: bool = True
    # Include the autoresearch per-topic best run as an additional retrieval source
    use_loop_run: bool = False
    # Final output size
    final_k: int = 20

    def to_key(self) -> str:
        """Short unique key for logging / deduplication."""
        parts = []
        if not self.use_original: parts.append("no-orig")
        if self.use_loop_run:    parts.append("loop")
        if self.use_reform:      parts.append("reform")
        if self.use_multi_query: parts.append("mq")
        if self.use_prf:         parts.append("prf")
        if self.use_dense:       parts.append("dense")
        if self.bm25_k1 != 1.5: parts.append(f"k1={self.bm25_k1}")
        if self.bm25_b  != 0.75: parts.append(f"b={self.bm25_b}")
        if self.bm25_stem:       parts.append("stem")
        if self.bm25_lemma:      parts.append("lemma")
        parts.append(f"pool={self.pool_size}")
        parts.append(self.fusion)
        parts.append("ce" if self.rerank else "no-ce")
        return "|".join(parts) if parts else "bm25_default"


# ── Seed configurations (systematic coverage) ─────────────────────────────────

SEED_CONFIGS: list[PipelineConfig] = [
    # NOTE: use_reform and use_multi_query are permanently disabled — they rely
    # on LLM-generated query rewriting which constitutes generative retrieval
    # (the LLM uses its own knowledge to answer, not the corpus).
    # All configs here use only the original query + BM25/dense retrieval.

    # ── BM25 baselines ───────────────────────────────────────────────
    PipelineConfig("bm25_base",
        "Baseline: BM25 top-100 with cross-encoder rerank.",
        pool_size=100, rerank=True),

    PipelineConfig("bm25_pool200",
        "Larger pool: BM25 top-200 improves recall before CE rerank.",
        pool_size=200, rerank=True),

    PipelineConfig("bm25_pool500",
        "Even larger pool: 500 candidates gives CE more to work with.",
        pool_size=500, rerank=True),

    PipelineConfig("bm25_no_rerank",
        "BM25 alone (no CE): first-stage ceiling without reranking.",
        pool_size=20, rerank=False),

    PipelineConfig("bm25_k1_high",
        "k1=2.5: higher TF saturation rewards repeated query terms.",
        bm25_k1=2.5, pool_size=200, rerank=True),

    PipelineConfig("bm25_k1_low",
        "k1=0.8: flatter TF lets IDF dominate — better for rare movie entities.",
        bm25_k1=0.8, pool_size=200, rerank=True),

    PipelineConfig("bm25_b_low",
        "b=0.2: less length normalisation — short factual passages get fairer chance.",
        bm25_b=0.2, pool_size=200, rerank=True),

    PipelineConfig("bm25_stem",
        "Porter stemming: morphological variants (direct/directed/director).",
        bm25_stem=True, pool_size=200, rerank=True),

    PipelineConfig("bm25_k1_low_pool500",
        "Low k1 with large pool: IDF-dominated retrieval over 500 candidates.",
        bm25_k1=0.8, pool_size=500, rerank=True),

    PipelineConfig("bm25_b_low_pool500",
        "Low length norm with large pool: 500 candidates, less penalisation of short passages.",
        bm25_b=0.2, pool_size=500, rerank=True),

    # ── Dense retrieval ──────────────────────────────────────────────
    PipelineConfig("dense_only",
        "Dense FAISS bi-encoder only: pure semantic retrieval, no BM25.",
        use_original=False, use_dense=True, pool_size=100, rerank=True),

    PipelineConfig("dense_pool200",
        "Dense bi-encoder with larger pool: 200 semantic candidates before CE rerank.",
        use_original=False, use_dense=True, pool_size=200, rerank=True),

    PipelineConfig("dense_pool500",
        "Dense bi-encoder with large pool: 500 semantic candidates before CE rerank.",
        use_original=False, use_dense=True, pool_size=500, rerank=True),

    PipelineConfig("dense_no_rerank",
        "Dense bi-encoder without CE rerank: measures pure semantic retrieval ceiling.",
        use_original=False, use_dense=True, pool_size=20, rerank=False),

    # ── Hybrid BM25 + dense ──────────────────────────────────────────
    PipelineConfig("bm25_plus_dense",
        "BM25 + dense RRF: classic hybrid covering lexical and semantic signals.",
        use_dense=True, pool_size=150, rerank=True),

    PipelineConfig("hybrid_pool200",
        "BM25 + dense RRF with pool=200: more hybrid candidates for CE to rerank.",
        use_dense=True, pool_size=200, rerank=True),

    PipelineConfig("hybrid_pool500",
        "BM25 + dense RRF with pool=500: maximum hybrid recall before CE rerank.",
        use_dense=True, pool_size=500, rerank=True),

    PipelineConfig("hybrid_k1_low",
        "Hybrid with k1=0.8: IDF-dominated BM25 + dense, merged via RRF.",
        use_dense=True, bm25_k1=0.8, pool_size=200, rerank=True),

    PipelineConfig("hybrid_b_low",
        "Hybrid with b=0.2: less length norm BM25 + dense, merged via RRF.",
        use_dense=True, bm25_b=0.2, pool_size=200, rerank=True),

    PipelineConfig("hybrid_no_rerank",
        "BM25 + dense RRF without CE: measures hybrid first-stage ceiling.",
        use_dense=True, pool_size=20, rerank=False),

    # ── PRF (term expansion, no LLM) ────────────────────────────────
    PipelineConfig("bm25_prf",
        "BM25 + pseudo-relevance feedback: term expansion from top-5 passages.",
        use_prf=True, pool_size=200, rerank=True),

    PipelineConfig("hybrid_prf",
        "Hybrid BM25+dense + PRF term expansion: combines all non-LLM signals.",
        use_dense=True, use_prf=True, pool_size=200, rerank=True),
]


# ── LLM helpers ───────────────────────────────────────────────────────────────

PROPOSE_PROMPT = """\
You are designing retrieval pipelines for a movie question-answering system.
The corpus contains short passages (≤200 chars) from Wikipedia and Wookieepedia.

We have explored {n_done} pipeline configurations so far. Here are the results
(sorted by mean nDCG@20, best first):

{results_table}

Available components (NO LLM query rewriting — pure retrieval only):
  use_original    : use the raw question for BM25 retrieval
  use_prf         : pseudo-relevance feedback term expansion (term-based, no LLM)
  use_dense       : FAISS bi-encoder dense retrieval (semantic similarity)
  bm25_k1         : TF saturation (0.8=flat/IDF-like, 1.5=default, 2.5=high)
  bm25_b          : length normalisation (0.2=low, 0.75=default, 0.9=high)
  bm25_stem       : Porter stemming on query tokens
  pool_size       : candidates before reranking (50 / 100 / 200 / 500)
  fusion          : "rrf" (reciprocal rank fusion) | "concat" (simple concatenation)
  rerank          : apply cross-encoder reranking on the candidate pool

Observations from results above (write your analysis first, then propose):
1. Which components clearly help?
2. Which combinations haven't been tried yet?
3. What hypothesis explains why the top configs work?

Then propose exactly 5 new pipeline configurations to try next.
For each, give:
  - A short name (snake_case, unique)
  - A one-sentence hypothesis
  - The configuration parameters (only list fields that differ from defaults:
    use_original=True, use_prf=False, use_dense=False,
    bm25_k1=1.5, bm25_b=0.75, bm25_stem=False,
    pool_size=100, fusion="rrf", rerank=True)
  - NEVER set use_reform=true or use_multi_query=true (LLM retrieval excluded)

Reply in JSON:
{{
  "analysis": "...",
  "proposed": [
    {{
      "name": "...",
      "hypothesis": "...",
      "use_original": true,
      "use_reform": false,
      "use_multi_query": false,
      "use_prf": false,
      "use_dense": false,
      "bm25_k1": 1.5,
      "bm25_b": 0.75,
      "bm25_stem": false,
      "pool_size": 100,
      "fusion": "rrf",
      "rerank": true
    }},
    ...
  ]
}}"""


def _parse_config(p: dict, fallback_name: str) -> PipelineConfig:
    return PipelineConfig(
        name=p.get("name", fallback_name),
        hypothesis=p.get("hypothesis", ""),
        use_original=p.get("use_original", True),
        use_loop_run=p.get("use_loop_run", False),
        use_reform=p.get("use_reform", False),
        use_multi_query=p.get("use_multi_query", False),
        use_prf=p.get("use_prf", False),
        use_dense=p.get("use_dense", False),
        bm25_k1=float(p.get("bm25_k1", 1.5)),
        bm25_b=float(p.get("bm25_b", 0.75)),
        bm25_stem=bool(p.get("bm25_stem", False)),
        pool_size=int(p.get("pool_size", 100)),
        fusion=p.get("fusion", "rrf"),
        rerank=bool(p.get("rerank", True)),
    )


# ── Random / evolutionary generators ─────────────────────────────────────────

_POOL_CHOICES  = [50, 100, 200, 500]
_K1_CHOICES    = [0.8, 1.2, 1.5, 2.0, 2.5]
_B_CHOICES     = [0.2, 0.5, 0.75, 0.9]
_BOOL_FIELDS   = ["use_original", "use_prf", "use_dense", "bm25_stem", "rerank"]
# use_reform and use_multi_query permanently excluded (LLM generative retrieval bias)
_NUMERIC_FIELDS = {"bm25_k1": _K1_CHOICES, "bm25_b": _B_CHOICES,
                   "pool_size": _POOL_CHOICES}


def random_config(rng: random.Random, n: int) -> PipelineConfig:
    """Sample a uniformly random pipeline config from the full space.
    Ensures at least one retrieval source. use_loop_run is never set True
    to avoid circular evaluation (loop passages evaluated on same topics)."""
    while True:
        d = {
            "use_original":    rng.random() < 0.7,
            "use_loop_run":    False,   # excluded: circular eval bias
            "use_reform":      False,   # excluded: LLM generative retrieval bias
            "use_multi_query": False,   # excluded: LLM generative retrieval bias
            "use_prf":         rng.random() < 0.3,
            "use_dense":       rng.random() < 0.5,
            "bm25_k1":         rng.choice(_K1_CHOICES),
            "bm25_b":          rng.choice(_B_CHOICES),
            "bm25_stem":       rng.random() < 0.2,
            "pool_size":       rng.choice(_POOL_CHOICES),
            "fusion":          rng.choice(["rrf", "rrf", "concat"]),  # rrf 2x likely
            "rerank":          rng.random() < 0.8,
        }
        if d["use_original"] or d["use_dense"]:
            break
    return PipelineConfig(name=f"rand_{n:03d}",
                          hypothesis="Random exploration — covering uncharted config space.",
                          **d)


def mutate_config(cfg: PipelineConfig, rng: random.Random, n: int) -> PipelineConfig:
    """Flip one random dimension of an existing config."""
    d = asdict(cfg)
    field = rng.choice(_BOOL_FIELDS + list(_NUMERIC_FIELDS.keys()))
    if field in _BOOL_FIELDS:
        d[field] = not d[field]
        hyp = f"Mutate '{field}' from {not d[field]} → {d[field]} on '{cfg.name}'."
    else:
        choices = [v for v in _NUMERIC_FIELDS[field] if v != d[field]]
        d[field] = rng.choice(choices) if choices else d[field]
        hyp = f"Mutate '{field}' to {d[field]} on '{cfg.name}'."
    # Never set LLM-based fields or loop_run; ensure at least one retrieval source
    d["use_loop_run"] = False
    d["use_reform"] = False
    d["use_multi_query"] = False
    if not (d["use_original"] or d["use_dense"]):
        d["use_original"] = True
    d.pop("name"); d.pop("hypothesis")
    return PipelineConfig(name=f"mut_{n:03d}", hypothesis=hyp, **d)


def crossover_configs(a: PipelineConfig, b: PipelineConfig,
                      rng: random.Random, n: int) -> PipelineConfig:
    """Take each field randomly from one of two parent configs."""
    da, db = asdict(a), asdict(b)
    d = {}
    for f in _BOOL_FIELDS:
        d[f] = da[f] if rng.random() < 0.5 else db[f]
    for f in _NUMERIC_FIELDS:
        d[f] = da[f] if rng.random() < 0.5 else db[f]
    d["fusion"] = da["fusion"] if rng.random() < 0.5 else db["fusion"]
    # Never set LLM-based fields or loop_run; ensure at least one retrieval source
    d["use_loop_run"] = False
    d["use_reform"] = False
    d["use_multi_query"] = False
    if not (d["use_original"] or d["use_dense"]):
        d["use_original"] = True
    hyp = f"Crossover of '{a.name}' × '{b.name}'."
    return PipelineConfig(name=f"cross_{n:03d}", hypothesis=hyp, **d)


def next_batch(results: list[dict], tried_keys: set, rng: random.Random,
               n_start: int, batch_size: int = 5,
               greedy_frac: float = 0.4, mutate_frac: float = 0.3,
               cross_frac: float = 0.2) -> list[PipelineConfig]:
    """
    Generate the next batch of configs using a mix of strategies:
      greedy_frac  — mutate one of the TOP configs (exploit best known)
      mutate_frac  — mutate ANY config sampled by rank-weighted probability (explore)
      cross_frac   — crossover two configs at DIFFERENT rank levels
      remainder    — pure random

    Rank-weighted sampling uses softmax over nDCG scores so that lower-ranked
    configs still get selected with nonzero probability.
    """
    sorted_r = sorted(results, key=lambda x: -x["mean_ndcg"])
    configs_by_key = {r["key"]: _parse_config(r["config"], r["name"]) for r in sorted_r}
    scores = np.array([r["mean_ndcg"] for r in sorted_r], dtype=float)
    # Softmax temperature = 0.1 → fairly uniform; higher → more greedy
    temp = 0.05
    weights = np.exp(scores / max(temp, scores.max() * temp))
    weights /= weights.sum()

    top_configs = [configs_by_key[r["key"]] for r in sorted_r[:5]]

    batch: list[PipelineConfig] = []
    batch_keys: set[str] = set()  # local dedup — do NOT mutate tried_keys here
    n = n_start

    def add(cfg):
        nonlocal n
        key = cfg.to_key()
        if key not in tried_keys and key not in batch_keys and len(batch) < batch_size:
            batch.append(cfg)
            batch_keys.add(key)
            n += 1

    targets = {
        "greedy": max(1, int(batch_size * greedy_frac)),
        "mutate": max(1, int(batch_size * mutate_frac)),
        "cross":  max(1, int(batch_size * cross_frac)),
    }
    remaining = batch_size - sum(targets.values())

    # Greedy: mutate top configs
    for _ in range(targets["greedy"] * 3):
        if sum(1 for t in ["greedy"] if True) and len([x for x in batch if "mut" in x.name or x.name in [c.name for c in top_configs]]) < targets["greedy"]:
            src = rng.choice(top_configs)
            add(mutate_config(src, rng, n))

    # Mutate: rank-weighted sampling (any config, not just top)
    for _ in range(targets["mutate"] * 3):
        idx = rng.choices(range(len(sorted_r)), weights=weights)[0]
        src = configs_by_key[sorted_r[idx]["key"]]
        add(mutate_config(src, rng, n))

    # Crossover: one parent from top-5, one from anywhere (different rank tier)
    for _ in range(targets["cross"] * 3):
        a = rng.choice(top_configs)
        idx = rng.choices(range(len(sorted_r)), weights=weights)[0]
        b = configs_by_key[sorted_r[idx]["key"]]
        if a.to_key() != b.to_key():
            add(crossover_configs(a, b, rng, n))

    # Random fill
    for _ in range((remaining + batch_size) * 3):
        if len(batch) >= batch_size:
            break
        add(random_config(rng, n))

    return batch[:batch_size]


def llm_propose(client, model: str, results: list[dict],
                tried_keys: set) -> tuple[str, list[PipelineConfig]]:
    """Ask LLM to propose configs, but show the FULL ranked table so it can
    reason about non-top configs too."""
    ranked = sorted(results, key=lambda x: -x["mean_ndcg"])
    rows = "\n".join(
        f"  {i+1:2d}. {r['name']:<28} nDCG={r['mean_ndcg']:.4f}  key={r['key']}"
        for i, r in enumerate(ranked)
    )
    prompt = PROPOSE_PROMPT.format(n_done=len(results), results_table=rows)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model, max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text.strip()
            text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(text)
            analysis = data.get("analysis", "")
            proposed = []
            for i, p in enumerate(data.get("proposed", [])):
                cfg = _parse_config(p, f"llm_{len(results)+i:03d}")
                if cfg.to_key() not in tried_keys:
                    proposed.append(cfg)
                    tried_keys.add(cfg.to_key())
            return analysis, proposed
        except Exception as e:
            print(f"  LLM proposal error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return "", []


# ── Core retrieval infrastructure ──────────────────────────────────────────────

class Retriever:
    def __init__(self, bm25_dir, meta_file, faiss_path, dense_model_name,
                 ce_model_name, lora_model_path=None):
        print("Loading BM25S index ...")
        self.retriever = bm25s.BM25.load(bm25_dir, load_corpus=False)
        with open(meta_file, "rb") as f:
            self.meta = pickle.load(f)
        print(f"  {len(self.meta):,} passages")

        self.faiss_index = None
        self.dense_model = None
        if Path(faiss_path).exists():
            print("Loading FAISS index ...")
            self.faiss_index = faiss.read_index(faiss_path)
            print(f"Loading dense model: {dense_model_name} ...")
            self.dense_model = SentenceTransformer(dense_model_name)
            print("  Dense retriever ready")

        ce_path = lora_model_path if (lora_model_path and Path(lora_model_path).exists()) else ce_model_name
        print(f"Loading cross-encoder: {ce_path} ...")
        self.cross_encoder = CrossEncoder(ce_path)
        print("  Cross-encoder ready")

    def _tokenize(self, text, stem=False, lemma=False):
        raw = re.findall(r"[a-z0-9]+", text.lower())
        raw = [t for t in raw if len(t) > 1]
        if stem:  raw = porter_stem(raw)
        if lemma: raw = wordnet_lemmatize(raw)
        return bm25s.tokenize([" ".join(raw)], stopwords="en", stemmer=None)

    def bm25_retrieve(self, query: str, k: int, k1: float = 1.5, b: float = 0.75,
                      stem: bool = False, lemma: bool = False) -> list[tuple]:
        # Temporarily patch BM25 params if different from loaded index
        orig_k1, orig_b = self.retriever.k1, self.retriever.b
        self.retriever.k1, self.retriever.b = k1, b
        tok = self._tokenize(query, stem=stem, lemma=lemma)
        try:
            results, _ = self.retriever.retrieve(tok, k=min(k, len(self.meta)))
        finally:
            self.retriever.k1, self.retriever.b = orig_k1, orig_b
        return [(i + 1, self.meta[idx]["doc_id"], self.meta[idx]["text_snippet"])
                for i, idx in enumerate(results[0])]

    def dense_retrieve(self, query: str, k: int) -> list[tuple]:
        if self.faiss_index is None or self.dense_model is None:
            return []
        emb = self.dense_model.encode([query], normalize_embeddings=True,
                                       convert_to_numpy=True).astype(np.float32)
        scores, indices = self.faiss_index.search(emb, k)
        return [(i + 1, self.meta[idx]["doc_id"], self.meta[idx]["text_snippet"])
                for i, idx in enumerate(indices[0]) if idx < len(self.meta)]

    def prf_expand(self, query: str, fb_docs: int = 5, fb_terms: int = 10) -> str:
        stopwords = {"a","an","the","and","or","but","in","on","at","to","for","of",
                     "with","is","was","are","were","be","been","have","has","had",
                     "do","does","did","will","would","could","should","that","this",
                     "it","from","by","as","not","what","who","which","how","when"}
        top = self.bm25_retrieve(query, k=fb_docs)
        term_freq: dict[str, float] = defaultdict(float)
        for _, _, text in top:
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                if w not in stopwords and len(w) > 2:
                    term_freq[w] += 1.0
        expansion = " ".join(t for t, _ in
                              sorted(term_freq.items(), key=lambda x: -x[1])[:fb_terms])
        return f"{query} {expansion}"

    def rerank(self, query: str, candidates: list[tuple], final_k: int) -> list[tuple]:
        if not candidates:
            return []
        pairs = [(query, text) for _, _, text in candidates]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [(i + 1, doc_id, text) for i, (_, (_, doc_id, text)) in enumerate(ranked[:final_k])]


def rrf_merge(lists: list[list[tuple]], k: int = 60) -> list[tuple]:
    scores: dict[tuple, float] = defaultdict(float)
    for lst in lists:
        for rank, doc_id, text in lst:
            scores[(doc_id, text)] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(i + 1, doc_id, text) for i, ((doc_id, text), _) in enumerate(ranked)]


def dedup_passages(passages: list, overlap_threshold: float = 0.3,
                   max_per_doc: int = 3) -> list:
    seen_by_doc: dict[str, list[set]] = defaultdict(list)
    result = []
    for _, doc_id, text in passages:
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        selected = seen_by_doc[doc_id]
        if len(selected) >= max_per_doc:
            continue
        if any(len(words & prev) / max(len(words | prev), 1) > overlap_threshold
               for prev in selected):
            continue
        selected.append(words)
        result.append((len(result) + 1, doc_id, text))
    return result


# ── Run a single pipeline config on all topics ────────────────────────────────

def run_config(cfg: PipelineConfig, topics: dict, retriever: Retriever,
               reform_cache: dict, multi_query_cache: dict,
               loop_run: dict | None = None) -> dict[str, list[tuple]]:
    """Execute a pipeline config for all topics. Returns {tid: passages}."""
    results = {}
    for tid, question in topics.items():
        lists = []

        # Include autoresearch per-topic best passages as a retrieval source
        if cfg.use_loop_run and loop_run and tid in loop_run:
            lists.append(loop_run[tid])

        # Collect queries
        queries_to_run = []
        if cfg.use_original:
            q = question
            if cfg.use_prf:
                q = retriever.prf_expand(q)
            queries_to_run.append(q)

        if cfg.use_reform and reform_cache.get(tid):
            rf = reform_cache[tid]
            # Multi-query is encoded as "Q1 ||| Q2" in reform cache
            if "|||" not in rf:
                queries_to_run.append(rf)

        if cfg.use_multi_query and multi_query_cache.get(tid):
            for sq in multi_query_cache[tid]:
                queries_to_run.append(sq)

        # Retrieve for each query (BM25 + optionally dense)
        for q in queries_to_run:
            bm25_results = retriever.bm25_retrieve(
                q, k=cfg.pool_size,
                k1=cfg.bm25_k1, b=cfg.bm25_b,
                stem=cfg.bm25_stem, lemma=cfg.bm25_lemma,
            )
            lists.append(bm25_results)

            if cfg.use_dense:
                dense_results = retriever.dense_retrieve(q, k=cfg.pool_size)
                lists.append(dense_results)

        # Dense-only mode: use_dense=True but no BM25 queries (use_original=False)
        if cfg.use_dense and not queries_to_run:
            dense_results = retriever.dense_retrieve(question, k=cfg.pool_size)
            lists.append(dense_results)

        if not lists:
            results[tid] = []
            continue

        # Fuse
        if len(lists) == 1:
            candidates = lists[0]
        elif cfg.fusion == "rrf":
            candidates = rrf_merge(lists)
        else:  # concat — deduplicate by (doc_id, text)
            seen = set()
            candidates = []
            for lst in lists:
                for rank, doc_id, text in lst:
                    if (doc_id, text) not in seen:
                        seen.add((doc_id, text))
                        candidates.append((len(candidates) + 1, doc_id, text))

        candidates = candidates[:cfg.pool_size * len(lists)]

        # Rerank
        if cfg.rerank:
            passages = retriever.rerank(question, candidates, cfg.final_k)
        else:
            passages = [(i + 1, doc_id, text)
                        for i, (_, doc_id, text) in enumerate(candidates[:cfg.final_k])]

        results[tid] = dedup_passages(passages)

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def ndcg(rels: list[int], k: int = 20) -> float:
    def dcg(r, k):
        return sum(v / math.log2(i + 2) for i, v in enumerate(r[:k]))
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def score_passages(passages: list[tuple], qrels: dict, tid: str) -> float:
    rels = [qrels.get(tid, {}).get((doc_id, text), 0) for _, doc_id, text in passages]
    return ndcg(rels)


def judge_new_passages(client, model, topics, qrels, passages_by_topic,
                       cache_path, min_interval=1.3):
    to_judge = [
        (tid, doc_id, text)
        for tid, passages in passages_by_topic.items()
        for _, doc_id, text in passages
        if (doc_id, text) not in qrels.get(tid, {})
    ]
    if not to_judge:
        return
    print(f"  Judging {len(to_judge)} new passages ...")
    JUDGE = ("You are an IR judge. Rate 0/1/2 (0=not relevant, 1=partially, "
             "2=highly relevant).\nQuestion: {q}\nPassage: {p}\nReply ONLY with 0, 1, or 2.")

    def _save():
        serialized = {
            tid: {f"{doc_id}\x00{text}": v for (doc_id, text), v in p.items()}
            for tid, p in qrels.items()
        }
        with cache_path.open("w") as f:
            json.dump(serialized, f)

    for i, (tid, doc_id, text) in enumerate(to_judge):
        prompt = JUDGE.format(q=topics.get(tid, ""), p=text)
        for attempt in range(4):
            start = time.monotonic()
            try:
                r = client.messages.create(
                    model=model, max_tokens=4,
                    messages=[{"role": "user", "content": prompt}]
                )
                elapsed = time.monotonic() - start
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                raw = r.content[0].text.strip()
                m = re.search(r"[012]", raw)
                score = int(m.group()) if m else 0
                break
            except Exception as e:
                wait = min(60, 5 * 2 ** attempt)
                print(f"    API error: {e}, retrying in {wait}s")
                time.sleep(wait)
                score = 0
        qrels.setdefault(tid, {})[(doc_id, text)] = score
        if (i + 1) % 100 == 0:
            _save()
            print(f"    Judged {i+1}/{len(to_judge)}")
    _save()
    print(f"  Done. {len(to_judge)} new passages judged.")


def load_qrels(path: Path) -> dict:
    with path.open() as f:
        raw = json.load(f)
    return {
        tid: {tuple(k.split("\x00", 1)): int(v) for k, v in passages.items()}
        for tid, passages in raw.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file",  default=str(BASE / "data/synthetic_val/synthetic_topics_val65.json"))
    parser.add_argument("--loop-state",   default=str(BASE / "data/synthetic_autoresearch_val65/loop_state.json"))
    parser.add_argument("--qrels",        default=str(BASE / "data/synthetic_autoresearch_val65/qrels_loop_claude_haiku_4_5_20251001.json"))
    parser.add_argument("--output-dir",   default=str(BASE / "data/pipeline_exploration"))
    parser.add_argument("--bm25-index",   default=str(BASE / "data/processed/bm25s_index"))
    parser.add_argument("--meta",         default=str(BASE / "data/processed/passages_meta.pkl"))
    parser.add_argument("--faiss-index",  default=str(BASE / "data/processed/faiss_index.bin"))
    parser.add_argument("--dense-model",  default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    parser.add_argument("--ce-model",     default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--lora-model",   default=str(BASE / "data/processed/lora_reranker/merged"),
                        help="Path to LoRA-merged cross-encoder (used if it exists)")
    parser.add_argument("--loop-run",     default=str(BASE / "data/synthetic_autoresearch_val65/final_run.txt"),
                        help="Autoresearch final run file (per-topic best passages)")
    parser.add_argument("--model",        default="claude-haiku-4-5-20251001",
                        help="Anthropic model for judging and LLM proposals")
    parser.add_argument("--max-configs",  type=int, default=50)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--propose-every", type=int, default=5,
                        help="Ask LLM to propose new configs every N completed configs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path  = out_dir / "results.json"
    log_path      = out_dir / "exploration_log.jsonl"
    analysis_path = out_dir / "llm_analysis.jsonl"

    # Load topics
    with open(args.topics_file) as f:
        raw_topics = json.load(f)
    topics = {
        (rec.get("loop_topic_id") or rec.get("topic_id", "")): rec["question"]
        for rec in raw_topics if rec.get("question")
    }
    print(f"Loaded {len(topics)} topics")

    # Load cached reformulations from loop state
    reform_cache: dict[str, str] = {}
    multi_query_cache: dict[str, list[str]] = {}
    if Path(args.loop_state).exists():
        with open(args.loop_state) as f:
            state = json.load(f)
        for tid, s in state.items():
            rf = s.get("_reformulated", "")
            if rf:
                reform_cache[tid] = rf
                if "|||" in rf:
                    parts = [p.strip() for p in rf.split("|||") if p.strip()]
                    if len(parts) >= 2:
                        multi_query_cache[tid] = parts
                        reform_cache[tid] = parts[0]  # reform = first sub-query
        print(f"Loaded reformulations: {len(reform_cache)} reform, "
              f"{len(multi_query_cache)} multi-query")

    # Load qrels
    qrels = load_qrels(Path(args.qrels))
    qrels_path = Path(args.qrels)
    total_judged = sum(len(v) for v in qrels.values())
    print(f"Loaded qrels: {len(qrels)} topics, {total_judged} judgments")

    # Load autoresearch final run (per-topic best passages)
    loop_run: dict[str, list[tuple]] = {}
    if Path(args.loop_run).exists():
        from collections import defaultdict as _dd
        _tmp = _dd(list)
        with open(args.loop_run) as f:
            for line in f:
                parts = line.strip().split(";", 3)
                if len(parts) == 4:
                    tid, rank, doc_id, text = parts
                    _tmp[tid].append((int(rank), doc_id, text))
        loop_run = {tid: sorted(v, key=lambda x: x[0]) for tid, v in _tmp.items()}
        print(f"Loaded autoresearch loop run: {len(loop_run)} topics, "
              f"{sum(len(v) for v in loop_run.values())} passages")

    # Load retriever
    retriever = Retriever(
        bm25_dir=args.bm25_index,
        meta_file=args.meta,
        faiss_path=args.faiss_index,
        dense_model_name=args.dense_model,
        ce_model_name=args.ce_model,
        lora_model_path=args.lora_model,
    )

    client = anthropic.Anthropic()

    # Load previous results if resuming
    all_results: list[dict] = []
    tried_keys: set[str] = set()
    if results_path.exists():
        with results_path.open() as f:
            all_results = json.load(f)
        tried_keys = {r["key"] for r in all_results}
        print(f"Resuming: {len(all_results)} configs already evaluated")

    # Build config queue: seeds first, then LLM-guided
    config_queue: list[PipelineConfig] = []
    for cfg in SEED_CONFIGS:
        if cfg.to_key() not in tried_keys:
            config_queue.append(cfg)

    def save_results():
        ranked = sorted(all_results, key=lambda x: -x["mean_ndcg"])
        with results_path.open("w") as f:
            json.dump(ranked, f, indent=2)

    def print_leaderboard(top_n: int = 5):
        ranked = sorted(all_results, key=lambda x: -x["mean_ndcg"])
        print(f"\n{'='*65}")
        print(f"LEADERBOARD (top {top_n})")
        print(f"{'='*65}")
        for i, r in enumerate(ranked[:top_n]):
            print(f"  {i+1}. {r['name']:<30} nDCG={r['mean_ndcg']:.4f}  {r['key']}")

    # ── Main exploration loop ──────────────────────────────────────────────────
    config_num = len(all_results)
    rng = random.Random(args.seed)

    while config_num < args.max_configs:

        # Refill queue when empty
        if not config_queue:
            n_done = len(all_results)
            n_seeds = len(SEED_CONFIGS)

            if n_done < n_seeds:
                # Still in seed phase — nothing to propose, will hit "done" below
                pass
            elif n_done % args.propose_every == 0:
                # Every propose_every configs: half from LLM, half from evolutionary
                print(f"\n── Refilling queue (after {n_done} configs) ──")

                # LLM proposals (2 configs) — sees full ranked table, not just top
                analysis, llm_cfgs = llm_propose(client, args.model,
                                                  all_results, tried_keys)
                if analysis:
                    print(f"  LLM analysis: {analysis[:150]}...")
                    with analysis_path.open("a") as f:
                        f.write(json.dumps({
                            "after_config": n_done,
                            "analysis": analysis,
                            "proposed": [asdict(p) for p in llm_cfgs],
                        }) + "\n")
                config_queue.extend(llm_cfgs[:2])

                # Evolutionary proposals (3 configs): greedy + random-rank mutations + crossover
                evo_cfgs = next_batch(all_results, tried_keys, rng,
                                      n_start=config_num,
                                      batch_size=3,
                                      greedy_frac=0.34,
                                      mutate_frac=0.33,
                                      cross_frac=0.33)
                config_queue.extend(evo_cfgs)
                print(f"  {len(llm_cfgs[:2])} LLM + {len(evo_cfgs)} evolutionary configs queued")
            else:
                # Between LLM rounds: pure evolutionary fill
                evo_cfgs = next_batch(all_results, tried_keys, rng,
                                      n_start=config_num, batch_size=5)
                config_queue.extend(evo_cfgs)

        if not config_queue:
            print("No more novel configs to try — done.")
            break

        cfg = config_queue.pop(0)
        key = cfg.to_key()
        if key in tried_keys:
            continue

        config_num += 1
        tried_keys.add(key)

        print(f"\n{'─'*65}")
        print(f"Config {config_num}/{args.max_configs}: {cfg.name}")
        print(f"  Hypothesis: {cfg.hypothesis}")
        print(f"  Key: {key}")

        # Run pipeline
        t0 = time.monotonic()
        passages_by_topic = run_config(cfg, topics, retriever, reform_cache,
                                       multi_query_cache, loop_run=loop_run)
        retrieval_time = time.monotonic() - t0
        print(f"  Retrieval: {retrieval_time:.1f}s")

        # Judge new passages
        judge_new_passages(
            client, args.model, topics, qrels,
            passages_by_topic, qrels_path,
        )

        # Score
        topic_ndcgs = {
            tid: score_passages(passages, qrels, tid)
            for tid, passages in passages_by_topic.items()
        }
        mean_ndcg = sum(topic_ndcgs.values()) / len(topic_ndcgs) if topic_ndcgs else 0.0
        n_improved = sum(1 for v in topic_ndcgs.values() if v > 0)

        print(f"  mean nDCG@20 = {mean_ndcg:.4f}  "
              f"({n_improved}/{len(topic_ndcgs)} topics with relevance)")

        result = {
            "rank":        0,
            "name":        cfg.name,
            "key":         key,
            "hypothesis":  cfg.hypothesis,
            "mean_ndcg":   round(mean_ndcg, 4),
            "topic_ndcgs": {tid: round(v, 4) for tid, v in topic_ndcgs.items()},
            "config":      asdict(cfg),
            "config_num":  config_num,
            "retrieval_s": round(retrieval_time, 1),
        }
        all_results.append(result)

        # Log to JSONL
        with log_path.open("a") as f:
            f.write(json.dumps({k: v for k, v in result.items()
                                 if k != "topic_ndcgs"}) + "\n")

        save_results()

        if config_num % 5 == 0:
            print_leaderboard()

    # ── Final summary ─────────────────────────────────────────────────────────
    ranked = sorted(all_results, key=lambda x: -x["mean_ndcg"])
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    print(f"\n{'='*65}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*65}")
    print(f"\nTop 3 pipeline configurations:")
    for r in ranked[:3]:
        print(f"\n  #{r['rank']} {r['name']}  nDCG={r['mean_ndcg']:.4f}")
        print(f"       Key: {r['key']}")
        print(f"       Hypothesis: {r['hypothesis']}")

    # Save final ranked results
    save_results()
    print(f"\nFull results saved to {results_path}")

    # Write top-3 run files for submission
    for r in ranked[:3]:
        cfg_data = r["config"]
        cfg = PipelineConfig(**cfg_data)
        passages_by_topic = run_config(cfg, topics, retriever,
                                       reform_cache, multi_query_cache,
                                       loop_run=loop_run)
        run_path = out_dir / f"top_{r['rank']}_{r['name']}.txt"
        with run_path.open("w") as f:
            for tid in sorted(passages_by_topic):
                for rank, doc_id, text in passages_by_topic[tid]:
                    f.write(f"{tid};{rank};{doc_id};{text}\n")
        print(f"  Run saved: {run_path}")


if __name__ == "__main__":
    main()
