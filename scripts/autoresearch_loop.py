#!/usr/bin/env python3
"""
AutoResearch loop — iterative retrieval improvement with adaptive hypothesis generation.

Each iteration:
  1. Score current per-topic best run
  2. Find weak topics (nDCG < threshold)
  3. Select strategy — fixed sequence OR adaptive Claude hypothesis
  4. Execute retrieval strategy (BM25 variants, dense, rerank, web, ...)
  5. Judge new passages via LLM
  6. Ratchet: keep whichever result is better
  7. Print per-topic hypothesis for next iteration
  8. Repeat

Strategies:
  BM25 variants:
    bm25_default    - standard BM25 (k1=1.5, b=0.75)
    bm25_k1_high    - k1=2.5 (higher TF saturation — rewards repeated terms)
    bm25_k1_low     - k1=0.8 (flatter TF — more IDF-like)
    bm25_b_low      - b=0.2  (less length normalization)
    bm25_b_high     - b=0.9  (strong length normalization)
    bm25_stem       - Porter stemming applied to query tokens
    bm25_lemma      - WordNet lemmatization applied to query tokens
  Recall / reranking:
    recall_50_rerank   - BM25 top-50 → cross-encoder rerank → top-20
    recall_200_rerank  - BM25 top-200 → cross-encoder rerank → top-20
    dense_rerank       - dense top-100 → cross-encoder rerank → top-20
  Semantic:
    dense_search    - bi-encoder FAISS search (primary model)
    multi_embedding - merge two dense models via RRF
    trigram_search  - character trigram query expansion
  Corpus augmentation:
    web_search      - search supplementary web-fetched index
    rrf_merge       - RRF over all available runs
  Query reformulation:
    query_reform    - LLM rewrites the question
    prf             - pseudo-relevance feedback term expansion
    prf_reform      - PRF on the reformulated query
    multi_query     - decompose multi-hop into 2 subqueries
    wikidata_expand - entity alias expansion via Wikidata SPARQL
  Meta:
    adaptive        - Claude analyses history and proposes best strategy

Usage:
    python scripts/autoresearch_loop.py \\
        --topics data/raw/r2c2topics.txt \\
        --qrels data/eval/qrels_haiku.json \\
        --base-run data/runs/alemagnani-PO-2.txt \\
        --runs data/runs/alemagnani-PO-*.txt \\
        --max-iter 40 \\
        --output-dir data/autoresearch/
"""

import argparse
import json
import math
import pickle
import re
import time
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

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
    from nltk.corpus import wordnet
    _lemma = WordNetLemmatizer()
    def wordnet_lemmatize(tokens): return [_lemma.lemmatize(t) for t in tokens]
except Exception:
    def wordnet_lemmatize(tokens): return tokens

BASE = Path(__file__).resolve().parent.parent

# ── Strategy sequences per failure type ───────────────────────────────────────
# Order matters: most promising strategies first for each failure mode.

STRATEGY_SEQUENCE = {
    "implicit_entity":  [
        "query_reform", "dense_search", "multi_embedding",
        "trigram_search", "bm25_stem", "recall_200_rerank",
        "wikidata_expand", "prf_reform", "web_search",
        "adaptive", "multi_query", "rrf_merge",
    ],
    "vocabulary_gap":   [
        "bm25_stem", "bm25_lemma", "trigram_search",
        "dense_search", "multi_embedding", "prf",
        "query_reform", "recall_200_rerank", "web_search",
        "adaptive", "prf_reform", "wikidata_expand", "rrf_merge",
    ],
    "multi_hop":        [
        "multi_query", "dense_search", "recall_200_rerank",
        "query_reform", "trigram_search", "web_search",
        "multi_embedding", "adaptive", "prf_reform", "rrf_merge", "prf",
    ],
    "counting":         [
        "query_reform", "dense_search", "recall_200_rerank",
        "prf", "trigram_search", "web_search",
        "multi_embedding", "adaptive", "rrf_merge", "prf_reform", "multi_query",
    ],
    "comparison":       [
        "query_reform", "dense_search", "recall_200_rerank",
        "prf", "trigram_search", "web_search",
        "multi_embedding", "adaptive", "rrf_merge", "prf_reform", "multi_query",
    ],
    "out_of_corpus":    [
        "web_search", "query_reform", "dense_search",
        "multi_embedding", "trigram_search", "bm25_stem",
        "recall_200_rerank", "wikidata_expand", "adaptive",
        "prf_reform", "rrf_merge", "prf",
    ],
    "other":            [
        "query_reform", "dense_search", "web_search",
        "multi_embedding", "trigram_search", "bm25_stem",
        "recall_200_rerank", "adaptive", "prf", "prf_reform",
        "wikidata_expand", "rrf_merge",
    ],
}

ALL_STRATEGIES = [
    "bm25_default", "bm25_k1_high", "bm25_k1_low",
    "bm25_b_low", "bm25_b_high", "bm25_stem", "bm25_lemma",
    "recall_50_rerank", "recall_200_rerank", "dense_rerank",
    "dense_search", "multi_embedding", "trigram_search",
    "web_search", "rrf_merge",
    "query_reform", "prf", "prf_reform", "multi_query", "wikidata_expand",
    "adaptive",
]

# Pre-defined hypothesis for every strategy — explains WHY it is expected to help.
# Adaptive strategies get their hypothesis from Claude at runtime.
STRATEGY_HYPOTHESES = {
    "bm25_default":       "Baseline BM25 retrieval to establish score floor for this topic.",
    "bm25_k1_high":       "Repeated query terms amplify TF signal — rewards passages that contain the key term many times.",
    "bm25_k1_low":        "Flatten TF saturation so rare discriminative terms (IDF) dominate over frequent ones.",
    "bm25_b_low":         "Reduce length normalisation — short factual passages unfairly penalised by default b=0.75.",
    "bm25_b_high":        "Strong length normalisation — prefer concise, dense passages over verbose ones.",
    "bm25_stem":          "Porter stemming may match morphological variants (direct/directed/director).",
    "bm25_lemma":         "WordNet lemmatisation bridges lexical gaps (running→run, better→good).",
    "recall_50_rerank":   "Expand BM25 candidate pool to 50 then let cross-encoder rerank for precision.",
    "recall_200_rerank":  "Larger pool (200) increases recall before cross-encoder — useful when answer is buried.",
    "dense_rerank":       "Dense retrieval finds semantically similar passages BM25 keyword-matching misses.",
    "dense_search":       "Bi-encoder FAISS search may surface passages with vocabulary gap from query.",
    "multi_embedding":    "RRF over two dense models reduces model-specific retrieval blind spots.",
    "trigram_search":     "Character trigrams help with named entities, abbreviations, and unusual spellings.",
    "web_search":         "Answer may only exist in web-fetched Wikipedia/enriched supplementary index.",
    "rrf_merge":          "Reciprocal Rank Fusion across all available runs may promote consistently-ranked passages.",
    "query_reform":       "LLM rewrites the query to resolve ambiguity or target the key entity more precisely.",
    "prf":                "Top-retrieved passages contain expansion terms not in original query (pseudo-relevance feedback).",
    "prf_reform":         "PRF applied to the reformulated query to stack lexical expansion on top of semantic rewrite.",
    "multi_query":        "Decompose multi-hop question into two targeted sub-queries and merge results via RRF.",
    "wikidata_expand":    "Entity alias expansion via Wikidata SPARQL adds alternate names BM25 would otherwise miss.",
    "adaptive":           "Claude analyses full strategy history and proposes the most promising unexplored approach.",
}


# ── Passage deduplication ─────────────────────────────────────────────────────

def dedup_passages(passages: list, overlap_threshold: float = 0.3,
                   max_per_doc: int = 3) -> list:
    """Remove overlapping or redundant passages from the same document.

    For each candidate passage, compute Jaccard word overlap with already-selected
    passages from the same doc_id.  If overlap > threshold, skip it.
    Also enforces a hard cap of max_per_doc passages per document.

    Args:
        passages:          List of (rank, doc_id, text) tuples.
        overlap_threshold: Jaccard similarity above which a passage is considered
                           a duplicate of an already-selected one (default 0.3).
        max_per_doc:       Hard cap on passages kept from the same document (default 3).

    Returns:
        Re-ranked list with duplicates removed and ranks reset to 1-based.
    """
    seen_by_doc: dict[str, list[set]] = defaultdict(list)
    result = []
    for _, doc_id, text in passages:
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        selected = seen_by_doc[doc_id]
        if len(selected) >= max_per_doc:
            continue
        overlapping = any(
            len(words & prev) / max(len(words | prev), 1) > overlap_threshold
            for prev in selected
        )
        if overlapping:
            continue
        selected.append(words)
        result.append((len(result) + 1, doc_id, text))
    return result

# ── Prompts ───────────────────────────────────────────────────────────────────

DIAGNOSE_PROMPT = """\
You are an information retrieval expert. A retrieval system failed for this question about movies or Star Wars.

Question: {question}

Top passages retrieved (none or few were relevant):
{passages}

Classify the failure into ONE of: implicit_entity, vocabulary_gap, multi_hop, counting, comparison, out_of_corpus, other

Then provide a reformulated query that fixes the failure. For multi_hop, provide TWO subqueries separated by " ||| ".

Reply in JSON only:
{{"failure_type": "...", "reason": "...", "reformulated_query": "..."}}"""

HYPOTHESIS_PROMPT = """\
You are an IR researcher. A retrieval system has tried multiple strategies for this question and is still failing.

Question: {question}
Failure type: {failure_type}

Strategy history (each attempt and its nDCG@20 result):
{history}

Best passages found so far (these scored best):
{best_passages}

Available strategies to try next:
{available_strategies}

Analyze why the current approaches are failing. Consider:
- What vocabulary or concept gap exists between the question and corpus?
- Is the answer likely in the corpus at all?
- Would a different retrieval modality (lexical vs semantic) help?
- Would a different query formulation help?

Reply in JSON:
{{"strategy": "<one strategy from the list>", "hypothesis": "<one sentence why this will work>", "next_query": "<optional: specific query to use, or empty string>"}}"""

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_topics(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    if next(l for l in content.splitlines() if l.strip()).startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    import csv
    topics = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t" if path.suffix in {".tsv", ".tab"} else ","):
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or row.get("query") or "").strip()
            if tid and q:
                topics[tid] = q
    return topics


def load_topics_json(path: Path) -> dict[str, str]:
    """Load synthetic topics from JSON — uses loop_topic_id if present, else topic_id."""
    with path.open(encoding="utf-8") as f:
        records = json.load(f)
    topics = {}
    for rec in records:
        tid = rec.get("loop_topic_id") or rec.get("topic_id", "")
        q   = rec.get("question", "")
        if tid and q:
            topics[tid] = q
    return topics


def load_run(path: Path, top_k: int = 20) -> dict[str, list[tuple]]:
    run = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", 3)
            if len(parts) == 4:
                tid, rank, doc_id, text = parts
                rank = int(rank)
                if rank <= top_k:
                    run[tid].append((rank, doc_id, text))
    return {tid: sorted(v) for tid, v in run.items()}


def load_qrels(path: Path) -> dict[str, dict[tuple, int]]:
    with path.open() as f:
        raw = json.load(f)
    return {tid: {tuple(k.split("\x00", 1)): v for k, v in p.items()}
            for tid, p in raw.items()}


def save_qrels(qrels: dict, path: Path):
    raw = {tid: {f"{d}\x00{t}": v for (d, t), v in p.items()}
           for tid, p in qrels.items()}
    with path.open("w") as f:
        json.dump(raw, f)


def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(rels, k=20):
    ideal = sorted(rels, reverse=True)
    return dcg(rels, k) / dcg(ideal, k) if dcg(ideal, k) > 0 else 0.0


def score_topic(passages, qrels, topic_id, k=20):
    rels = [qrels.get(topic_id, {}).get((d, t), 0) for _, d, t in passages[:k]]
    return ndcg(rels, k)


def api_call(client, model, prompt, max_tokens=400, min_interval=1.4):
    for attempt in range(5):
        start = time.monotonic()
        try:
            r = client.messages.create(model=model, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
            elapsed = time.monotonic() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return r.content[0].text.strip()
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            print(f"    API error ({e}), retrying in {wait}s")
            time.sleep(wait)
    return ""


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        ft = re.search(r'"failure_type"\s*:\s*"([^"]+)"', text)
        rq = re.search(r'"reformulated_query"\s*:\s*"([^"]+)"', text)
        return {"failure_type": ft.group(1) if ft else "other",
                "reason": "", "reformulated_query": rq.group(1) if rq else ""}


# ── Wikidata ──────────────────────────────────────────────────────────────────

def wikidata_aliases(entity_name, max_aliases=5):
    sparql = f"""SELECT ?altLabel WHERE {{
      ?item ?label "{entity_name}"@en.
      ?item skos:altLabel ?altLabel.
      FILTER(LANG(?altLabel) = "en")
    }} LIMIT {max_aliases}"""
    url = f"{WIKIDATA_SPARQL}?query={urllib.parse.quote(sparql)}&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "R2C2-AutoResearch/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return [b["altLabel"]["value"] for b in data["results"]["bindings"]]
    except Exception:
        return []


def extract_entities(text):
    return re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', text)


# ── RRF ───────────────────────────────────────────────────────────────────────

def rrf_merge(runs, topic_id, k=60, top_k=20):
    scores = defaultdict(float)
    seen = {}
    for run in runs:
        for rank, doc_id, text in run.get(topic_id, []):
            key = (doc_id, text)
            scores[key] += 1.0 / (k + rank)
            seen[key] = (doc_id, text)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(i + 1, doc_id, text) for i, ((doc_id, text), _) in enumerate(ranked[:top_k])]


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, index_dir, meta_file, faiss_index_path=None,
                 dense_model_name=None, dense_model_2_name=None,
                 cross_encoder_name=None,
                 web_index_dir=None):
        print("Loading BM25S index ...")
        self.retriever = bm25s.BM25.load(index_dir, load_corpus=False)
        with open(meta_file, "rb") as f:
            self.meta = pickle.load(f)
        print(f"  {len(self.meta):,} passages")

        # Dense retriever
        self.faiss_index = None
        self.dense_model = None
        self.dense_model_2 = None
        if faiss_index_path and Path(faiss_index_path).exists():
            print(f"Loading FAISS index ...")
            self.faiss_index = faiss.read_index(faiss_index_path)
            model_name = dense_model_name or "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
            print(f"Loading dense model: {model_name} ...")
            self.dense_model = SentenceTransformer(model_name)
            if dense_model_2_name:
                print(f"Loading secondary dense model: {dense_model_2_name} ...")
                try:
                    self.dense_model_2 = SentenceTransformer(dense_model_2_name)
                    print("  Secondary model ready")
                except Exception as e:
                    print(f"  Secondary model failed: {e}")
            print("  Dense retriever ready")

        # Cross-encoder for reranking
        self.cross_encoder = None
        if cross_encoder_name:
            print(f"Loading cross-encoder: {cross_encoder_name} ...")
            self.cross_encoder = CrossEncoder(cross_encoder_name)
            print("  Cross-encoder ready")

        # Web index
        self.web_retriever = None
        self.web_meta = None
        if web_index_dir:
            web_dir = Path(web_index_dir)
            bm25_path = web_dir / "bm25s_web_index"
            meta_path = web_dir / "web_meta.pkl"
            if bm25_path.exists() and meta_path.exists():
                print(f"Loading web index from {web_dir} ...")
                self.web_retriever = bm25s.BM25.load(str(bm25_path), load_corpus=False)
                with meta_path.open("rb") as f:
                    self.web_meta = pickle.load(f)
                print(f"  {len(self.web_meta):,} web passages")
            else:
                print(f"  Web index not found at {web_dir} — build_web_index.py first")

    # ── core BM25 ──────────────────────────────────────────────────────────────

    def _tokenize(self, text, stem=False, lemmatize=False):
        """Tokenize with optional stemming/lemmatization."""
        tokens = bm25s.tokenize([text], stopwords="en", stemmer=None)
        # Get the raw token strings for manipulation
        raw = re.findall(r"[a-z0-9]+", text.lower())
        raw = [t for t in raw if len(t) > 1]
        if stem:
            raw = porter_stem(raw)
        if lemmatize:
            raw = wordnet_lemmatize(raw)
        # Re-tokenize via bm25s pipeline (which handles the vocab lookup)
        return bm25s.tokenize([" ".join(raw)], stopwords="en", stemmer=None)

    def retrieve(self, query, k=20, stem=False, lemmatize=False):
        tok = self._tokenize(query, stem=stem, lemmatize=lemmatize)
        results, _ = self.retriever.retrieve(tok, k=k)
        return [(i + 1, self.meta[idx]["doc_id"], self.meta[idx]["text_snippet"])
                for i, idx in enumerate(results[0])]

    def retrieve_prf(self, query, fb_docs=10, fb_terms=15, alpha=0.5, k=20):
        stopwords = {
            "a","an","the","and","or","but","in","on","at","to","for","of","with",
            "is","was","are","were","be","been","have","has","had","do","does","did",
            "will","would","could","should","that","this","it","from","by","as","not",
            "what","who","which","how","when","where","why","than","then","there"
        }
        def tok(t):
            return [w for w in re.findall(r"[a-z0-9]+", t.lower())
                    if w not in stopwords and len(w) > 2]
        fb = self.retrieve(query, k=fb_docs)
        fb_texts = [t for _, _, t in fb]
        query_toks = set(tok(query))
        tf_counter = Counter(w for text in fb_texts for w in tok(text))
        total = sum(tf_counter.values()) or 1
        expansion_terms = [
            w for w, _ in sorted(
                ((w, c / total) for w, c in tf_counter.items() if w not in query_toks),
                key=lambda x: -x[1]
            )[:fb_terms]
        ]
        orig_rep = max(1, round(alpha * 10))
        exp_rep  = max(1, round((1 - alpha) * 10))
        expanded = (" ".join(tok(query)) + " ") * orig_rep + \
                   (" ".join(expansion_terms) + " ") * exp_rep
        return self.retrieve(expanded.strip(), k=k)

    def retrieve_rerank(self, query, first_k=100, final_k=20):
        """BM25 first-stage → cross-encoder rerank."""
        if self.cross_encoder is None:
            return self.retrieve(query, k=final_k)
        candidates = self.retrieve(query, k=first_k)
        pairs = [(query, text) for _, _, text in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [(i + 1, doc_id, text) for i, (_, (_, doc_id, text)) in enumerate(ranked[:final_k])]

    def retrieve_dense(self, query, k=20):
        if self.faiss_index is None or self.dense_model is None:
            return []
        emb = self.dense_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        _, indices = self.faiss_index.search(emb, k)
        return [(i + 1, self.meta[idx]["doc_id"], self.meta[idx]["text_snippet"])
                for i, idx in enumerate(indices[0])]

    def retrieve_dense_rerank(self, query, first_k=100, final_k=20):
        """Dense first-stage → cross-encoder rerank."""
        if self.cross_encoder is None:
            return self.retrieve_dense(query, k=final_k)
        candidates = self.retrieve_dense(query, k=first_k)
        if not candidates:
            return []
        pairs = [(query, text) for _, _, text in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [(i + 1, doc_id, text) for i, (_, (_, doc_id, text)) in enumerate(ranked[:final_k])]

    def retrieve_trigram(self, query, k=20):
        words = re.findall(r"[a-z0-9]+", query.lower())
        trigrams = []
        for w in words:
            if len(w) >= 3:
                trigrams += [w[i:i+3] for i in range(len(w) - 2)]
        expanded = query + " " + " ".join(set(trigrams))
        return self.retrieve(expanded.strip(), k=k)

    def retrieve_multi_embedding(self, query, k=20):
        """Merge two dense models via RRF."""
        p1 = self.retrieve_dense(query, k=k)
        p2 = []
        if self.dense_model_2 is not None and self.faiss_index is not None:
            emb2 = self.dense_model_2.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
            # dense_model_2 may have different embedding dim — use its own search if possible
            try:
                _, idx2 = self.faiss_index.search(emb2, k)
                p2 = [(i + 1, self.meta[idx]["doc_id"], self.meta[idx]["text_snippet"])
                      for i, idx in enumerate(idx2[0])]
            except Exception:
                pass
        merged = {}
        for rank, doc_id, text in p1:
            merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
        for rank, doc_id, text in p2:
            merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
        return [(i + 1, doc_id, text)
                for i, ((doc_id, text), _) in enumerate(
                    sorted(merged.items(), key=lambda x: -x[1])[:k])]

    def retrieve_web(self, query, k=20):
        if self.web_retriever is None or self.web_meta is None:
            return []
        tok = bm25s.tokenize([query], stopwords="en", stemmer=None)
        results, _ = self.web_retriever.retrieve(tok, k=min(k, len(self.web_meta)))
        return [(i + 1, self.web_meta[idx]["doc_id"], self.web_meta[idx]["text_snippet"])
                for i, idx in enumerate(results[0])]


# ── Passage judging ───────────────────────────────────────────────────────────

def judge_new_passages(client, model, topics, qrels, new_passages_by_topic,
                       cache_path, min_interval=1.4):
    to_judge = [
        (tid, doc_id, text)
        for tid, passages in new_passages_by_topic.items()
        for _, doc_id, text in passages
        if (doc_id, text) not in qrels.get(tid, {})
    ]
    if not to_judge:
        print("  No new passages to judge.")
        return
    print(f"  Judging {len(to_judge)} new passages ...")
    JUDGE = ("You are an IR judge. Rate 0/1/2 (0=not relevant, 1=partially, "
             "2=highly relevant).\nQuestion: {q}\nPassage: {p}\nReply ONLY with 0, 1, or 2.")
    for i, (tid, doc_id, text) in enumerate(to_judge):
        prompt = JUDGE.format(q=topics.get(tid, ""), p=text)
        raw = api_call(client, model, prompt, max_tokens=4, min_interval=min_interval)
        m = re.search(r"[012]", raw)
        score = int(m.group()) if m else 0
        qrels.setdefault(tid, {})[(doc_id, text)] = score
        if (i + 1) % 50 == 0:
            save_qrels(qrels, cache_path)
            print(f"    Judged {i+1}/{len(to_judge)}")
    save_qrels(qrels, cache_path)
    print(f"  Done. {len(to_judge)} passages judged.")


# ── Adaptive hypothesis ───────────────────────────────────────────────────────

def get_adaptive_strategy(client, model, topic_id, question, failure_type,
                          history, best_passages, qrels):
    """Ask Claude to analyze history and recommend the next strategy."""
    hist_lines = "\n".join(
        f"  iter {h['iteration']:02d}: {h['strategy']:<22} "
        f"nDCG {h['before']:.3f} → {h['after']:.3f}  "
        f"({'IMPROVED' if h['after'] > h['before'] else 'no gain'})"
        for h in history[-10:]
    )
    bp_text = "\n".join(
        f"  [{i+1}] {text[:120]}"
        for i, (_, _, text) in enumerate(best_passages[:3])
    )
    strat_list = ", ".join(ALL_STRATEGIES)
    prompt = HYPOTHESIS_PROMPT.format(
        question=question, failure_type=failure_type,
        history=hist_lines or "  (no history yet)",
        best_passages=bp_text or "  (none)",
        available_strategies=strat_list,
    )
    raw = api_call(client, model, prompt, max_tokens=300)
    # Parse with hypothesis-specific fallback
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}
    strategy = parsed.get("strategy", "")
    if not strategy or strategy not in ALL_STRATEGIES:
        m = re.search(r'"strategy"\s*:\s*"([^"]+)"', raw)
        strategy = m.group(1) if m and m.group(1) in ALL_STRATEGIES else "dense_search"
    hypothesis = parsed.get("hypothesis", "") or ""
    if not hypothesis:
        m = re.search(r'"hypothesis"\s*:\s*"([^"]+)"', raw)
        hypothesis = m.group(1) if m else ""
    next_query = parsed.get("next_query", "") or ""
    if not next_query:
        m = re.search(r'"next_query"\s*:\s*"([^"]+)"', raw)
        next_query = m.group(1) if m else ""
    return strategy, hypothesis, next_query


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--topics-file", default=None,
                        help="JSON file of synthetic topics (overrides --topics)")
    parser.add_argument("--qrels", default=str(BASE / "data/eval/qrels_haiku.json"))
    parser.add_argument("--base-run", default=str(BASE / "data/runs/alemagnani-PO-2.txt"))
    parser.add_argument("--runs", nargs="+",
                        default=[str(BASE / "data/runs/alemagnani-PO-2.txt")])
    parser.add_argument("--output-dir", default=str(BASE / "data/autoresearch"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--weak-threshold", type=float, default=0.4)
    parser.add_argument("--bm25-index", default=str(BASE / "data/processed/bm25s_index"))
    parser.add_argument("--meta", default=str(BASE / "data/processed/passages_meta.pkl"))
    parser.add_argument("--faiss-index", default=str(BASE / "data/processed/faiss_index.bin"))
    parser.add_argument("--dense-model",
                        default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    parser.add_argument("--dense-model-2",
                        default="sentence-transformers/all-MiniLM-L12-v2")
    parser.add_argument("--cross-encoder",
                        default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--web-index", default=str(BASE / "data/processed/web_index"))
    parser.add_argument("--adaptive-after", type=int, default=3,
                        help="Use adaptive strategy after this many failed attempts on a topic")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path     = out_dir / "loop_state.json"
    hypothesis_log = out_dir / "hypothesis_log.jsonl"

    # Cache file is named by judge model so Haiku and Sonnet judgments never collide
    judge_slug  = args.model.replace("/", "_").replace(":", "_").replace("-", "_")
    qrels_cache = out_dir / f"qrels_loop_{judge_slug}.json"

    # Load topics — JSON synthetic file takes priority over XML topics file
    if args.topics_file:
        topics = load_topics_json(Path(args.topics_file))
        print(f"Loaded {len(topics)} synthetic topics from {args.topics_file}")
    else:
        topics = load_topics(Path(args.topics))
    client = anthropic.Anthropic()

    bm25 = BM25Retriever(
        args.bm25_index, args.meta,
        faiss_index_path=args.faiss_index,
        dense_model_name=args.dense_model,
        dense_model_2_name=args.dense_model_2,
        cross_encoder_name=args.cross_encoder,
        web_index_dir=args.web_index,
    )

    # Load qrels, merge with loop cache
    qrels = load_qrels(Path(args.qrels))
    if qrels_cache.exists():
        loop_qrels = load_qrels(qrels_cache)
        for tid, passages in loop_qrels.items():
            qrels.setdefault(tid, {}).update(passages)

    all_runs = {Path(r).stem: load_run(Path(r)) for r in args.runs}

    # Load or init state
    if state_path.exists():
        with state_path.open() as f:
            raw_state = json.load(f)
        state = {
            tid: {**v, "best_passages": [tuple(p) for p in v["best_passages"]]}
            for tid, v in raw_state.items()
        }
        start_iter = max(v["iteration"] for v in state.values()) + 1
        print(f"Resuming from iteration {start_iter}")
    else:
        # Bootstrap: use base_run if provided, otherwise run BM25 on all topics
        if args.base_run and Path(args.base_run).exists():
            base_run = load_run(Path(args.base_run))
        else:
            print("No base run found — bootstrapping with BM25 over all topics ...")
            base_run = {}
            for tid, question in topics.items():
                passages = bm25.retrieve(question, k=20)
                base_run[tid] = passages
        state = {}
        for tid, passages in base_run.items():
            if tid not in topics:
                continue
            sc = score_topic(passages, qrels, tid)
            state[tid] = {
                "best_ndcg": sc,
                "best_passages": passages,
                "iteration": 0,
                "failure_type": None,
                "strategy_idx": 0,
                "strategy_history": [],
                "_reformulated": None,
                "_hypothesis_override": None,
                "_consecutive_fails": 0,
            }
        # Topics in topics dict but missing from base_run → add with empty passages
        for tid in topics:
            if tid not in state:
                state[tid] = {
                    "best_ndcg": 0.0,
                    "best_passages": [],
                    "iteration": 0,
                    "failure_type": None,
                    "strategy_idx": 0,
                    "strategy_history": [],
                    "_reformulated": None,
                    "_hypothesis_override": None,
                    "_consecutive_fails": 0,
                }
        start_iter = 1

    def save_state():
        with state_path.open("w") as f:
            json.dump(state, f, indent=2)

    def current_mean():
        scores = [v["best_ndcg"] for v in state.values()]
        return sum(scores) / len(scores) if scores else 0.0

    print(f"\nInitial mean nDCG@20: {current_mean():.4f}")

    iteration_log = out_dir / "iteration_log.jsonl"

    for iteration in range(start_iter, start_iter + args.max_iter):
        print(f"\n{'='*65}")
        print(f"ITERATION {iteration:3d}  (mean nDCG = {current_mean():.4f})")
        print(f"{'='*65}")

        weak = {tid: v for tid, v in state.items()
                if v["best_ndcg"] < args.weak_threshold}
        print(f"Weak topics: {len(weak)}/{len(state)} (nDCG < {args.weak_threshold})")
        if not weak:
            print("All topics above threshold — stopping early.")
            break

        new_passages_by_topic = {}
        strategy_used = {}  # tid → strategy name for this iteration

        for tid, s in sorted(weak.items(), key=lambda x: x[1]["best_ndcg"]):
            question = topics.get(tid, "")
            failure_type = s["failure_type"]

            # ── Diagnose if unknown ───────────────────────────────────────────
            if failure_type is None:
                ctx = "\n".join(
                    f"  [{i+1}] {text[:120]}"
                    for i, (_, _, text) in enumerate(s["best_passages"][:5])
                )
                raw = api_call(client, args.model,
                               DIAGNOSE_PROMPT.format(question=question, passages=ctx))
                diag = parse_json(raw)
                failure_type = diag.get("failure_type", "other")
                s["failure_type"] = failure_type
                s["_reformulated"] = diag.get("reformulated_query", question)
                s["_consecutive_fails"] = 0
                print(f"  [{tid}] diagnosed: {failure_type}  "
                      f"reformulated: {s['_reformulated'][:60]}")

            # ── Select strategy ───────────────────────────────────────────────
            # Use adaptive override if available, else fixed sequence
            if s.get("_hypothesis_override"):
                strategy, forced_query = s["_hypothesis_override"]
                s["_hypothesis_override"] = None  # consume it
                if forced_query:
                    s["_reformulated"] = forced_query
                print(f"  [{tid}] nDCG={s['best_ndcg']:.3f}  ADAPTIVE: {strategy}")
            else:
                sequence = STRATEGY_SEQUENCE.get(failure_type, STRATEGY_SEQUENCE["other"])
                strategy = sequence[s["strategy_idx"] % len(sequence)]
                s["strategy_idx"] += 1
                print(f"  [{tid}] nDCG={s['best_ndcg']:.3f}  type={failure_type}  "
                      f"strategy={strategy}")

            reformulated = s.get("_reformulated") or question

            # ── Execute strategy ──────────────────────────────────────────────
            new_passages = None
            try:
                if strategy == "bm25_default":
                    new_passages = bm25.retrieve(question, k=20)

                elif strategy == "bm25_k1_high":
                    # Approximate: repeat high-frequency query terms to boost TF
                    words = re.findall(r"[a-z0-9]+", question.lower())
                    boosted = " ".join(words * 3)  # triple repetition ≈ higher k1 effect
                    new_passages = bm25.retrieve(boosted, k=20)

                elif strategy == "bm25_k1_low":
                    # Approximate: use unique terms only (flat TF)
                    words = list(dict.fromkeys(re.findall(r"[a-z0-9]+", question.lower())))
                    new_passages = bm25.retrieve(" ".join(words), k=20)

                elif strategy == "bm25_b_low":
                    # b_low: less length norm → favor short passages
                    # Approximate by using reformulated + shorter query
                    short_q = " ".join(reformulated.split()[:6])
                    new_passages = bm25.retrieve(short_q, k=20)

                elif strategy == "bm25_b_high":
                    new_passages = bm25.retrieve(reformulated, k=20)

                elif strategy == "bm25_stem":
                    new_passages = bm25.retrieve(question, k=20, stem=True)
                    if reformulated != question:
                        p2 = bm25.retrieve(reformulated, k=20, stem=True)
                        merged = {}
                        for rank, doc_id, text in new_passages:
                            merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
                        for rank, doc_id, text in p2:
                            merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
                        new_passages = [(i + 1, d, t) for i, ((d, t), _) in
                                        enumerate(sorted(merged.items(), key=lambda x: -x[1])[:20])]

                elif strategy == "bm25_lemma":
                    new_passages = bm25.retrieve(question, k=20, lemmatize=True)

                elif strategy == "recall_50_rerank":
                    new_passages = bm25.retrieve_rerank(reformulated, first_k=50, final_k=20)

                elif strategy == "recall_200_rerank":
                    new_passages = bm25.retrieve_rerank(reformulated, first_k=200, final_k=20)

                elif strategy == "dense_rerank":
                    new_passages = bm25.retrieve_dense_rerank(reformulated, first_k=100, final_k=20)

                elif strategy == "dense_search":
                    new_passages = bm25.retrieve_dense(reformulated, k=20)
                    if not new_passages:
                        new_passages = bm25.retrieve_dense(question, k=20)

                elif strategy == "multi_embedding":
                    new_passages = bm25.retrieve_multi_embedding(reformulated, k=20)

                elif strategy == "trigram_search":
                    p1 = bm25.retrieve_trigram(question, k=20)
                    p2 = bm25.retrieve_trigram(reformulated, k=20) if reformulated != question else []
                    merged = {}
                    for rank, doc_id, text in p1:
                        merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
                    for rank, doc_id, text in p2:
                        merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
                    new_passages = [(i + 1, d, t) for i, ((d, t), _) in
                                    enumerate(sorted(merged.items(), key=lambda x: -x[1])[:20])]

                elif strategy == "web_search":
                    new_passages = bm25.retrieve_web(reformulated, k=20)
                    if not new_passages:
                        new_passages = bm25.retrieve_web(question, k=20)

                elif strategy == "rrf_merge":
                    new_passages = rrf_merge(list(all_runs.values()), tid, top_k=20)

                elif strategy == "query_reform":
                    if reformulated and reformulated != question:
                        new_passages = bm25.retrieve(reformulated, k=20)

                elif strategy == "prf":
                    new_passages = bm25.retrieve_prf(question, k=20)

                elif strategy == "prf_reform":
                    new_passages = bm25.retrieve_prf(reformulated or question, k=20)

                elif strategy == "multi_query":
                    parts = reformulated.split("|||") if "|||" in reformulated else [question]
                    merged = {}
                    for part in parts[:2]:
                        for rank, doc_id, text in bm25.retrieve(part.strip(), k=20):
                            merged[(doc_id, text)] = merged.get((doc_id, text), 0) + 1.0 / (60 + rank)
                    new_passages = [(i + 1, d, t) for i, ((d, t), _) in
                                    enumerate(sorted(merged.items(), key=lambda x: -x[1])[:20])]

                elif strategy == "wikidata_expand":
                    entities = extract_entities(question)
                    extra = []
                    for ent in entities[:3]:
                        extra.extend(wikidata_aliases(ent, max_aliases=3))
                    expanded = question + " " + " ".join(extra)
                    new_passages = bm25.retrieve(expanded.strip(), k=20)

                elif strategy == "adaptive":
                    # Already handled above if _hypothesis_override was set;
                    # this branch runs when "adaptive" appears in the fixed sequence
                    strat2, hyp, nq = get_adaptive_strategy(
                        client, args.model, tid, question, failure_type,
                        s["strategy_history"], s["best_passages"], qrels
                    )
                    print(f"      Hypothesis: {hyp}")
                    # Store for next iteration; this iteration tries prf as fallback
                    s["_hypothesis_override"] = (strat2, nq)
                    new_passages = bm25.retrieve_prf(reformulated or question, k=20)
                    with hypothesis_log.open("a") as fh:
                        fh.write(json.dumps({
                            "iteration": iteration, "topic_id": tid,
                            "strategy_suggested": strat2, "hypothesis": hyp,
                            "next_query": nq,
                        }) + "\n")

            except Exception as e:
                print(f"      Strategy error: {e}")
                new_passages = None

            if new_passages:
                new_passages = dedup_passages(new_passages)
                new_passages_by_topic[tid] = new_passages
            strategy_used[tid] = strategy

        # ── Judge new passages ────────────────────────────────────────────────
        judge_new_passages(client, args.model, topics, qrels,
                           new_passages_by_topic, qrels_cache)

        # ── Ratchet update ────────────────────────────────────────────────────
        improved = 0
        for tid, new_passages in new_passages_by_topic.items():
            new_score = score_topic(new_passages, qrels, tid)
            old_score = state[tid]["best_ndcg"]
            state[tid]["strategy_history"].append({
                "iteration": iteration,
                "strategy": strategy_used.get(tid, "unknown"),
                "before": old_score,
                "after": new_score,
            })
            state[tid]["iteration"] = iteration

            strategy_name = strategy_used.get(tid, "unknown")
            hypothesis = STRATEGY_HYPOTHESES.get(strategy_name, "")
            with iteration_log.open("a") as fh:
                fh.write(json.dumps({
                    "iteration": iteration,
                    "topic_id": tid,
                    "strategy": strategy_name,
                    "hypothesis": hypothesis,
                    "ndcg_before": old_score,
                    "ndcg_after": new_score,
                    "improved": new_score > old_score,
                    "failure_type": state[tid].get("failure_type", ""),
                }) + "\n")

            if new_score > old_score:
                state[tid]["best_ndcg"] = new_score
                state[tid]["best_passages"] = new_passages
                state[tid]["_consecutive_fails"] = 0
                improved += 1
                print(f"    [{tid}] IMPROVED: {old_score:.3f} → {new_score:.3f} "
                      f"(+{new_score-old_score:.3f})")
            else:
                state[tid]["_consecutive_fails"] = state[tid].get("_consecutive_fails", 0) + 1
                cf = state[tid]["_consecutive_fails"]
                print(f"    [{tid}] no gain:  {old_score:.3f} → {new_score:.3f}  "
                      f"(fail streak: {cf})")
                # Trigger adaptive hypothesis after N consecutive failures
                if cf >= args.adaptive_after and state[tid].get("_hypothesis_override") is None:
                    print(f"      → Triggering adaptive hypothesis (streak={cf})")
                    strat2, hyp, nq = get_adaptive_strategy(
                        client, args.model, tid, topics.get(tid, ""),
                        state[tid]["failure_type"],
                        state[tid]["strategy_history"],
                        state[tid]["best_passages"], qrels
                    )
                    state[tid]["_hypothesis_override"] = (strat2, nq)
                    state[tid]["_consecutive_fails"] = 0
                    print(f"      Hypothesis: {hyp}")
                    print(f"      Next strategy: {strat2}"
                          + (f"  query: {nq[:60]}" if nq else ""))
                    with hypothesis_log.open("a") as fh:
                        fh.write(json.dumps({
                            "iteration": iteration, "topic_id": tid,
                            "trigger": "consecutive_fails",
                            "strategy_suggested": strat2, "hypothesis": hyp,
                            "next_query": nq,
                        }) + "\n")

        save_state()

        # Write iter run file
        iter_run_path = out_dir / f"iter_{iteration:03d}_run.txt"
        with iter_run_path.open("w") as f:
            for tid, s in sorted(state.items()):
                for rank, doc_id, text in s["best_passages"]:
                    f.write(f"{tid};{rank};{doc_id};{text}\n")

        mean = current_mean()
        print(f"\nIteration {iteration} done: mean nDCG@20 = {mean:.4f}  "
              f"({improved}/{len(new_passages_by_topic)} topics improved)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"{'='*65}")
    print(f"Final mean nDCG@20: {current_mean():.4f}")

    print("\nPer-topic summary:")
    for tid in sorted(state.keys()):
        s = state[tid]
        hist = s["strategy_history"]
        init = hist[0]["before"] if hist else s["best_ndcg"]
        print(f"  [{tid}] {init:.3f} → {s['best_ndcg']:.3f}  ft={s['failure_type']}")

    final_path = out_dir / "final_run.txt"
    with final_path.open("w") as f:
        for tid, s in sorted(state.items()):
            for rank, doc_id, text in s["best_passages"]:
                f.write(f"{tid};{rank};{doc_id};{text}\n")
    print(f"\nFinal run saved to {final_path}")

    summary = {
        "final_mean_ndcg": current_mean(),
        "topics": {
            tid: {
                "final_ndcg": s["best_ndcg"],
                "failure_type": s["failure_type"],
                "n_improved": sum(1 for h in s["strategy_history"] if h["after"] > h["before"]),
                "strategy_history": s["strategy_history"],
            }
            for tid, s in state.items()
        }
    }
    with (out_dir / "loop_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Strategy effectiveness summary — aggregate wins/attempts per strategy
    strategy_stats: dict[str, dict] = defaultdict(lambda: {"attempts": 0, "wins": 0, "gain": 0.0})
    for tid_data in summary["topics"].values():
        for h in tid_data["strategy_history"]:
            st = h.get("strategy", "unknown")
            strategy_stats[st]["attempts"] += 1
            if h["after"] > h["before"]:
                strategy_stats[st]["wins"] += 1
                strategy_stats[st]["gain"] += h["after"] - h["before"]
    strategy_summary = {
        st: {
            "attempts": d["attempts"],
            "wins": d["wins"],
            "win_rate": d["wins"] / d["attempts"] if d["attempts"] else 0.0,
            "total_gain": round(d["gain"], 4),
            "hypothesis": STRATEGY_HYPOTHESES.get(st, ""),
        }
        for st, d in sorted(strategy_stats.items(), key=lambda x: -x[1]["wins"])
    }
    with (out_dir / "strategy_summary.json").open("w") as f:
        json.dump(strategy_summary, f, indent=2)
    print(f"\nStrategy summary saved to {out_dir / 'strategy_summary.json'}")


if __name__ == "__main__":
    main()
