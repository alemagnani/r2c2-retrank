#!/usr/bin/env python3
"""
Stage 1.5 — Answer-conditional pool refinement.

Re-rank a broad pool using the candidate answer from Stage 1, then keep top-K.

Three algorithms:
  (a) ce      — CE(query|answer, passage). Free if we cache; one CE inference pass.
  (b) sonnet  — Sonnet rates each passage 0–3 for "supports the answer". Expensive.
  (c) lexical — original CE × (1 + α · token-overlap with answer). Free.

If Stage 1 produced a refusal (empty answer), the broad pool is passed through
unchanged with a flag indicating no refinement happened.

Usage:
    python scripts/ac_stage15_refine.py \\
        --pool data/processed/ac_pool_tier1.json \\
        --stage1 data/processed/ac_stage1_tier1.json \\
        --algo ce --top-k 30 \\
        --output data/processed/ac_pool_tier1_refined_ce.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "with", "by", "from", "as", "if",
    "it", "its", "their", "his", "her", "they", "them", "we", "you",
    "what", "who", "where", "when", "how", "why", "which",
}


# ─── Algorithm (a): CE on (query|answer) ──────────────────────────────────────


def run_ce(pool: dict, stage1: dict, topics: dict, top_k: int,
           ce_cache_path: Path) -> dict:
    """Score passages with CE on the combined (question + answer) input.

    Caches scores in `data/eval/ac_stage15_ce.pkl` keyed by sha256.
    """
    import pickle
    print(f"Loading CE model (cross-encoder/ms-marco-MiniLM-L-12-v2)...")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)

    cache: dict[str, float] = {}
    if ce_cache_path.exists():
        cache = pickle.loads(ce_cache_path.read_bytes())
        print(f"  loaded {len(cache):,} cached scores from {ce_cache_path}")

    # Collect all (combined_query, passage_text) pairs we need
    pairs: list[tuple[str, str, str, int]] = []  # (qid, passage_idx, combined, text)
    queries: dict[str, str] = {}
    for qid, passages in pool.items():
        s = stage1.get(qid, {})
        ans = (s.get("answer") or "").strip()
        if not ans:
            continue  # refusal — skip refinement
        combined = f"{topics.get(qid, '')} | {ans}"
        queries[qid] = combined
        for i, p in enumerate(passages):
            key = hashlib.sha256(f"{combined}\x1f{p['text'][:300]}".encode()).hexdigest()[:24]
            if key in cache:
                continue
            pairs.append((qid, i, combined, p["text"][:1500], key))

    print(f"  {len(pairs):,} new (combined-query, passage) pairs need scoring")
    if pairs:
        BATCH = 64
        inputs = [(c, t) for (_, _, c, t, _) in pairs]
        scores = model.predict(inputs, batch_size=BATCH, show_progress_bar=True)
        for (qid, i, combined, text, key), s in zip(pairs, scores):
            cache[key] = float(s)
        ce_cache_path.parent.mkdir(parents=True, exist_ok=True)
        ce_cache_path.write_bytes(pickle.dumps(cache))
        print(f"  saved cache → {ce_cache_path}")

    # Build refined pool
    refined: dict[str, dict] = {}
    for qid, passages in pool.items():
        s = stage1.get(qid, {})
        ans = (s.get("answer") or "").strip()
        if not ans:
            refined[qid] = {"refined": False, "reason": "stage1 refused", "passages": passages[:top_k]}
            continue
        combined = queries[qid]
        scored = []
        for p in passages:
            key = hashlib.sha256(f"{combined}\x1f{p['text'][:300]}".encode()).hexdigest()[:24]
            scored.append((cache.get(key, -1e9), p))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s_v, p in scored[:top_k]:
            p2 = dict(p)
            p2["refined_score"] = s_v
            p2["refined_algo"] = "ce"
            out.append(p2)
        refined[qid] = {"refined": True, "passages": out}

    return refined


# ─── Algorithm (b): Sonnet pointwise ──────────────────────────────────────────


SONNET_RATE_PROMPT = """Question: {question}
Candidate answer: {answer}

Passage: {passage}

Rate 0–3 how strongly the passage SUPPORTS the candidate answer:
  3 = directly states the answer
  2 = strongly implies the answer (multiple supporting facts)
  1 = mentions the answer entity but does not directly support it
  0 = unrelated or contradictory

Reply JSON only: {{"score": <0|1|2|3>, "reason": "<one short phrase>"}}"""


def run_sonnet(pool: dict, stage1: dict, topics: dict, top_k: int,
               cache_path: Path, model_name: str = "claude-sonnet-4-6",
               max_passages_to_rate: int = 50) -> dict:
    """`run_sonnet` is a misnomer — works with any LLM model passed as model_name."""
    """Rate each passage with Sonnet 0–3. Cap at max_passages_to_rate per topic."""
    import anthropic

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = anthropic.Anthropic()

    refined: dict[str, dict] = {}
    n_calls = 0

    for qid in sorted(pool.keys()):
        passages = pool[qid][:max_passages_to_rate]
        s = stage1.get(qid, {})
        ans = (s.get("answer") or "").strip()
        if not ans:
            refined[qid] = {"refined": False, "reason": "stage1 refused",
                            "passages": passages[:top_k]}
            continue

        question = topics.get(qid, "")
        scored = []
        for p in passages:
            key = hashlib.sha256(f"{question}\x1f{ans}\x1f{p['text'][:300]}".encode()).hexdigest()[:24]
            if key in cache:
                v = cache[key]
            else:
                prompt = SONNET_RATE_PROMPT.format(question=question, answer=ans, passage=p["text"][:1500])
                v = _call_sonnet(client, model_name, prompt)
                cache[key] = v
                n_calls += 1
                if n_calls % 20 == 0:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
                    print(f"  [{qid}] {n_calls} new calls so far")
            try:
                score = int(v.get("score", 0))
            except (TypeError, ValueError):
                score = 0
            scored.append((score, p))

        scored.sort(key=lambda x: -x[0])
        out = []
        for s_v, p in scored[:top_k]:
            p2 = dict(p)
            p2["refined_score"] = s_v
            p2["refined_algo"] = "sonnet"
            out.append(p2)
        refined[qid] = {"refined": True, "passages": out}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    print(f"  Sonnet new calls: {n_calls}")
    return refined


def _call_sonnet(client, model: str, prompt: str, retries: int = 3,
                 min_interval: float = 1.0) -> dict:
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model=model, max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            dt = time.monotonic() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)
            text = r.content[0].text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            return json.loads(text)
        except Exception:
            time.sleep(2 ** attempt)
    return {"score": 0, "reason": "call failed"}


# ─── Algorithm (c): Lexical answer-presence ───────────────────────────────────


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if t not in STOPWORDS and len(t) > 2}


def run_lexical(pool: dict, stage1: dict, top_k: int, alpha: float = 0.5) -> dict:
    """score = ce_orig * (1 + alpha · |answer_tokens ∩ passage_tokens| / |answer_tokens|)"""
    refined: dict[str, dict] = {}
    for qid, passages in pool.items():
        s = stage1.get(qid, {})
        ans = (s.get("answer") or "").strip()
        if not ans:
            refined[qid] = {"refined": False, "reason": "stage1 refused", "passages": passages[:top_k]}
            continue
        ans_toks = _tokenize(ans)
        if not ans_toks:
            refined[qid] = {"refined": False, "reason": "no distinctive answer tokens", "passages": passages[:top_k]}
            continue

        scored = []
        for p in passages:
            ce0 = p.get("ce_score") or 0
            p_toks = _tokenize(p["text"])
            overlap = len(ans_toks & p_toks) / max(len(ans_toks), 1)
            new_score = ce0 * (1.0 + alpha * overlap) + alpha * overlap  # add small bonus when ce0 ~ 0
            scored.append((new_score, p))
        scored.sort(key=lambda x: -x[0])

        out = []
        for s_v, p in scored[:top_k]:
            p2 = dict(p)
            p2["refined_score"] = s_v
            p2["refined_algo"] = "lexical"
            out.append(p2)
        refined[qid] = {"refined": True, "passages": out}

    return refined


# ─── Driver ───────────────────────────────────────────────────────────────────


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        return {m.group(1).strip(): m.group(2).strip()
                for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL)}
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--algo", choices=["ce", "sonnet", "lexical"], required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ce-cache", default=str(BASE / "data/eval/ac_stage15_ce.pkl"))
    parser.add_argument("--sonnet-cache", default=str(BASE / "data/eval/ac_stage15_sonnet.json"))
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="(lexical) bonus weight on answer-token overlap")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="(sonnet algo) LLM to use for pointwise scoring")
    args = parser.parse_args()

    pool_data = json.loads(Path(args.pool).read_text())
    pool = pool_data["topics"]
    pool_meta = pool_data.get("_meta", {})

    stage1_data = json.loads(Path(args.stage1).read_text())
    stage1 = stage1_data["topics"]
    topics = load_topics(Path(args.topics))

    print(f"Pool:        {Path(args.pool).name} ({pool_meta.get('teams', '?')})")
    print(f"Stage 1:     {Path(args.stage1).name}")
    print(f"Algorithm:   {args.algo}")
    print(f"Top-K:       {args.top_k}")
    print()

    if args.algo == "ce":
        refined = run_ce(pool, stage1, topics, args.top_k, Path(args.ce_cache))
    elif args.algo == "sonnet":
        # Cache file separated by model so Haiku and Sonnet results don't collide
        cache_path = Path(args.sonnet_cache)
        if args.model != "claude-sonnet-4-6":
            cache_path = cache_path.with_name(cache_path.stem + f"_{args.model.split('-')[1]}.json")
        refined = run_sonnet(pool, stage1, topics, args.top_k, cache_path,
                              model_name=args.model)
    else:
        refined = run_lexical(pool, stage1, args.top_k, args.alpha)

    n_refined = sum(1 for r in refined.values() if r["refined"])
    print(f"\nRefined {n_refined}/{len(refined)} topics (rest passed-through due to Stage 1 refusal)")

    output = {
        "_meta": {
            "pool_file": Path(args.pool).name,
            "stage1_file": Path(args.stage1).name,
            "algo": args.algo,
            "top_k": args.top_k,
            "n_topics": len(refined),
            "n_refined": n_refined,
        },
        "topics": refined,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    print(f"Wrote refined pool to {output_path}")


if __name__ == "__main__":
    main()
