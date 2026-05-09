#!/usr/bin/env python3
"""
Stage 1 — Candidate answer generation.

For each topic, send the question + the broad pool of passages to Sonnet and
get back a single concise answer string + self-rated confidence (0-100) +
short reasoning. The candidate answer is what Stage 1.5 conditions on, and
Stage 2 uses to extract targeted nuggets.

Cached: SHA256(pool_signature, model) so re-runs are free. Different pools
produce different cache keys, so we can run this on all 5 pools.

Usage:
    python scripts/ac_stage1_candidate.py \\
        --pool data/processed/ac_pool_tier1.json \\
        --output data/processed/ac_stage1_tier1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

CANDIDATE_PROMPT = """You are answering a movie question using ONLY the passages provided. Do not use \
external knowledge to fill gaps.

Question: {question}

Passages (numbered, with citation keys; truncated where long):
{passages_block}

Tasks:
1. Determine the answer. Be concise — a name, phrase, sentence, or short \
quotation matching the granularity the question implies. For counting or \
arithmetic questions, output ONLY the numeric or list answer (do NOT show your \
arithmetic in the answer field — that goes in "reason").
2. If the passages do not contain a clear answer, output:
   {{"answer": "", "confidence": 0, "reason": "passages do not support an answer"}}
   (Do NOT use external knowledge to invent an answer.)
3. Otherwise, rate your confidence 0–100 based ONLY on how strongly the \
passages support the answer (not your prior knowledge).

CRITICAL: Reply with JSON ONLY. No prose before or after. No "I need to..." or \
"Let me think...". Output the JSON object first; if you need to show reasoning, \
put it inside the "reason" field of the JSON.

Reply with JSON only:
{{
  "answer": "<answer string or empty>",
  "confidence": <0-100 int>,
  "reason": "<one short sentence>"
}}"""


def passage_block(passages: list[dict], max_chars: int = 1800) -> str:
    """Format passages for the prompt. Truncate very long passages.

    Each passage is labelled with `(P{n})` so Stage 2 can refer back, and shows
    `team={team} rank={rank}` for citation context.
    """
    lines = []
    for i, p in enumerate(passages, 1):
        text = p["text"]
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        lines.append(f"[P{i}] team={p['source_team']} rank={p['source_rank']} doc={p['doc_id']}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def pool_signature(question: str, passages: list[dict]) -> str:
    """Hash representing this exact pool for this topic. Used as cache key."""
    h = hashlib.sha256()
    h.update(question.encode())
    h.update(b"\x00")
    for p in passages:
        h.update(p["text"][:200].encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def call_llm(client: anthropic.Anthropic, model: str, prompt: str,
             retries: int = 4, min_interval: float = 1.2,
             temperature: float | None = None) -> dict:
    last_err: Exception | None = None
    last_text = ""
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            kwargs = dict(model=model, max_tokens=400,
                          messages=[{"role": "user", "content": prompt}])
            if temperature is not None:
                kwargs["temperature"] = temperature
            r = client.messages.create(**kwargs)
            elapsed = time.monotonic() - t0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            text = r.content[0].text.strip()
            last_text = text
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            return json.loads(text)
        except (json.JSONDecodeError, KeyError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            wait = min(60, 5 * 2 ** attempt)
            print(f"  API error: {e}, retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    print(f"  failed text was: {last_text[:200]!r}", file=sys.stderr)
    return {"answer": "", "confidence": 0, "reason": f"parse failed: {last_err}"}


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
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", required=True,
                        help="Output JSON: per-topic candidate answers")
    parser.add_argument("--cache", default=None,
                        help="Cache file (defaults to <output>.cache.json)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-passages", type=int, default=50,
                        help="Cap on # passages per topic to feed (default 50; reduce if context too big)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; default = deterministic")
    parser.add_argument("--cache-suffix", default="",
                        help="Suffix for cache key (e.g., 'sample1') so different temperatures don't collide")
    args = parser.parse_args()

    pool_data = json.loads(Path(args.pool).read_text())
    pool = pool_data["topics"]
    pool_meta = pool_data.get("_meta", {})
    topics = load_topics(Path(args.topics))

    print(f"Pool: {Path(args.pool).name} ({pool_meta.get('teams', '?')})")
    print(f"Topics: {len(pool)} | Model: {args.model}")
    if args.temperature is not None:
        print(f"Temperature: {args.temperature}  Cache suffix: {args.cache_suffix!r}")

    cache_path = Path(args.cache) if args.cache else Path(args.output + ".cache.json")
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Loaded {len(cache)} cached entries from {cache_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    results: dict[str, dict] = {}
    new_calls = 0

    for i, qid in enumerate(sorted(pool.keys())):
        question = topics.get(qid, "")
        if not question:
            print(f"  [{i+1}/{len(pool)}] {qid}: no topic, skipping")
            continue
        passages = pool[qid][:args.max_passages]
        sig = pool_signature(question, passages) + (":" + args.cache_suffix if args.cache_suffix else "")
        key = f"{args.model}:{sig}"

        if key in cache:
            results[qid] = cache[key]
            continue

        prompt = CANDIDATE_PROMPT.format(
            question=question,
            passages_block=passage_block(passages),
        )
        result = call_llm(client, args.model, prompt, temperature=args.temperature)
        result["topic_id"] = qid
        result["question"] = question
        result["n_passages_seen"] = len(passages)
        cache[key] = result
        results[qid] = result
        new_calls += 1
        if new_calls % 5 == 0:
            cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
            print(f"  [{i+1}/{len(pool)}] {qid}: {result.get('answer','')[:60]} (conf={result.get('confidence',0)})")

    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    output = {
        "_meta": {
            "pool_file": str(Path(args.pool).name),
            "pool_meta": pool_meta,
            "model": args.model,
            "temperature": args.temperature,
            "n_topics": len(results),
            "max_passages": args.max_passages,
        },
        "topics": results,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nWrote Stage 1 outputs to {output_path}")
    print(f"  New API calls: {new_calls}")

    # Quick stats
    n_empty = sum(1 for r in results.values() if not r.get("answer"))
    confs = [r.get("confidence", 0) for r in results.values() if r.get("answer")]
    if confs:
        avg_conf = sum(confs) / len(confs)
    else:
        avg_conf = 0
    print(f"\n  Empty answers (refusals): {n_empty}/{len(results)}")
    print(f"  Mean confidence on non-empty: {avg_conf:.1f}")


if __name__ == "__main__":
    main()
