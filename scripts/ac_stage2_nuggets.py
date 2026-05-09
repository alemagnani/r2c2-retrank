#!/usr/bin/env python3
"""
Stage 2 — Targeted nugget extraction.

For each topic, given (question, candidate_answer, top-K passages), Sonnet
emits 3–8 atomic nuggets. Each nugget cites exactly one passage via
`(source_run, source_rank)` so the official Stage A bogus check can verify
entailment per (passage, nugget) pair.

Inputs:
  - Stage 1 output (candidate answer per topic)
  - Pool: either a broad pool from Stage 0 or a refined pool from Stage 1.5
    (auto-detected by structure)

Output:
  Per-topic list of {nugget_text, passage_key=(source_run, source_rank), reasoning}.
  Topics where Stage 1 refused → empty nugget list.

Usage:
    python scripts/ac_stage2_nuggets.py \\
        --pool data/processed/ac_pool_tier1_refined_ce.json \\
        --stage1 data/processed/ac_stage1_tier1.json \\
        --output data/processed/ac_stage2_tier1_refined_ce.json
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

NUGGET_PROMPT = """You are extracting nuggets from passages to ground a candidate answer.

Question: {question}

Candidate answer: {candidate_answer}

Passages (numbered with citation keys; some may be irrelevant):
{passages_block}

Definition of a nugget:
- A SINGLE atomic factual claim
- Directly RELEVANT to deriving the candidate answer
- ENTAILED by exactly ONE of the passages above (do NOT use external knowledge)
- Stated concisely (paraphrasing allowed; do not add facts beyond the passage)
- A nugget that is mostly true but adds a wrong detail (wrong year, wrong name, \
wrong number) is BOGUS — do not include it

Extract 3–8 nuggets that together support the candidate answer. For EACH nugget:
- cite EXACTLY ONE passage by its label (P1, P2, …) — the one that most directly \
entails it
- if multiple passages could be cited, pick the one with the most direct support

Important: do NOT include a nugget if no single passage entails it. Quality over quantity.

Reply with JSON only:
{{
  "nuggets": [
    {{"text": "<concise atomic claim>", "cite": "P<N>", "reason": "<one short phrase>"}},
    ...
  ]
}}"""


def _normalise_pool(pool_data: dict) -> dict[str, list[dict]]:
    """Return {qid: [passage, ...]}.

    Broad pools (Stage 0) have `topics[qid] = [passage, ...]`.
    Refined pools (Stage 1.5) have `topics[qid] = {"refined": bool, "passages": [...]}`.
    """
    topics = pool_data["topics"]
    out: dict[str, list[dict]] = {}
    for qid, v in topics.items():
        if isinstance(v, list):
            out[qid] = v
        else:
            out[qid] = v.get("passages", [])
    return out


def passages_block(passages: list[dict], top_k: int = 30, max_chars: int = 1800) -> str:
    lines = []
    for i, p in enumerate(passages[:top_k], 1):
        text = p["text"]
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        lines.append(f"[P{i}] team={p['source_team']} run={p['source_run']} rank={p['source_rank']}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def call_llm(client: anthropic.Anthropic, model: str, prompt: str,
             retries: int = 4, min_interval: float = 1.2) -> dict:
    last_err = None
    last_text = ""
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model=model, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            dt = time.monotonic() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)
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
    return {"nuggets": [], "_error": f"parse failed: {last_err}"}


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        return {m.group(1).strip(): m.group(2).strip()
                for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL)}
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def signature(question: str, answer: str, passages: list[dict], top_k: int) -> str:
    h = hashlib.sha256()
    h.update(question.encode())
    h.update(b"\x1f")
    h.update(answer.encode())
    h.update(b"\x1f")
    for p in passages[:top_k]:
        h.update(p["text"][:200].encode("utf-8", errors="replace"))
        h.update(b"\x1e")
    return h.hexdigest()[:24]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, help="Broad or refined pool JSON")
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    pool_data = json.loads(Path(args.pool).read_text())
    pool = _normalise_pool(pool_data)
    pool_meta = pool_data.get("_meta", {})

    stage1_data = json.loads(Path(args.stage1).read_text())
    stage1 = stage1_data["topics"]

    topics = load_topics(Path(args.topics))

    cache_path = Path(args.cache) if args.cache else Path(args.output + ".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"Pool:       {Path(args.pool).name}  ({pool_meta.get('algo', 'broad')})")
    print(f"Stage 1:    {Path(args.stage1).name}")
    print(f"Top-K:      {args.top_k}")
    print(f"Cache:      {cache_path} ({len(cache)} entries)")

    client = anthropic.Anthropic()
    results: dict[str, dict] = {}
    new_calls = 0
    n_refused = 0

    for i, qid in enumerate(sorted(pool.keys())):
        question = topics.get(qid, "")
        s1 = stage1.get(qid, {})
        ans = (s1.get("answer") or "").strip()

        if not ans:
            results[qid] = {"answer": "", "nuggets": [], "refused": True}
            n_refused += 1
            continue

        passages = pool[qid][:args.top_k]
        sig = signature(question, ans, passages, args.top_k)
        cache_key = f"{args.model}:{sig}"

        if cache_key in cache:
            data = cache[cache_key]
        else:
            prompt = NUGGET_PROMPT.format(
                question=question, candidate_answer=ans,
                passages_block=passages_block(passages, args.top_k))
            data = call_llm(client, args.model, prompt)
            cache[cache_key] = data
            new_calls += 1
            if new_calls % 5 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
                print(f"  [{i+1}/{len(pool)}] {qid}: {len(data.get('nuggets', []))} nuggets")

        # Resolve cite=P<N> -> passage_key
        nuggets = []
        for n in data.get("nuggets", []):
            cite = (n.get("cite") or "").strip()
            m = re.match(r"P(\d+)", cite)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if not (0 <= idx < len(passages)):
                continue
            p = passages[idx]
            nuggets.append({
                "text": (n.get("text") or "").strip(),
                "passage_key": [p["source_run"], p["source_rank"]],
                "passage_text": p["text"],
                "doc_id": p["doc_id"],
                "cite_label": cite,
                "reason": n.get("reason", ""),
            })
        results[qid] = {
            "answer": ans,
            "confidence_a": s1.get("confidence", 0),
            "nuggets": nuggets,
            "refused": False,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    output = {
        "_meta": {
            "pool_file": Path(args.pool).name,
            "pool_meta": pool_meta,
            "stage1_file": Path(args.stage1).name,
            "model": args.model,
            "top_k": args.top_k,
            "n_topics": len(results),
            "n_refused": n_refused,
            "new_api_calls": new_calls,
        },
        "topics": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    counts = [len(r["nuggets"]) for r in results.values() if not r["refused"]]
    if counts:
        avg = sum(counts) / len(counts)
        zero = sum(1 for c in counts if c == 0)
        print(f"\nWrote {output_path}")
        print(f"  Refused: {n_refused}/{len(results)}")
        print(f"  Mean nuggets/topic: {avg:.2f}")
        print(f"  Topics with 0 nuggets (despite non-refusal): {zero}")
        print(f"  New API calls: {new_calls}")


if __name__ == "__main__":
    main()
