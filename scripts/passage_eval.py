#!/usr/bin/env python3
"""
Passage-level evaluation using Claude as judge.

For each topic, sends the top-K passages to Claude and gets:
  - nugget_count: number of distinct atomic relevant facts across all passages
  - grades: per-passage relevance grade (0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant)
  - ndcg20: nDCG@20 computed from the grades

Two eval modes:
  --model claude-haiku-4-5-20251001   (Phase 2: fast, cheap, sweep many conditions)
  --model claude-sonnet-4-6           (Phase 3: accurate, top-4 final selection)

Cache key: sha256(topic_id + ordered passage texts) — any change triggers a new call.
Cache is shared across all runs in a session, keyed by (model, hash).

Usage:
    # Evaluate one run (Haiku, fast)
    python scripts/passage_eval.py \\
        --topics-file data/synthetic_val/synthetic_topics_val250.json \\
        --run data/runs/own_bm25_100_overlap.txt \\
        --output data/eval/own_bm25_100_overlap.json \\
        --model claude-haiku-4-5-20251001

    # Evaluate multiple runs and print comparison table
    python scripts/passage_eval.py \\
        --topics-file data/synthetic_val/synthetic_topics_val250.json \\
        --run data/runs/run1.txt data/runs/run2.txt \\
        --output data/eval/comparison.json
"""

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

EVAL_PROMPT = """\
You are evaluating retrieved passages for a question.

Question: {question}

Retrieved passages (in ranked order):
{passages}

Your task: evaluate these passages for answer quality.

**Part 1 — Nuggets**: Count distinct atomic facts that are:
- Directly relevant to answering the question
- Actually stated in at least one passage (no inference)
- Deduplicated (same fact in multiple passages = 1 nugget)

**Part 2 — Per-passage grades**: Grade EACH passage 0–3:
  3 = Highly relevant: directly answers the question or provides key supporting fact
  2 = Relevant: clearly related, adds useful context
  1 = Marginal: tangentially related, limited value
  0 = Not relevant: off-topic or no useful information

Reply with JSON only:
{{
  "nuggets": ["fact 1", "fact 2", ...],
  "nugget_count": <integer>,
  "grades": [<grade for passage 1>, <grade for passage 2>, ...],
  "reasoning": "<one sentence>"
}}

The grades array must have exactly {n_passages} elements, one per passage."""


def load_topics(topics_path: Path | None, topics_file: Path | None) -> dict[str, str]:
    if topics_file and topics_file.exists():
        data = json.loads(topics_file.read_text())
        topics = {}
        for rec in data:
            tid = rec.get("loop_topic_id") or rec.get("topic_id", "")
            q = rec.get("question", "")
            if tid and q:
                topics[tid] = q
        return topics
    if topics_path and topics_path.exists():
        content = topics_path.read_text(encoding="utf-8")
        if content.strip().startswith("<"):
            pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
            return {qid.strip(): q.strip() for qid, q in pairs}
    return {}


def load_run(run_path: Path, top_k: int = 20) -> dict[str, list[tuple]]:
    run = defaultdict(list)
    with run_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", 3)
            if len(parts) != 4:
                continue
            tid, rank, doc_id, text = parts
            rank = int(rank)
            if rank <= top_k:
                run[tid].append((rank, doc_id, text))
    return {tid: sorted(v) for tid, v in run.items()}


def passage_hash(topic_id: str, passages: list[tuple]) -> str:
    content = topic_id + "|" + "|".join(t for _, _, t in passages)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def ndcg_at_k(grades: list[int], k: int = 20) -> float:
    """Compute nDCG@k from a list of relevance grades (already in rank order)."""
    grades = grades[:k]
    if not grades or max(grades) == 0:
        return 0.0

    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    ideal = sorted(grades, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(grades) / idcg


def call_judge(client: anthropic.Anthropic, model: str, question: str,
               passages: list[tuple], retries: int = 4) -> dict:
    n = len(passages)
    passage_text = "\n".join(
        f"[{i+1}] {text[:200]}" for i, (_, _, text) in enumerate(passages)
    )
    prompt = EVAL_PROMPT.format(question=question, passages=passage_text, n_passages=n)

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(text)
            grades = data.get("grades", [0] * n)
            # Pad/trim grades to match n passages
            grades = (grades + [0] * n)[:n]
            return {
                "nuggets": data.get("nuggets", []),
                "nugget_count": int(data.get("nugget_count", len(data.get("nuggets", [])))),
                "grades": grades,
                "ndcg20": ndcg_at_k(grades),
                "reasoning": data.get("reasoning", ""),
            }
        except json.JSONDecodeError:
            m = re.search(r'"nugget_count"\s*:\s*(\d+)', raw)
            if m:
                return {"nuggets": [], "nugget_count": int(m.group(1)),
                        "grades": [0] * n, "ndcg20": 0.0, "reasoning": "parse_error"}
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
        except Exception as e:
            wait = min(60, 2 ** attempt * 5)
            if attempt < retries - 1:
                print(f"  API error: {e}, retry in {wait}s...")
                time.sleep(wait)
            else:
                return {"nuggets": [], "nugget_count": 0,
                        "grades": [0] * n, "ndcg20": 0.0, "reasoning": "api_error"}
    return {"nuggets": [], "nugget_count": 0, "grades": [0] * n,
            "ndcg20": 0.0, "reasoning": "failed"}


def summarize(results: dict) -> dict:
    nuggets = [v["nugget_count"] for v in results.values() if "nugget_count" in v]
    ndcgs = [v["ndcg20"] for v in results.values() if "ndcg20" in v]
    if not nuggets:
        return {}
    return {
        "mean_nuggets": round(sum(nuggets) / len(nuggets), 3),
        "mean_ndcg20": round(sum(ndcgs) / len(ndcgs), 4),
        "total_topics": len(nuggets),
        "zero_nugget_topics": nuggets.count(0),
        "nugget_dist": {str(k): nuggets.count(k) for k in sorted(set(nuggets))},
    }


def evaluate_run(run_path: Path, topics: dict, client: anthropic.Anthropic,
                 model: str, cache: dict, top_k: int = 20) -> dict:
    run = load_run(run_path, top_k)
    results = {}
    total = len(run)
    cache_key_prefix = model[:20]

    for i, (tid, passages) in enumerate(sorted(run.items())):
        question = topics.get(tid, "")
        if not question:
            continue

        h = f"{cache_key_prefix}:{passage_hash(tid, passages)}"
        if h in cache:
            results[tid] = cache[h]
            r = cache[h]
            print(f"  [{i+1}/{total}] {tid} (cached): "
                  f"{r['nugget_count']} nuggets, nDCG={r['ndcg20']:.3f}")
            continue

        result = call_judge(client, model, question, passages)
        result["topic_id"] = tid
        result["question"] = question[:80]
        cache[h] = result
        results[tid] = result
        print(f"  [{i+1}/{total}] {tid}: {result['nugget_count']} nuggets, "
              f"nDCG={result['ndcg20']:.3f}  — {question[:55]}")

        # Rate limiting: Haiku allows ~50 req/min, Sonnet ~50 req/min
        time.sleep(1.2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--topics-file", default=None)
    parser.add_argument("--run", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="claude-haiku-4-5-20251001 (fast) or claude-sonnet-4-6 (accurate)")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    topics = load_topics(
        Path(args.topics) if args.topics else None,
        Path(args.topics_file) if args.topics_file else None,
    )
    print(f"Loaded {len(topics)} topics")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = output_path.parent / "eval_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Loaded {len(cache)} cached judgments")

    client = anthropic.Anthropic()
    all_results = {}

    for run_file in args.run:
        run_path = Path(run_file)
        print(f"\n── Evaluating {run_path.name} [{args.model}] ──")
        results = evaluate_run(run_path, topics, client, args.model, cache, args.top_k)
        summary = summarize(results)
        all_results[run_path.stem] = {"summary": summary, "topics": results}
        print(f"  → mean_nuggets={summary['mean_nuggets']:.3f}, "
              f"mean_nDCG={summary['mean_ndcg20']:.4f}, "
              f"zero-nugget={summary['zero_nugget_topics']}/{summary['total_topics']}")

        cache_path.write_text(json.dumps(cache, indent=2))

    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")

    print("\n── Summary ──")
    print(f"{'Run':<45} {'Nuggets':>8} {'nDCG@20':>8} {'Zero':>6}")
    print("-" * 70)
    for name, data in all_results.items():
        s = data["summary"]
        print(f"{name:<45} {s['mean_nuggets']:>8.3f} {s['mean_ndcg20']:>8.4f} "
              f"{s['zero_nugget_topics']:>4}/{s['total_topics']}")


if __name__ == "__main__":
    main()
