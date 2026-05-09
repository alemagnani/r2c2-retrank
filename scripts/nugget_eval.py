#!/usr/bin/env python3
"""
Nugget-based evaluation for passage retrieval runs.

A "nugget" is a single atomic fact that:
  - Is relevant to answering the question
  - Is distinct (if multiple passages say the same thing, count it once)
  - Is actually present in at least one passage (not inferred)

Sends each (query, top-20 passages) to claude-sonnet-4-6 and counts nuggets.
Cache key = hash of (topic_id, ordered passage texts) — any change triggers a new call.

Usage:
    # Evaluate a run file
    python scripts/nugget_eval.py \\
        --topics data/raw/r2c2topics.txt \\
        --run data/runs/alemagnani-PO-1.txt \\
        --output data/eval/nuggets_PO1.json

    # Evaluate multiple runs and compare
    python scripts/nugget_eval.py \\
        --topics data/raw/r2c2topics.txt \\
        --run data/runs/alemagnani-PO-1.txt data/runs/alemagnani-PO-2.txt \\
        --output data/eval/nuggets_compare.json

    # Use synthetic val65 topics
    python scripts/nugget_eval.py \\
        --topics-file data/synthetic_val/synthetic_topics_val65.json \\
        --run data/runs/bm25_pool500_val65.txt \\
        --output data/eval/nuggets_bm25_pool500.json
"""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

NUGGET_PROMPT = """\
You are evaluating the quality and diversity of retrieved passages for answering a question.

Question: {question}

Retrieved passages:
{passages}

Your task: identify all distinct "nuggets" of information present in these passages that are relevant to answering the question.

Definition of a nugget:
- A single atomic fact that directly helps answer the question
- Must actually appear in at least one passage (do not infer or add outside knowledge)
- If multiple passages express the same fact (even in different words), count it ONCE
- Ignore facts that are irrelevant to the question even if interesting

Examples of nuggets for "Who directed The Godfather?":
  - "Francis Ford Coppola directed The Godfather" ✓
  - "The Godfather was released in 1972" ✗ (not relevant to who directed it)
  - "Al Pacino starred in The Godfather" ✗ (not relevant to director)

Reply with JSON only:
{{
  "nuggets": [
    "concise statement of nugget 1",
    "concise statement of nugget 2"
  ],
  "count": <integer>,
  "reasoning": "<one sentence explaining your choices>"
}}"""


def load_topics(topics_path: Path, topics_file_path: Path | None) -> dict[str, str]:
    """Load topic_id → question mapping."""
    if topics_file_path and Path(topics_file_path).exists():
        data = json.loads(Path(topics_file_path).read_text())
        topics = {}
        for rec in data:
            tid = rec.get("loop_topic_id") or rec.get("topic_id", "")
            q = rec.get("question", "")
            if tid and q:
                topics[tid] = q
        return topics

    content = Path(topics_path).read_text(encoding="utf-8")
    first_line = next(l for l in content.splitlines() if l.strip())
    if first_line.strip().startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    import csv
    topics = {}
    with Path(topics_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t" if topics_path.suffix in {".tsv", ".tab"} else ",")
        for row in reader:
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or "").strip()
            if tid and q:
                topics[tid] = q
    return topics


def load_run(run_path: Path, top_k: int = 20) -> dict[str, list[tuple[int, str, str]]]:
    """Load run file → {topic_id: [(rank, doc_id, text), ...]} sorted by rank."""
    run = defaultdict(list)
    with run_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", 3)
            if len(parts) != 4:
                continue
            topic_id, rank, doc_id, text = parts
            rank = int(rank)
            if rank <= top_k:
                run[topic_id].append((rank, doc_id, text))
    return {tid: sorted(v) for tid, v in run.items()}


def passage_set_hash(topic_id: str, passages: list[tuple]) -> str:
    """Cache key: hash of topic_id + ordered passage texts."""
    content = topic_id + "|" + "|".join(t for _, _, t in passages)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def call_nugget_judge(client: anthropic.Anthropic, model: str, question: str,
                      passages: list[tuple], retries: int = 4) -> dict:
    """Ask Sonnet to count nuggets. Returns {nuggets, count, reasoning}."""
    passage_text = "\n".join(
        f"[{i+1}] {text[:200]}" for i, (_, _, text) in enumerate(passages)
    )
    prompt = NUGGET_PROMPT.format(question=question, passages=passage_text)

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(text)
            return {
                "nuggets": data.get("nuggets", []),
                "count": int(data.get("count", len(data.get("nuggets", [])))),
                "reasoning": data.get("reasoning", ""),
            }
        except json.JSONDecodeError:
            # Try to extract count from text
            m = re.search(r'"count"\s*:\s*(\d+)', raw)
            if m:
                return {"nuggets": [], "count": int(m.group(1)), "reasoning": "parse_error"}
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
        except Exception as e:
            wait = min(60, 2 ** attempt * 5)
            if attempt < retries - 1:
                print(f"  API error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {e}")
                return {"nuggets": [], "count": 0, "reasoning": "api_error"}
    return {"nuggets": [], "count": 0, "reasoning": "failed"}


def evaluate_run(run_path: Path, topics: dict, client: anthropic.Anthropic,
                 model: str, cache: dict, top_k: int = 20) -> dict:
    """Evaluate a single run file. Returns {topic_id: nugget_result}."""
    run = load_run(run_path, top_k)
    results = {}
    total = len(run)

    for i, (tid, passages) in enumerate(sorted(run.items())):
        question = topics.get(tid, "")
        if not question:
            continue

        key = passage_set_hash(tid, passages)
        if key in cache:
            results[tid] = cache[key]
            print(f"  [{i+1}/{total}] {tid} (cached): {cache[key]['count']} nuggets")
            continue

        result = call_nugget_judge(client, model, question, passages)
        result["topic_id"] = tid
        result["question"] = question[:80]
        cache[key] = result
        results[tid] = result

        print(f"  [{i+1}/{total}] {tid}: {result['count']} nuggets  — {question[:60]}")
        time.sleep(1.2)  # ~50 req/min limit

    return results


def summarize(results: dict) -> dict:
    counts = [v["count"] for v in results.values() if "count" in v]
    if not counts:
        return {"mean": 0, "total_topics": 0}
    return {
        "mean_nuggets": round(sum(counts) / len(counts), 3),
        "total_topics": len(counts),
        "min": min(counts),
        "max": max(counts),
        "dist": {str(k): counts.count(k) for k in sorted(set(counts))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--topics-file", default=None)
    parser.add_argument("--run", nargs="+", required=True, help="Run file(s) to evaluate")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    topics = load_topics(Path(args.topics), args.topics_file)
    print(f"Loaded {len(topics)} topics")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared cache across all runs in this invocation
    cache_path = output_path.parent / "nugget_cache.json"
    cache = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        cache = raw
        print(f"Loaded {len(cache)} cached nugget judgments")

    client = anthropic.Anthropic()
    all_results = {}

    for run_file in args.run:
        run_path = Path(run_file)
        print(f"\n── Evaluating {run_path.name} ──")
        results = evaluate_run(run_path, topics, client, args.model, cache, args.top_k)
        summary = summarize(results)
        print(f"  Mean nuggets: {summary['mean_nuggets']:.2f} over {summary['total_topics']} topics")
        all_results[run_path.stem] = {"summary": summary, "topics": results}

        # Save cache after each run
        cache_path.write_text(json.dumps(cache, indent=2))

    # Final output
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")

    print("\n── Summary ──")
    for run_name, data in all_results.items():
        s = data["summary"]
        print(f"  {run_name}: {s['mean_nuggets']:.2f} nuggets/query ({s['total_topics']} topics)")


if __name__ == "__main__":
    main()
