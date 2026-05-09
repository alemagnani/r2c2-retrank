#!/usr/bin/env python3
"""
LLM-as-judge evaluation for R2C2 runs.

Strategy B: Pool top-k results from all run files, ask an LLM to grade each
(question, passage) pair on a 0/1/2 scale, then compute nDCG@20.

Grading scale:
  2 = highly relevant (directly answers the question)
  1 = partially relevant (related but doesn't fully answer)
  0 = not relevant

Usage:
    # Grade with claude-haiku-4-5 (cheap, fast)
    python scripts/evaluate_runs.py \
        --topics data/raw/r2c2topics.txt \
        --runs data/runs/alemagnani-PO-1.txt data/runs/alemagnani-PO-2.txt \
        --model claude-haiku-4-5-20251001 \
        --output data/eval/qrels_haiku.json

    # Grade with claude-sonnet-4-6 (higher quality)
    python scripts/evaluate_runs.py \
        --topics data/raw/r2c2topics.txt \
        --runs data/runs/alemagnani-PO-*.txt \
        --model claude-sonnet-4-6 \
        --output data/eval/qrels_sonnet.json

    # Just compute nDCG from existing qrels (skip API calls)
    python scripts/evaluate_runs.py \
        --topics data/raw/r2c2topics.txt \
        --runs data/runs/alemagnani-PO-*.txt \
        --qrels data/eval/qrels_haiku.json \
        --no-judge
"""

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

JUDGE_PROMPT = """\
You are an information retrieval judge. Given a question about movies or Star Wars, \
rate how relevant the following passage is for answering it.

Question: {question}

Passage: {passage}

Rate the relevance on this scale:
2 = Highly relevant: the passage directly answers or strongly helps answer the question
1 = Partially relevant: the passage is related to the topic but does not clearly answer the question
0 = Not relevant: the passage does not help answer the question

Reply with ONLY the single digit 0, 1, or 2. No explanation."""


def load_topics(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    # JSON synthetic topics
    if path.suffix == ".json":
        import json as _json
        records = _json.loads(content)
        topics = {}
        for rec in records:
            tid = rec.get("loop_topic_id") or rec.get("topic_id", "")
            q   = rec.get("question", "")
            if tid and q:
                topics[tid] = q
        return topics
    first_line = next(l for l in content.splitlines() if l.strip())
    if first_line.strip().startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    import csv
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    topics = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or row.get("query") or "").strip()
            if tid and q:
                topics[tid] = q
    return topics


def load_runs(run_files: list[Path], top_k: int) -> dict[str, dict[str, tuple[str, int]]]:
    """
    Returns {run_name: {(topic_id, doc_id, text_key): rank}}.
    Also returns the pool: {topic_id: set of (doc_id, text) pairs}.
    """
    runs = {}
    for run_file in run_files:
        name = run_file.stem
        run = {}
        with run_file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(";", 3)
                if len(parts) != 4:
                    continue
                topic_id, rank, doc_id, text = parts
                rank = int(rank)
                if rank <= top_k:
                    run[(topic_id, doc_id, text)] = rank
        runs[name] = run
    return runs


def pool_passages(runs: dict) -> dict[str, list[tuple[str, str]]]:
    """Return {topic_id: [(doc_id, text), ...]} — union across all runs."""
    pool = defaultdict(dict)
    for run in runs.values():
        for (topic_id, doc_id, text), rank in run.items():
            pool[topic_id][(doc_id, text)] = True
    return {tid: list(passages.keys()) for tid, passages in pool.items()}


def judge_passage(client: anthropic.Anthropic, model: str, question: str, passage: str,
                  retries: int = 5, min_interval: float = 1.3) -> int:
    """Judge one passage. min_interval enforces ~46 req/min to stay under the 50/min limit."""
    prompt = JUDGE_PROMPT.format(question=question, passage=passage)
    for attempt in range(retries):
        start = time.monotonic()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - start
            # Pace requests to stay within rate limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            text = response.content[0].text.strip()
            if text in {"0", "1", "2"}:
                return int(text)
            match = re.search(r"[012]", text)
            if match:
                return int(match.group())
            print(f"  Unexpected response: {repr(text)}, defaulting to 0")
            return 0
        except Exception as e:
            wait = min(60, 2 ** attempt * 5)  # 5s, 10s, 20s, 40s, 60s
            if attempt < retries - 1:
                print(f"  Rate limit / error, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {e}. Defaulting to 0.")
                return 0


def dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_rels: list[int], k: int) -> float:
    ideal = sorted(ranked_rels, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(ranked_rels, k) / ideal_dcg


def compute_ndcg(runs: dict, qrels: dict, k: int = 20) -> dict[str, dict]:
    """Returns {run_name: {topic_id: ndcg, ..., 'mean': float}}."""
    results = {}
    for run_name, run in runs.items():
        topic_scores = defaultdict(list)
        # Group by topic, sort by rank
        by_topic = defaultdict(list)
        for (topic_id, doc_id, text), rank in run.items():
            rel = qrels.get(topic_id, {}).get((doc_id, text), 0)
            by_topic[topic_id].append((rank, rel))

        ndcg_scores = {}
        for topic_id, ranked in by_topic.items():
            ranked.sort(key=lambda x: x[0])
            rels = [r for _, r in ranked]
            ndcg_scores[topic_id] = ndcg_at_k(rels, k)

        ndcg_scores["mean"] = sum(ndcg_scores.values()) / len(ndcg_scores) if ndcg_scores else 0.0
        results[run_name] = ndcg_scores
    return results


def print_results(results: dict, k: int):
    print(f"\n{'='*60}")
    print(f"nDCG@{k} Results")
    print(f"{'='*60}")
    run_names = sorted(results.keys())
    for run_name in run_names:
        scores = results[run_name]
        mean = scores.get("mean", 0)
        print(f"\n{run_name}: mean nDCG@{k} = {mean:.4f}")
        for topic_id in sorted(s for s in scores if s != "mean"):
            print(f"  Topic {topic_id}: {scores[topic_id]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model for judging")
    parser.add_argument("--output", default=str(BASE / "data/eval/qrels.json"),
                        help="Where to save/load qrels")
    parser.add_argument("--qrels", default=None,
                        help="Load existing qrels JSON instead of calling API")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judging, just compute nDCG from --qrels")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top passages per run to consider")
    parser.add_argument("--ndcg-k", type=int, default=20)
    args = parser.parse_args()

    topics = load_topics(Path(args.topics))
    print(f"Loaded {len(topics)} topics")

    run_files = [Path(r) for r in args.runs]
    runs = load_runs(run_files, args.top_k)
    print(f"Loaded {len(runs)} runs: {list(runs.keys())}")

    # Cache file is always keyed by model name, stored alongside --output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    cache_path = output_path.parent / f"cache_{model_slug}.json"

    def _serialize(qrels_dict):
        return {
            tid: {f"{doc_id}\x00{text}": v for (doc_id, text), v in passages.items()}
            for tid, passages in qrels_dict.items()
        }

    def _deserialize(raw):
        return {
            tid: {tuple(k.split("\x00", 1)): v for k, v in passages.items()}
            for tid, passages in raw.items()
        }

    def _save_cache(qrels_dict):
        with cache_path.open("w") as f:
            json.dump(_serialize(qrels_dict), f)

    # Load or build qrels
    if args.no_judge:
        src = Path(args.qrels) if args.qrels else cache_path
        print(f"Loading qrels from {src} ...")
        with src.open() as f:
            qrels = _deserialize(json.load(f))
    else:
        # Load existing cache (partial or complete)
        qrels = defaultdict(dict)
        if cache_path.exists():
            print(f"Resuming from cache: {cache_path}")
            with cache_path.open() as f:
                qrels.update(_deserialize(json.load(f)))

        pool = pool_passages(runs)
        total = sum(len(v) for v in pool.values())
        already = sum(len(v) for v in qrels.values())
        print(f"Pool: {total} unique (topic, passage) pairs — {already} already cached")
        print(f"Using model: {args.model}")

        client = anthropic.Anthropic()
        judged = 0
        for topic_id, passages in sorted(pool.items()):
            question = topics.get(topic_id, "")
            if not question:
                continue
            new_passages = [(doc_id, text) for doc_id, text in passages
                            if (doc_id, text) not in qrels.get(topic_id, {})]
            if not new_passages:
                continue
            print(f"\nTopic {topic_id}: {question[:60]}... ({len(new_passages)} new / {len(passages)} total)")
            for doc_id, text in new_passages:
                score = judge_passage(client, args.model, question, text)
                qrels[topic_id][(doc_id, text)] = score
                judged += 1
            # Save after each topic so we can resume on interruption
            _save_cache(qrels)
            print(f"  Saved cache ({already + judged}/{total} total judged)")

        print(f"\nCache saved to {cache_path}")

    # Also write to --output for explicit reference
    with output_path.open("w") as f:
        json.dump(_serialize(qrels), f, indent=2)
    print(f"Qrels written to {output_path}")

    results = compute_ndcg(runs, qrels, args.ndcg_k)
    print_results(results, args.ndcg_k)

    # Save summary
    summary_path = output_path.parent / f"ndcg_results_{output_path.stem}.json"
    with summary_path.open("w") as f:
        json.dump({k: {str(t): s for t, s in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
