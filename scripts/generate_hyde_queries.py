#!/usr/bin/env python3
"""
HyDE: Hypothetical Document Embedding query generation.

For each topic, asks Claude to write a hypothetical Wikipedia/Wookieepedia-style
passage that would answer the question. These passages are then used as retrieval
queries instead of (or alongside) the raw question.

Output: data/processed/hyde_queries.json
  {topic_id: {"question": "...", "hypothesis": "..."}}

Usage:
    python scripts/generate_hyde_queries.py \
        --topics data/raw/r2c2topics.txt \
        --model claude-haiku-4-5-20251001 \
        --output data/processed/hyde_queries.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

HYDE_PROMPT = """\
You are a Wikipedia/Wookieepedia author. Write a short encyclopedic passage \
(2-4 sentences, factual, specific) that directly contains the answer to this question \
about a movie or Star Wars:

Question: {question}

Write ONLY the passage text, no preamble, no explanation. \
The passage should read like a fragment from a Wikipedia article — \
include the specific names, dates, or facts that answer the question."""


def load_topics(path: Path) -> list[tuple[str, str]]:
    content = path.read_text(encoding="utf-8")
    first_line = next(l for l in content.splitlines() if l.strip())
    if first_line.strip().startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return [(qid.strip(), q.strip()) for qid, q in pairs]
    import csv
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    topics = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or row.get("query") or "").strip()
            if tid and q:
                topics.append((tid, q))
    return topics


def generate_hypothesis(client: anthropic.Anthropic, model: str, question: str,
                         min_interval: float = 1.3, retries: int = 4) -> str:
    prompt = HYDE_PROMPT.format(question=question)
    for attempt in range(retries):
        start = time.monotonic()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return response.content[0].text.strip()
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            if attempt < retries - 1:
                print(f"  Error, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  Failed: {e}. Using question as fallback.")
                return question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--output", default=str(BASE / "data/processed/hyde_queries.json"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output if interrupted
    existing = {}
    if output_path.exists():
        with output_path.open() as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} topics already done")

    topics = load_topics(Path(args.topics))
    print(f"Loaded {len(topics)} topics")
    print(f"Using model: {args.model}")

    client = anthropic.Anthropic()
    results = dict(existing)

    for topic_id, question in topics:
        if topic_id in results:
            continue
        print(f"\n[{topic_id}] {question}")
        hypothesis = generate_hypothesis(client, args.model, question)
        print(f"  → {hypothesis[:120]}...")
        results[topic_id] = {"question": question, "hypothesis": hypothesis}
        # Save after each topic
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} hypotheses saved to {output_path}")


if __name__ == "__main__":
    main()
