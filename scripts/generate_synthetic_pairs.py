#!/usr/bin/env python3
"""
Generate synthetic (query, positive_passage) pairs using Claude Haiku.

Strategy:
  1. For each topic, take passages judged relevant (score >= 1) from qrels.
  2. Ask Haiku to generate 3-5 paraphrase queries for each relevant passage.
  3. Also ask Haiku to generate a hypothetical Wikipedia passage for each topic
     (same as HyDE but we collect it as a positive for the embedding model).
  4. Save as JSONL: {"query": "...", "positive": "...", "doc_id": "..."}

These synthetic pairs augment the real judged data for bi-encoder fine-tuning.

Usage:
    python scripts/generate_synthetic_pairs.py \
        --qrels data/autoresearch/qrels_loop.json \
        --topics data/raw/r2c2topics.txt \
        --output data/processed/synthetic_pairs.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

PARAPHRASE_PROMPT = """\
Given this movie/Star Wars question and a relevant passage, write 4 different \
paraphrase questions that would also be answered by the passage. \
Vary vocabulary and phrasing. Each paraphrase should be a complete question.

Original question: {question}
Relevant passage: {passage}

Reply with exactly 4 paraphrases, one per line, no numbering or bullets."""

HYDE_PROMPT = """\
Write a 2-4 sentence Wikipedia-style passage that directly answers this question \
about a movie or Star Wars. Use encyclopedic language, include specific names/dates.

Question: {question}

Passage:"""


def load_topics(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    if next(l for l in content.splitlines() if l.strip()).startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    import csv
    topics = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or "").strip()
            if tid and q:
                topics[tid] = q
    return topics


def call_claude(client, model, prompt, min_interval=1.4):
    for attempt in range(4):
        start = time.monotonic()
        try:
            r = client.messages.create(
                model=model, max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            elapsed = time.monotonic() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return r.content[0].text.strip()
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            print(f"  API error ({e}), retrying in {wait}s")
            time.sleep(wait)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default=str(BASE / "data/autoresearch/qrels_loop.json"))
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", default=str(BASE / "data/processed/synthetic_pairs.jsonl"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--min-rel", type=int, default=1,
                        help="Minimum relevance score to use as positive (default: 1)")
    parser.add_argument("--paraphrases-per-passage", type=int, default=4)
    parser.add_argument("--include-hyde", action="store_true", default=True,
                        help="Also generate HyDE passage as synthetic positive")
    args = parser.parse_args()

    topics = load_topics(Path(args.topics))
    with open(args.qrels) as f:
        raw = json.load(f)

    # Build (topic_id, doc_id, text, score) list of positives
    positives = []
    for tid, passages in raw.items():
        for key, score in passages.items():
            if score >= args.min_rel:
                doc_id, text = key.split("\x00", 1)
                positives.append((tid, doc_id, text, score))

    print(f"Topics: {len(topics)}")
    print(f"Positive passages to use: {len(positives)} (rel >= {args.min_rel})")

    client = anthropic.Anthropic()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    real_pairs = 0  # original (question, passage) pairs
    synth_pairs = 0  # generated paraphrases

    with output_path.open("w") as fout:
        # 1. Write the original (question, passage) pairs as ground truth
        print("\nWriting original judged pairs ...")
        for tid, doc_id, text, score in positives:
            question = topics.get(tid, "")
            if not question:
                continue
            fout.write(json.dumps({
                "query": question,
                "positive": text,
                "doc_id": doc_id,
                "topic_id": tid,
                "score": score,
                "source": "judged",
            }) + "\n")
            real_pairs += 1

        # 2. Generate paraphrase queries for each positive passage
        print(f"\nGenerating paraphrase queries for {len(positives)} passages ...")
        for i, (tid, doc_id, text, score) in enumerate(positives):
            question = topics.get(tid, "")
            if not question:
                continue
            prompt = PARAPHRASE_PROMPT.format(question=question, passage=text[:300])
            raw_resp = call_claude(client, args.model, prompt)
            paraphrases = [l.strip() for l in raw_resp.splitlines()
                           if l.strip() and len(l.strip()) > 10][:args.paraphrases_per_passage]
            for para in paraphrases:
                fout.write(json.dumps({
                    "query": para,
                    "positive": text,
                    "doc_id": doc_id,
                    "topic_id": tid,
                    "score": score,
                    "source": "paraphrase",
                }) + "\n")
                synth_pairs += 1
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(positives)} passages processed, "
                      f"{synth_pairs} synthetic pairs generated so far")

        # 3. HyDE: generate a hypothetical passage per topic as an additional positive
        if args.include_hyde:
            print(f"\nGenerating HyDE passages for {len(topics)} topics ...")
            for tid, question in sorted(topics.items()):
                prompt = HYDE_PROMPT.format(question=question)
                hyde_passage = call_claude(client, args.model, prompt)
                if hyde_passage:
                    fout.write(json.dumps({
                        "query": question,
                        "positive": hyde_passage,
                        "doc_id": f"hyde_{tid}",
                        "topic_id": tid,
                        "score": 2,
                        "source": "hyde",
                    }) + "\n")
                    synth_pairs += 1

    total = real_pairs + synth_pairs
    print(f"\nDone. Written {total} pairs to {output_path}")
    print(f"  Original judged pairs: {real_pairs}")
    print(f"  Synthetic (paraphrase + HyDE): {synth_pairs}")


if __name__ == "__main__":
    main()
