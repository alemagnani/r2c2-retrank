#!/usr/bin/env python3
"""
E0.4 — Knowledge oracle for AC pool evaluation.

Asks Sonnet (no passages, parametric knowledge only) to answer each of the 65
R2C2 questions. Caches results per topic. Used in pool eval to check whether
the pool contains a passage that mentions the likely answer.

Note: oracle answers are NOISY — Sonnet may be wrong on obscure films. We use
the oracle as a sanity-check signal, not a gold standard.

Usage:
    python scripts/ac_oracle.py
    python scripts/ac_oracle.py --model claude-sonnet-4-6 --output data/eval/ac/oracle_answers.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

ORACLE_PROMPT = """You are answering a movie question using only your knowledge.

Question: {question}

Reply with JSON ONLY. Do NOT explain or reason aloud — just the JSON object.

If the question requires counting or arithmetic, perform it silently and
write only the numeric answer in "answer".

If you don't know or are unsure, set "answer" to empty and confidence to 0.

Schema:
{{
  "answer": "<concise answer string or empty>",
  "key_entities": ["<entity1>", "<entity2>", "..."],
  "confidence": <0-100 int>,
  "reason": "<one short sentence, max 15 words>"
}}

Output the JSON object now:"""


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        topics = {}
        for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL):
            topics[m.group(1).strip()] = m.group(2).strip()
        return topics
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def call_sonnet(client: anthropic.Anthropic, model: str, prompt: str,
                retries: int = 4, min_interval: float = 1.2) -> dict:
    last_err: Exception | None = None
    last_text: str = ""
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - t0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            text = r.content[0].text.strip()
            last_text = text
            # Strip code fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            # Extract first {...} block if extra prose surrounds it
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
    # Don't crash the whole run for one topic — return a sentinel
    return {"answer": "", "key_entities": [], "confidence": 0,
            "reason": f"oracle parse failed: {last_err}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--output", default=str(BASE / "data/eval/ac/oracle_answers.json"))
    args = parser.parse_args()

    topics = load_topics(Path(args.topics))
    print(f"Loaded {len(topics)} topics")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if output_path.exists():
        cache = json.loads(output_path.read_text())
        print(f"Loaded {len(cache)} cached oracle answers")

    client = anthropic.Anthropic()
    new_calls = 0
    for i, (qid, q) in enumerate(sorted(topics.items())):
        if qid in cache:
            continue
        prompt = ORACLE_PROMPT.format(question=q)
        result = call_sonnet(client, args.model, prompt)
        cache[qid] = {
            "question": q,
            "answer": result.get("answer", ""),
            "key_entities": result.get("key_entities", []),
            "confidence": result.get("confidence", 0),
            "reason": result.get("reason", ""),
            "model": args.model,
        }
        new_calls += 1
        if new_calls % 5 == 0:
            output_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
            print(f"  [{i+1}/{len(topics)}] {qid}: {result.get('answer', '')[:60]}  (conf={result.get('confidence', 0)})")

    output_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    print(f"\nWrote {len(cache)} oracle answers to {output_path}")
    print(f"  New API calls this run: {new_calls}")

    # Quick stats
    n_empty = sum(1 for v in cache.values() if not v["answer"])
    avg_conf = sum(v["confidence"] for v in cache.values()) / len(cache)
    print(f"  Empty answers (oracle declined): {n_empty}/{len(cache)}")
    print(f"  Mean oracle confidence: {avg_conf:.1f}")


if __name__ == "__main__":
    main()
