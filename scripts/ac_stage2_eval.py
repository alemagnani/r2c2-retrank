#!/usr/bin/env python3
"""
Stage 2 — Evaluate extracted nuggets.

Three evals:
  E2.1 Self-Stage A check: for every nugget, ask Sonnet whether the cited
       passage entails the nugget. Bogus rate = % of nuggets not entailed.
       Target ≤ 15%.
  E2.2 Hand spotcheck: write 10 random topics' nuggets to a file for review.
  E2.3 Count distribution: histogram of nuggets per topic.

Usage:
    python scripts/ac_stage2_eval.py \\
        --stage2 data/processed/ac_stage2_tier1_broad.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

BOGUS_PROMPT = """You are evaluating whether a passage entails a factual claim (a "nugget").

Passage: {passage_text}

Nugget: {nugget_text}

Does the passage entail the nugget? An entailed nugget is a claim that follows \
directly from the passage text without external knowledge. Paraphrasing is allowed; \
adding facts not in the passage is not. If the nugget contradicts the passage or \
asserts something the passage does not support, it is not entailed.

Reply with JSON only:
{{"entailed": true|false, "reason": "<one short sentence>"}}"""


def _call(client, model: str, prompt: str, retries: int = 5,
          min_interval: float = 1.2) -> dict:
    """Robust call with longer back-off. Raise on persistent failure rather
    than silently mark entailed=False (which corrupts the bogus rate)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model=model, max_tokens=200,
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
        except Exception as e:
            last_err = e
            wait = min(60, 5 * (2 ** attempt))
            print(f"    bogus call attempt {attempt+1}/{retries} failed ({type(e).__name__}: {e}); retry in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"bogus check failed after {retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", required=True)
    parser.add_argument("--cache", default=str(BASE / "data/eval/ac_cache/bogus_claude-sonnet-4-6.json"),
                        help="Shared bogus cache (also used by ac_eval.py)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--n-spotcheck", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stage2 = json.loads(Path(args.stage2).read_text())
    topics = stage2["topics"]

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"Loaded {len(cache)} cached bogus judgments from {cache_path}")

    client = anthropic.Anthropic()

    # ─── E2.1 — bogus check ───────────────────────────────────────────────
    print(f"\n─── E2.1 Self-Stage A bogus check ─────────────────────────────")
    bogus_per_topic: dict[str, int] = {}
    nuggets_per_topic: dict[str, int] = {}
    bogus_examples: list[dict] = []
    new_calls = 0

    for qid in sorted(topics.keys()):
        rec = topics[qid]
        if rec.get("refused"):
            bogus_per_topic[qid] = 0
            nuggets_per_topic[qid] = 0
            continue
        nuggets = rec.get("nuggets", [])
        nuggets_per_topic[qid] = len(nuggets)
        bogus = 0
        for n in nuggets:
            key = hashlib.sha256(
                f"{n['passage_text'][:200]}\x1f{n['text']}".encode()
            ).hexdigest()[:24]
            if key in cache:
                v = cache[key]
            else:
                prompt = BOGUS_PROMPT.format(
                    passage_text=n["passage_text"][:1500],
                    nugget_text=n["text"])
                v = _call(client, args.model, prompt)
                cache[key] = v
                new_calls += 1
                if new_calls % 10 == 0:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
                    print(f"  {new_calls} new bogus checks done")
            if not v.get("entailed", True):
                bogus += 1
                bogus_examples.append({
                    "qid": qid, "nugget": n["text"][:100],
                    "passage": n["passage_text"][:200],
                    "reason": v.get("reason", "")[:120],
                })
        bogus_per_topic[qid] = bogus

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    total_nuggets = sum(nuggets_per_topic.values())
    total_bogus = sum(bogus_per_topic.values())
    bogus_rate = total_bogus / total_nuggets if total_nuggets else 0
    pass_thresh = bogus_rate <= 0.15

    print(f"\n  Total nuggets:     {total_nuggets}")
    print(f"  Bogus nuggets:     {total_bogus}")
    print(f"  Overall bogus rate: {bogus_rate*100:.1f}%   {'✓ PASS' if pass_thresh else '✗ FAIL'} (target ≤15%)")
    print(f"  Topics with bogus: {sum(1 for v in bogus_per_topic.values() if v > 0)}/{len(topics)}")
    print(f"  New bogus-check calls: {new_calls}")

    # Show 5 bogus examples
    if bogus_examples:
        print(f"\n  Sample bogus nuggets:")
        for e in bogus_examples[:5]:
            print(f"    {e['qid']}: nugget=\"{e['nugget']}\"")
            print(f"        passage: \"{e['passage'][:150]}\"")
            print(f"        reason:  {e['reason']}")

    # ─── E2.3 — count distribution ────────────────────────────────────────
    print(f"\n─── E2.3 Nugget count distribution ──────────────────────────")
    counts = list(nuggets_per_topic.values())
    counts_sorted = sorted(counts)
    n_t = len(counts)
    print(f"  median: {counts_sorted[n_t//2]}  p25: {counts_sorted[n_t//4]}  p75: {counts_sorted[3*n_t//4]}")
    print(f"  min: {counts_sorted[0]}  max: {counts_sorted[-1]}  mean: {sum(counts)/n_t:.2f}")
    histogram = Counter(counts)
    print(f"  histogram:")
    for k in sorted(histogram):
        print(f"    {k:>3}: {'█' * histogram[k]:<30} ({histogram[k]})")

    # ─── E2.2 — spotcheck ─────────────────────────────────────────────────
    print(f"\n─── E2.2 Spotcheck (manual) ─────────────────────────────────")
    rng = random.Random(42)
    qids_sc = rng.sample([q for q, r in topics.items() if not r.get("refused")],
                          min(args.n_spotcheck, len(topics)))
    sc_lines = []
    for qid in qids_sc:
        rec = topics[qid]
        sc_lines.append(f"\n{'═'*80}\nTopic {qid}: candidate answer = {rec['answer']}\n{'═'*80}")
        for i, n in enumerate(rec["nuggets"], 1):
            sc_lines.append(f"  [{i}] cite={n['passage_key'][0]}#{n['passage_key'][1]} doc={n['doc_id']}")
            sc_lines.append(f"      nugget: \"{n['text']}\"")
            sc_lines.append(f"      passage: \"{n['passage_text'][:200]}{'...' if len(n['passage_text'])>200 else ''}\"")
    sc_path = Path(args.stage2).with_suffix(".spotcheck.txt")
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sc_path.write_text("\n".join(sc_lines))
    print(f"  {args.n_spotcheck} topics' nuggets written to {sc_path}")

    # Summary
    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  SUMMARY — {Path(args.stage2).name:<53}║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    print(f"║  Topics:               {len(topics):<41}║")
    print(f"║  Refused:              {sum(1 for r in topics.values() if r.get('refused')):<41}║")
    print(f"║  Total nuggets:        {total_nuggets:<41}║")
    print(f"║  Bogus rate:           {bogus_rate*100:.1f}%   {'✓ PASS' if pass_thresh else '✗ FAIL':<29} ║")
    print(f"║  Mean nuggets/topic:   {sum(counts)/n_t:.2f}{'':<37}║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")

    output = {
        "stage2_file": str(Path(args.stage2).name),
        "total_nuggets": total_nuggets,
        "total_bogus": total_bogus,
        "bogus_rate": bogus_rate,
        "n_topics_with_bogus": sum(1 for v in bogus_per_topic.values() if v > 0),
        "mean_nuggets_per_topic": sum(counts) / n_t if n_t else 0,
        "count_histogram": dict(histogram),
        "bogus_per_topic": bogus_per_topic,
        "nuggets_per_topic": nuggets_per_topic,
    }
    out_path = Path(args.output) if args.output else Path(args.stage2).with_suffix(".eval.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
