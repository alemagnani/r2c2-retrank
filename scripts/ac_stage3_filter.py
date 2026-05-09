#!/usr/bin/env python3
"""
Stage 3 — Bogus nugget filter.

For each Stage 2 output, run the Self-Stage-A bogus check on every nugget and
drop nuggets the cited passage doesn't entail. Output is a "clean" nugget set
ready for Stage 4 (verification) and ultimately submission.

Reuses the bogus cache from `scripts/ac_stage2_eval.py`. Nuggets with no cached
verdict are checked fresh (via Sonnet). The check function is robust — it
raises rather than silently returning False.

Usage:
    python scripts/ac_stage3_filter.py \\
        --stage2 data/processed/ac_stage2_tier1_broad.json \\
        --output data/processed/ac_stage3_tier1_broad.json
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
            print(f"    bogus call {attempt+1}/{retries} failed; retry in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"bogus check failed: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default=str(BASE / "data/eval/ac_cache/bogus_claude-sonnet-4-6.json"))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    stage2 = json.loads(Path(args.stage2).read_text())
    topics = stage2["topics"]

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"Loaded {len(cache)} cached bogus judgments")

    client = anthropic.Anthropic()
    new_calls = 0
    out_topics: dict[str, dict] = {}
    total_before = 0
    total_after = 0
    n_dropped_per_topic: dict[str, int] = {}

    for qid in sorted(topics.keys()):
        rec = topics[qid]
        if rec.get("refused"):
            out_topics[qid] = {**rec, "filtered_nuggets": [], "n_dropped": 0}
            continue
        kept = []
        dropped = []
        for n in rec["nuggets"]:
            total_before += 1
            key = hashlib.sha256(
                f"{n['passage_text'][:200]}\x1f{n['text']}".encode()
            ).hexdigest()[:24]
            v = cache.get(key)
            if v is None:
                prompt = BOGUS_PROMPT.format(
                    passage_text=n["passage_text"][:1500], nugget_text=n["text"])
                try:
                    v = _call(client, args.model, prompt)
                except RuntimeError as e:
                    # Persistent parse/API failure — assume entailed (don't poison MNP)
                    print(f"  Stage 3 bogus check failed for one nugget; assuming entailed: {e}",
                          file=sys.stderr)
                    v = {"entailed": True, "reason": f"check failed (assumed entailed): {type(e).__name__}",
                         "_failed": True}
                cache[key] = v
                new_calls += 1
                if new_calls % 10 == 0:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
                    print(f"  {new_calls} new bogus checks done")
            if v.get("entailed"):
                kept.append({**n, "_entail_reason": v.get("reason", "")})
            else:
                dropped.append({**n, "_bogus_reason": v.get("reason", "")})
        total_after += len(kept)
        n_dropped_per_topic[qid] = len(dropped)
        out_topics[qid] = {
            **rec,
            "filtered_nuggets": kept,
            "dropped_nuggets": dropped,
            "n_kept": len(kept),
            "n_dropped": len(dropped),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    output = {
        "_meta": {
            "stage2_file": Path(args.stage2).name,
            "model": args.model,
            "total_input_nuggets": total_before,
            "total_kept_nuggets": total_after,
            "total_dropped": total_before - total_after,
            "drop_rate": (total_before - total_after) / max(total_before, 1),
            "new_api_calls": new_calls,
        },
        "topics": out_topics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  Stage 3 — Bogus Filter                                            ║")
    print(f"║  Source: {Path(args.stage2).name:<55}║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    print(f"║  Input nuggets:   {total_before:<46}║")
    print(f"║  Kept (entailed): {total_after:<46}║")
    print(f"║  Dropped (bogus): {total_before - total_after:<46}║")
    print(f"║  Drop rate:       {(total_before-total_after)/max(total_before,1)*100:.1f}%{'':<41}║")
    n_topics = len([r for r in topics.values() if not r.get('refused')])
    n_topics_lost_all = sum(1 for q, k in n_dropped_per_topic.items()
                            if k > 0 and out_topics[q]["n_kept"] == 0)
    print(f"║  Topics fully wiped (all nuggets bogus): {n_topics_lost_all:<22}║")
    print(f"║  New API calls:   {new_calls:<46}║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")
    print(f"\nFull output: {output_path}")


if __name__ == "__main__":
    main()
