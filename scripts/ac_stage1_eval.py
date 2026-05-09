#!/usr/bin/env python3
"""
Stage 1 — Evaluate candidate answer quality.

Three intermediate evals:

  E1.1 — Spotcheck: write 10 random topics' candidate answers + question + top-3
         pool passages to a file for manual review.

  E1.2 — Oracle-match: compare candidate_answer to the cached oracle answer
         (from scripts/ac_oracle.py) using semantic match via Sonnet.
         Cheap accuracy proxy. Cost ~$1 per pool.

  E1.3 — Sampling consistency: run Stage 1 again with temperature=0.7 (uses a
         separate cache key); measure agreement between deterministic and
         sampled answers. Free if you've already pre-computed both.

Usage:
    python scripts/ac_stage1_eval.py \\
        --stage1 data/processed/ac_stage1_tier1.json \\
        --pool data/processed/ac_pool_tier1.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

MATCH_PROMPT = """You are checking if two answers refer to the same fact.

Question: {question}

Answer A: {answer_a}
Answer B: {answer_b}

Are these answers equivalent (same name/value/quotation, paraphrasing or \
formatting differences allowed)? Reply JSON only:
{{"match": true|false, "reason": "<one short sentence>"}}"""


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        return {m.group(1).strip(): m.group(2).strip()
                for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL)}
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def call_match(client: anthropic.Anthropic, model: str, prompt: str,
               retries: int = 3, min_interval: float = 1.0) -> dict:
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model=model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - t0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            text = r.content[0].text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            return json.loads(text)
        except Exception:
            time.sleep(2 ** attempt)
    return {"match": None, "reason": "match call failed"}


# ─── E1.1 ─────────────────────────────────────────────────────────────────────


def eval_e11(stage1: dict, pool: dict, topics: dict, n: int, output_path: Path) -> dict:
    rng = random.Random(0)
    qids = rng.sample(list(stage1.keys()), min(n, len(stage1)))
    lines = []
    for qid in qids:
        s = stage1[qid]
        passages = pool.get(qid, [])
        question = topics.get(qid, "?")
        lines.append(f"\n{'═'*80}")
        lines.append(f"Topic {qid}: {question}")
        lines.append(f"{'═'*80}")
        lines.append(f"  Candidate answer: {s.get('answer','(none)')}")
        lines.append(f"  Confidence:       {s.get('confidence',0)}")
        lines.append(f"  Reason:           {s.get('reason','')}")
        lines.append(f"\n  Top-3 pool passages:")
        for p in passages[:3]:
            lines.append(f"    [CE={p.get('ce_score','?'):>5.2f}] {p['source_run']}#{p['source_rank']} doc={p['doc_id']}")
            lines.append(f"      {p['text'][:200]}{'...' if len(p['text'])>200 else ''}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\n─── E1.1 Spotcheck (manual) ─────────────────────────────────")
    print(f"  {n} random topics' candidate answers written to:")
    print(f"    {output_path}")
    print(f"  Manual review: count how many candidate answers look correct.")
    return {"sample_qids": qids, "output_path": str(output_path)}


# ─── E1.2 ─────────────────────────────────────────────────────────────────────


def eval_e12(stage1: dict, oracle: dict, topics: dict,
             cache_path: Path, model: str) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = anthropic.Anthropic()

    matches = 0
    no_oracle = 0
    no_candidate = 0
    decisions: dict[str, dict] = {}
    new_calls = 0

    print(f"\n─── E1.2 Oracle-match accuracy proxy ───────────────────────")
    for qid, s in sorted(stage1.items()):
        cand = (s.get("answer") or "").strip()
        o = oracle.get(qid, {})
        oracle_ans = (o.get("answer") or "").strip()

        if not oracle_ans:
            no_oracle += 1
            decisions[qid] = {"match": None, "reason": "no oracle answer"}
            continue
        if not cand:
            no_candidate += 1
            decisions[qid] = {"match": False, "reason": "candidate is empty"}
            continue

        # Cheap exact-match short-circuit
        norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
        if norm(cand) == norm(oracle_ans) or norm(cand) in norm(oracle_ans) or norm(oracle_ans) in norm(cand):
            decisions[qid] = {"match": True, "reason": "string match"}
            matches += 1
            continue

        # Otherwise, ask Sonnet
        cache_key = f"{qid}:{cand[:80]}:{oracle_ans[:80]}"
        if cache_key in cache:
            verdict = cache[cache_key]
        else:
            prompt = MATCH_PROMPT.format(question=topics.get(qid, ""), answer_a=cand, answer_b=oracle_ans)
            verdict = call_match(client, model, prompt)
            cache[cache_key] = verdict
            new_calls += 1
        decisions[qid] = verdict
        if verdict.get("match"):
            matches += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    n_assessable = len(stage1) - no_oracle
    accuracy = matches / n_assessable if n_assessable else 0.0
    print(f"  Topics assessable (oracle answered): {n_assessable}/{len(stage1)}")
    print(f"  Candidate matches oracle:             {matches}/{n_assessable} ({accuracy*100:.1f}%)")
    print(f"  Candidate empty (refusals):           {no_candidate}")
    print(f"  No oracle (excluded):                 {no_oracle}")
    print(f"  New match-check API calls:            {new_calls}")
    return {
        "accuracy_proxy": accuracy,
        "matches": matches,
        "n_assessable": n_assessable,
        "n_refusals": no_candidate,
        "decisions": decisions,
    }


# ─── E1.3 ─────────────────────────────────────────────────────────────────────


def eval_e13(stage1_det: dict, stage1_sample: dict | None) -> dict:
    print(f"\n─── E1.3 Sampling consistency ──────────────────────────────")
    if stage1_sample is None:
        print(f"  SKIPPED — no sampled run provided (use --stage1-sampled).")
        print(f"  To run: re-execute scripts/ac_stage1_candidate.py with")
        print(f"    --temperature 0.7 --cache-suffix sample1 --output ...")
        return {"skipped": True}

    agree = 0
    total = 0
    for qid, sd in stage1_det.items():
        ss = stage1_sample.get(qid)
        if ss is None:
            continue
        total += 1
        a, b = (sd.get("answer") or "").lower().strip(), (ss.get("answer") or "").lower().strip()
        if a == b or (a and b and (a in b or b in a)):
            agree += 1
    pct = agree / total if total else 0.0
    print(f"  Agreement (det vs temp=0.7 sample): {agree}/{total} ({pct*100:.1f}%)")
    return {"agreement_rate": pct, "agreement": agree, "n": total}


# ─── Driver ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Output of ac_stage1_candidate.py")
    parser.add_argument("--pool", required=True, help="The pool used to generate stage1")
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--oracle", default=str(BASE / "data/eval/ac/oracle_answers.json"))
    parser.add_argument("--stage1-sampled", default=None,
                        help="Optional sampled-temperature stage1 output for E1.3")
    parser.add_argument("--n-spotcheck", type=int, default=10)
    parser.add_argument("--match-model", default="claude-haiku-4-5-20251001",
                        help="Cheap model for the answer-equivalence check")
    parser.add_argument("--match-cache", default=str(BASE / "data/eval/ac/stage1_match_cache.json"))
    parser.add_argument("--output", default=None,
                        help="Path to write eval report JSON (defaults to <stage1>.eval.json)")
    args = parser.parse_args()

    stage1_path = Path(args.stage1)
    stage1_data = json.loads(stage1_path.read_text())
    stage1 = stage1_data["topics"]

    pool_data = json.loads(Path(args.pool).read_text())
    pool = pool_data["topics"]
    topics = load_topics(Path(args.topics))
    oracle = json.loads(Path(args.oracle).read_text())

    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  Stage 1 Quality Report                                            ║")
    print(f"║  Stage1 file: {stage1_path.name:<53}║")
    pool_meta = pool_data.get("_meta", {})
    print(f"║  Pool teams: {str(pool_meta.get('teams', '?')):<54}║")
    print(f"║  Topics: {len(stage1):<59}║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")

    spotcheck_path = stage1_path.with_suffix(".spotcheck.txt")
    e11 = eval_e11(stage1, pool, topics, args.n_spotcheck, spotcheck_path)

    e12 = eval_e12(stage1, oracle, topics, Path(args.match_cache), args.match_model)

    sampled = None
    if args.stage1_sampled:
        sampled = json.loads(Path(args.stage1_sampled).read_text())["topics"]
    e13 = eval_e13(stage1, sampled)

    # Summary
    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  SUMMARY                                                            ║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    print(f"║  E1.1 Spotcheck:                ○ MANUAL (10 topics on disk)       ║")
    if e12["accuracy_proxy"] >= 0.65:
        e12_status = f"✓ {e12['accuracy_proxy']*100:.1f}% match (target ≥65%)"
    else:
        e12_status = f"✗ {e12['accuracy_proxy']*100:.1f}% match"
    print(f"║  E1.2 Oracle-match accuracy:    {e12_status:<35} ║")
    if e13.get("skipped"):
        e13_status = "○ skipped (no sampled run)"
    elif e13.get("agreement_rate", 0) >= 0.5:
        e13_status = f"✓ {e13['agreement_rate']*100:.1f}% agreement"
    else:
        e13_status = f"✗ {e13['agreement_rate']*100:.1f}% agreement"
    print(f"║  E1.3 Sampling consistency:     {e13_status:<35} ║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")

    output_path = Path(args.output) if args.output else stage1_path.with_suffix(".eval.json")
    output_path.write_text(json.dumps({"e11": e11, "e12": e12, "e13": e13}, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
