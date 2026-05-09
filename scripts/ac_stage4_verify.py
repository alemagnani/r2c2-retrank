#!/usr/bin/env python3
"""
Stage 4 — Verification.

Given (question, kept entailed nuggets — no passages), Sonnet derives an answer
using ONLY the nuggets and rates its own confidence. We compare verifier_answer
to candidate_answer (Stage 1) to produce a match-score signal:
  2 = full match  (semantic equivalence; same name/value)
  1 = partial match
  0 = no match (verifier disagrees or refuses)

This match-score is the calibration signal for Stage 5 (Variant A confidence).
It also probes RQ5: does verification gap correlate with answer correctness?

Usage:
    python scripts/ac_stage4_verify.py \\
        --stage3 data/processed/ac_stage3_tier1_broad.json \\
        --output data/processed/ac_stage4_tier1_broad.json
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

VERIFIER_PROMPT = """You are answering a movie question using ONLY the listed nuggets.
Treat each nugget as a factually correct claim. Do NOT use external knowledge.

Question: {question}

Nuggets:
{nuggets_block}

Tasks:
1. Derive the answer using only the nuggets above. Be concise.
2. If the nuggets are insufficient, output an empty answer.
3. Rate confidence 0–100 based on how directly the nuggets imply the answer.

Reply JSON only:
{{"verifier_answer": "<answer string or empty>", "verifier_confidence": <0-100 int>, "reason": "<one short sentence>"}}"""


MATCH_PROMPT = """Are these two answers equivalent (same name/value/quotation,
paraphrasing or formatting differences allowed)? Question context: {question}

Answer A: {answer_a}
Answer B: {answer_b}

Reply JSON only:
{{"match_score": 0|1|2, "reason": "<one short sentence>"}}

  2 = full match (semantic equivalence — same entity/value)
  1 = partial match (related but not exact)
  0 = no match (different entities/values, or one is empty)"""


def _call(client, model: str, prompt: str, max_tokens: int = 250,
          retries: int = 5, min_interval: float = 1.2) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            r = client.messages.create(
                model=model, max_tokens=max_tokens,
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
            print(f"    call {attempt+1}/{retries} failed; retry in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"call failed: {last_err}")


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def quick_match(a: str, b: str) -> int | None:
    """Cheap exact/substring check. Returns 2/1/0 or None for ambiguous → LLM."""
    na, nb = normalise(a), normalise(b)
    if not na and not nb:
        return 0
    if not na or not nb:
        return 0
    if na == nb:
        return 2
    if na in nb or nb in na:
        return 2
    return None  # LLM decides


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        return {m.group(1).strip(): m.group(2).strip()
                for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL)}
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", required=True)
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-verify", default=str(BASE / "data/eval/ac_cache/stage4_verify.json"))
    parser.add_argument("--cache-match", default=str(BASE / "data/eval/ac_cache/stage4_match.json"))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--match-model", default="claude-haiku-4-5-20251001",
                        help="Cheap model for the equivalence comparison")
    args = parser.parse_args()

    stage3 = json.loads(Path(args.stage3).read_text())
    topics_text = load_topics(Path(args.topics))

    cache_v = json.loads(Path(args.cache_verify).read_text()) if Path(args.cache_verify).exists() else {}
    cache_m = json.loads(Path(args.cache_match).read_text()) if Path(args.cache_match).exists() else {}
    print(f"Loaded {len(cache_v)} verifier cache, {len(cache_m)} match cache")

    client = anthropic.Anthropic()
    out_topics: dict[str, dict] = {}
    n_calls_v = 0
    n_calls_m = 0

    for qid in sorted(stage3["topics"].keys()):
        rec = stage3["topics"][qid]
        question = topics_text.get(qid, "")
        cand_ans = (rec.get("answer") or "").strip()
        nuggets = rec.get("filtered_nuggets", [])

        if rec.get("refused") or not nuggets:
            out_topics[qid] = {
                **{k: v for k, v in rec.items() if k != "dropped_nuggets"},
                "verifier_answer": "",
                "verifier_confidence": 0,
                "match_score": 0,
                "verifier_reason": "no entailed nuggets" if not nuggets else "stage1 refused",
            }
            continue

        # Verifier
        nug_texts = [n["text"] for n in nuggets]
        nug_block = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(nug_texts))
        v_key = hashlib.sha256(
            f"{question}\x1f{'|'.join(nug_texts)}".encode()
        ).hexdigest()[:24]
        if v_key in cache_v:
            v = cache_v[v_key]
        else:
            prompt = VERIFIER_PROMPT.format(question=question, nuggets_block=nug_block)
            try:
                v = _call(client, args.model, prompt)
            except RuntimeError as e:
                # Persistent parse/API failure on this single topic.
                # Treat as a "verifier refused" (match_score=0) rather than crash run.
                print(f"  [{qid}] verifier failed permanently: {e}", file=sys.stderr)
                v = {"verifier_answer": "", "verifier_confidence": 0,
                     "reason": f"verifier failed: {type(e).__name__}"}
            cache_v[v_key] = v
            n_calls_v += 1
            if n_calls_v % 5 == 0:
                Path(args.cache_verify).parent.mkdir(parents=True, exist_ok=True)
                Path(args.cache_verify).write_text(json.dumps(cache_v, indent=1, ensure_ascii=False))
                print(f"  [{qid}] verifier: {v.get('verifier_answer','')[:60]}")

        verif_ans = (v.get("verifier_answer") or "").strip()
        verif_conf = int(v.get("verifier_confidence") or 0)

        # Match score: cheap exact/substring first, fallback to Haiku
        ms = quick_match(cand_ans, verif_ans)
        if ms is None:
            m_key = hashlib.sha256(
                f"{question[:100]}\x1f{cand_ans[:100]}\x1f{verif_ans[:100]}".encode()
            ).hexdigest()[:24]
            if m_key in cache_m:
                m_resp = cache_m[m_key]
            else:
                prompt = MATCH_PROMPT.format(question=question, answer_a=cand_ans, answer_b=verif_ans)
                m_resp = _call(client, args.match_model, prompt, max_tokens=120)
                cache_m[m_key] = m_resp
                n_calls_m += 1
            try:
                ms = int(m_resp.get("match_score", 0))
            except (TypeError, ValueError):
                ms = 0

        out_topics[qid] = {
            **{k: v for k, v in rec.items() if k != "dropped_nuggets"},
            "verifier_answer": verif_ans,
            "verifier_confidence": verif_conf,
            "verifier_reason": v.get("reason", ""),
            "match_score": ms,
        }

    Path(args.cache_verify).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_verify).write_text(json.dumps(cache_v, indent=1, ensure_ascii=False))
    Path(args.cache_match).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_match).write_text(json.dumps(cache_m, indent=1, ensure_ascii=False))

    # Stats
    n_full = sum(1 for r in out_topics.values() if r["match_score"] == 2)
    n_partial = sum(1 for r in out_topics.values() if r["match_score"] == 1)
    n_no = sum(1 for r in out_topics.values() if r["match_score"] == 0)
    n_assess = n_full + n_partial + n_no
    n_refused = sum(1 for r in out_topics.values() if r.get("refused"))

    output = {
        "_meta": {
            "stage3_file": Path(args.stage3).name,
            "model": args.model,
            "match_model": args.match_model,
            "n_topics": len(out_topics),
            "n_full_match": n_full,
            "n_partial_match": n_partial,
            "n_no_match": n_no,
            "n_refused_or_empty": n_refused,
            "new_verify_calls": n_calls_v,
            "new_match_calls": n_calls_m,
        },
        "topics": out_topics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n╔═════════════════════════════════════════════════════════════════╗")
    print(f"║  Stage 4 — Verification                                            ║")
    print(f"║  Source: {Path(args.stage3).name:<55}║")
    print(f"╠═════════════════════════════════════════════════════════════════╣")
    print(f"║  Full match (=2):     {n_full:>3}/{n_assess} ({n_full/max(n_assess,1)*100:.1f}%){'':<25}║")
    print(f"║  Partial match (=1):  {n_partial:>3}/{n_assess} ({n_partial/max(n_assess,1)*100:.1f}%){'':<25}║")
    print(f"║  No match (=0):       {n_no:>3}/{n_assess} ({n_no/max(n_assess,1)*100:.1f}%){'':<25}║")
    print(f"║  Refused/empty:       {n_refused:>3}{'':<41}║")
    print(f"║  New verifier calls:  {n_calls_v:<43}║")
    print(f"║  New match calls:     {n_calls_m:<43}║")
    print(f"╚═════════════════════════════════════════════════════════════════╝")
    print(f"\nFull output: {output_path}")


if __name__ == "__main__":
    main()
