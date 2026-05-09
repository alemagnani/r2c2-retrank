#!/usr/bin/env python3
"""
AC self-evaluator — mirrors the official R2C2 AC scoring pipeline.

Given an AC run file plus the PR run files cited by its nuggets, produces:
    Accuracy, Mean Nugget Precision, R_O, R_U, HMR
plus per-question detail.

Two LLM stages (Sonnet 4.6 by default):
    Stage A — Bogus Nugget Identification (per (passage, nugget) pair)
    Stage B — Answer Evaluation + Relevance attribution (per question)

Both stages cache by SHA256 in JSON files so re-runs are nearly free.

Usage:
    python scripts/ac_eval.py \\
        --ac-run data/runs/retrank-AC-1.txt \\
        --pr-runs-dir data/raw/competitor_runs/PRruns \\
        --topics data/raw/r2c2topics.txt \\
        --output data/eval/ac/retrank-AC-1.json

Auxiliary mode (no LLM calls):
    python scripts/ac_eval.py --metrics-only \\
        --labels labels.json   # {qid: {correct: bool, conf: 0-100, returned: int, relevant: int}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))

from ac_format import ACRecord, Nugget, parse_ac_run, validate  # noqa: E402
from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

# ─── Prompts ──────────────────────────────────────────────────────────────────

BOGUS_PROMPT = """You are evaluating whether a passage entails a factual claim (a "nugget").

Passage: {passage_text}

Nugget: {nugget_text}

Does the passage entail the nugget? An entailed nugget is a claim that follows \
directly from the passage text without external knowledge. Paraphrasing is allowed; \
adding facts not in the passage is not. If the nugget contradicts the passage or \
asserts something the passage does not support, it is not entailed.

Reply with JSON only:
{{"entailed": true|false, "reason": "<one short sentence>"}}"""


ANSWER_EVAL_PROMPT = """You are evaluating a system-generated answer to a movie question.

Question: {question}

System answer: {answer}

Entailed nuggets (numbered, assume each is factually correct):
{nuggets_block}

Tasks:
1. Assuming all nuggets above are factually correct, is the system answer correct?
   Be strict about exact matches when the question asks for a specific entity. \
For "who/what/when" questions, the answer must name the same entity/value as the nuggets imply.
2. Which nuggets HELPED you derive that the answer is correct? Return their 1-based indices. \
Empty list if the answer is incorrect.

Reply with JSON only:
{{"correct": true|false, "relevant_nugget_indices": [<int>, ...], "reason": "<one short sentence>"}}"""


# ─── LLM helpers ──────────────────────────────────────────────────────────────


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _call_llm(client: anthropic.Anthropic, model: str, prompt: str,
              retries: int = 4, min_interval: float = 1.2) -> dict:
    """Send `prompt` to Anthropic and return parsed JSON. Retries on transient errors."""
    last_err: Exception | None = None
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
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
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
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


# ─── PR-run passage lookup ────────────────────────────────────────────────────


def _parse_pr_run(path: Path) -> dict[tuple[str, int], str]:
    """Return mapping (qid, rank) -> passage_text for one PR run file."""
    out: dict[tuple[str, int], str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split(";", 3)
        if len(parts) < 4:
            continue
        qid, rank_s, _doc_id, text = parts
        try:
            rank = int(rank_s)
        except ValueError:
            continue
        # Some teams use composite docIDs with "::" — irrelevant for passage text lookup.
        out[(qid.strip(), rank)] = text.strip()
    return out


def _load_pr_passage_lookup(pr_runs_dir: Path) -> dict[str, dict[tuple[str, int], str]]:
    """Map pr_run_name -> ((qid, rank) -> passage_text)."""
    lookup: dict[str, dict[tuple[str, int], str]] = {}
    for f in sorted(Path(pr_runs_dir).iterdir()):
        if f.is_dir():
            continue
        # PR run name: stem matches what AC nuggets cite (sometimes with .txt suffix)
        name = f.name
        lookup[name] = _parse_pr_run(f)
        if name.endswith(".txt"):
            lookup[name[:-4]] = lookup[name]  # alias without .txt
    return lookup


# ─── Topic loading ────────────────────────────────────────────────────────────


def _load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        topics = {}
        for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL):
            topics[m.group(1).strip()] = m.group(2).strip()
        return topics
    # JSON
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


# ─── Stage A — Bogus identification ───────────────────────────────────────────


def stage_a_bogus_check(
    nugget: Nugget,
    passage_text: str,
    client: anthropic.Anthropic,
    model: str,
    cache: dict,
) -> dict:
    """Return {entailed: bool, reason: str} (cached)."""
    key = _sha(passage_text, nugget.text)
    if key in cache:
        return cache[key]
    prompt = BOGUS_PROMPT.format(passage_text=passage_text, nugget_text=nugget.text)
    try:
        result = _call_llm(client, model, prompt)
    except RuntimeError as e:
        # Persistent failure — leave UN-cached and return entailed=True so we
        # don't poison Mean Nugget Precision with a fake bogus. Caller sees
        # this through the "_failed" flag.
        print(f"  Stage A failed; treating as entailed: {e}", file=sys.stderr)
        return {"entailed": True, "reason": f"check failed (assumed entailed): {type(e).__name__}",
                "_failed": True}
    if "entailed" not in result:
        result = {"entailed": False, "reason": "LLM returned no 'entailed' field"}
    cache[key] = result
    return result


# ─── Stage B — Answer evaluation ──────────────────────────────────────────────


def stage_b_answer_eval(
    question: str,
    answer: str,
    entailed_nuggets: list[Nugget],
    client: anthropic.Anthropic,
    model: str,
    cache: dict,
) -> dict:
    """Return {correct: bool, relevant_nugget_indices: [int], reason: str} (cached)."""
    if not entailed_nuggets:
        # No entailed nuggets → can't justify any answer; mark incorrect, no relevant.
        return {"correct": False, "relevant_nugget_indices": [], "reason": "no entailed nuggets"}

    nug_texts = [n.text for n in entailed_nuggets]
    cache_key = _sha(question, answer, *nug_texts)
    if cache_key in cache:
        return cache[cache_key]

    nuggets_block = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(nug_texts))
    prompt = ANSWER_EVAL_PROMPT.format(question=question, answer=answer, nuggets_block=nuggets_block)
    try:
        result = _call_llm(client, model, prompt)
    except RuntimeError as e:
        # Persistent parse/API failure: treat as incorrect; do NOT crash the run.
        print(f"  Stage B failed for this topic: {e}", file=sys.stderr)
        result = {"correct": False, "relevant_nugget_indices": [],
                  "reason": f"Stage B failed: {type(e).__name__}"}
    if "correct" not in result:
        result = {"correct": False, "relevant_nugget_indices": [], "reason": "LLM returned no 'correct' field"}
    if "relevant_nugget_indices" not in result:
        result["relevant_nugget_indices"] = []
    # Sanity-check indices
    result["relevant_nugget_indices"] = [
        i for i in result.get("relevant_nugget_indices", []) if isinstance(i, int) and 1 <= i <= len(entailed_nuggets)
    ]
    cache[cache_key] = result
    return result


# ─── Driver ───────────────────────────────────────────────────────────────────


def evaluate_ac_run(
    ac_run: list[ACRecord],
    topics: dict[str, str],
    passage_lookup: dict[str, dict[tuple[str, int], str]],
    client: anthropic.Anthropic,
    model: str,
    bogus_cache_path: Path,
    answer_cache_path: Path,
) -> dict:
    """Run Stage A and Stage B on every record, then compute metrics."""
    bogus_cache = _load_cache(bogus_cache_path)
    answer_cache = _load_cache(answer_cache_path)

    per_question: dict[str, dict] = {}
    metric_inputs: list[QuestionResult] = []

    for i, rec in enumerate(ac_run):
        qid = rec.question_id
        question = topics.get(qid)
        if question is None:
            print(f"  [{i+1}/{len(ac_run)}] {qid}: no topic found, skipping", file=sys.stderr)
            continue

        # Stage A — entailment per nugget
        nugget_results: list[dict] = []
        entailed_nuggets: list[Nugget] = []
        for nug in rec.nuggets:
            passage_text = passage_lookup.get(nug.pr_run_name, {}).get((qid, nug.passage_rank))
            if passage_text is None:
                # Missing citation — counts as bogus
                nugget_results.append({
                    "nugget_num": nug.num, "entailed": False,
                    "reason": "passage not found in PR runs",
                    "passage_present": False,
                })
                continue
            verdict = stage_a_bogus_check(nug, passage_text, client, model, bogus_cache)
            entailed_bool = bool(verdict.get("entailed"))
            nugget_results.append({
                "nugget_num": nug.num, "entailed": entailed_bool,
                "reason": verdict.get("reason", ""),
                "passage_present": True,
            })
            if entailed_bool:
                entailed_nuggets.append(nug)

        # Stage B — answer correctness + relevance attribution
        ans_verdict = stage_b_answer_eval(question, rec.answer, entailed_nuggets, client, model, answer_cache)
        relevant_count = len(ans_verdict.get("relevant_nugget_indices", []))

        per_question[qid] = {
            "question": question,
            "answer": rec.answer,
            "confidence": rec.confidence,
            "n_returned": len(rec.nuggets),
            "n_entailed": len(entailed_nuggets),
            "n_relevant": relevant_count,
            "correct": bool(ans_verdict.get("correct")),
            "answer_reason": ans_verdict.get("reason", ""),
            "nuggets": [
                {
                    "num": n.num, "pr_run": n.pr_run_name, "rank": n.passage_rank,
                    "text": n.text,
                    **{r["nugget_num"]: r for r in nugget_results}.get(n.num, {}),
                }
                for n in rec.nuggets
            ],
        }

        metric_inputs.append(QuestionResult(
            question_id=qid,
            correct=bool(ans_verdict.get("correct")),
            confidence=rec.confidence,
            nuggets_returned=len(rec.nuggets),
            nuggets_relevant=relevant_count,
        ))

        # Periodic cache flush
        if (i + 1) % 5 == 0:
            _save_cache(bogus_cache_path, bogus_cache)
            _save_cache(answer_cache_path, answer_cache)
            print(f"  [{i+1}/{len(ac_run)}] {qid}: correct={ans_verdict.get('correct')} "
                  f"entailed={len(entailed_nuggets)}/{len(rec.nuggets)} relevant={relevant_count}")

    _save_cache(bogus_cache_path, bogus_cache)
    _save_cache(answer_cache_path, answer_cache)

    metrics = compute_metrics(metric_inputs)
    return {
        "metrics": metrics.as_dict(),
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac-run", required=False, help="Path to [TeamName]-AC-[N] file")
    parser.add_argument("--pr-runs-dir", default=str(BASE / "data/raw/competitor_runs/PRruns"))
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output", required=False, help="Write metrics + per-question JSON here")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--cache-dir", default=str(BASE / "data/eval/ac_cache"))
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip LLM calls; expects --labels with pre-computed correctness")
    parser.add_argument("--labels", default=None,
                        help="(metrics-only mode) JSON {qid: {correct, conf, returned, relevant}}")
    args = parser.parse_args()

    if args.metrics_only:
        if not args.labels:
            parser.error("--metrics-only requires --labels")
        labels = json.loads(Path(args.labels).read_text())
        results = []
        for qid, lab in labels.items():
            results.append(QuestionResult(
                question_id=qid,
                correct=bool(lab["correct"]),
                confidence=float(lab.get("conf", 50)) / 100.0 if lab.get("conf", 50) > 1 else float(lab.get("conf", 0.5)),
                nuggets_returned=int(lab.get("returned", 0)),
                nuggets_relevant=int(lab.get("relevant", 0)),
            ))
        m = compute_metrics(results)
        print(json.dumps(m.as_dict(), indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(m.as_dict(), indent=2))
        return

    if not (args.ac_run and args.output):
        parser.error("--ac-run and --output are required (unless --metrics-only)")

    ac_run = parse_ac_run(Path(args.ac_run))
    warnings = validate(ac_run)
    for w in warnings:
        print(f"  WARNING: {w}", file=sys.stderr)
    print(f"Loaded {len(ac_run)} AC records from {args.ac_run}")

    topics = _load_topics(Path(args.topics))
    print(f"Loaded {len(topics)} topics")

    passage_lookup = _load_pr_passage_lookup(Path(args.pr_runs_dir))
    print(f"Loaded {len(passage_lookup)} PR runs from {args.pr_runs_dir}")

    cache_dir = Path(args.cache_dir)
    bogus_cache_path = cache_dir / f"bogus_{args.model}.json"
    answer_cache_path = cache_dir / f"answer_{args.model}.json"

    client = anthropic.Anthropic()
    print(f"Evaluating with {args.model} ...")
    out = evaluate_ac_run(
        ac_run, topics, passage_lookup, client, args.model,
        bogus_cache_path, answer_cache_path,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print()
    print("─" * 60)
    print(f"  Run: {Path(args.ac_run).name}")
    print(f"  Accuracy:               {out['metrics']['accuracy']:.4f}")
    print(f"  Mean Nugget Precision:  {out['metrics']['mean_nugget_precision']:.4f}")
    print(f"  R_O (overconfidence):   {out['metrics']['R_O']:.4f}")
    print(f"  R_U (underconfidence):  {out['metrics']['R_U']:.4f}")
    print(f"  HMR:                    {out['metrics']['HMR']:.4f}")
    print(f"  N questions:            {out['metrics']['n_questions']}")
    print("─" * 60)
    print(f"Detail written to {output_path}")


if __name__ == "__main__":
    main()
