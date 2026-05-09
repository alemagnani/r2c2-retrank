#!/usr/bin/env python3
"""
AutoResearch: LLM-driven diagnosis and retrieval improvement loop.

Pipeline:
  1. EVALUATE  — load nDCG@20 scores per topic from existing qrels
  2. DIAGNOSE  — find failing/weak topics, fetch their top retrieved passages
  3. HYPOTHESIZE — Claude explains WHY retrieval failed and proposes a fix
  4. IMPLEMENT — auto-generate an improved run using the proposed strategy
  5. EVALUATE  — score the new run and compare delta nDCG

Failure taxonomy (Claude classifies each failing topic into one):
  - implicit_entity   : question references entity indirectly ("the Bond film in space")
  - vocabulary_gap    : question uses different words than corpus ("torturing device" vs "ventilator")
  - multi_hop         : answer requires chaining two facts
  - counting          : question asks how many X, requires aggregation
  - comparison        : question compares two entities (which is better/earlier/higher)
  - out_of_corpus     : answer likely not in Wikipedia/Wookieepedia
  - other             : catch-all

Usage:
    python scripts/autoresearch.py \
        --topics data/raw/r2c2topics.txt \
        --runs data/runs/alemagnani-PO-1.txt data/runs/alemagnani-PO-2.txt \
        --qrels data/eval/qrels_haiku.json \
        --base-run data/runs/alemagnani-PO-1.txt \
        --output-dir data/autoresearch/
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

# ── Failure taxonomy ──────────────────────────────────────────────────────────

FAILURE_TYPES = [
    "implicit_entity",   # indirect reference to a named entity
    "vocabulary_gap",    # different terminology in question vs corpus
    "multi_hop",         # requires chaining two or more facts
    "counting",          # requires aggregation/counting
    "comparison",        # requires comparing two entities
    "out_of_corpus",     # answer likely not in Wikipedia/Wookieepedia
    "other",
]

DIAGNOSE_PROMPT = """\
You are an information retrieval expert. A retrieval system FAILED to find relevant \
passages for the following question about movies or Star Wars.

Question: {question}

The system returned these top passages (none were relevant):
{passages}

Tasks:
1. Classify the failure into ONE of these types:
   {failure_types}

2. Explain in 1-2 sentences WHY retrieval failed.

3. Suggest a concrete query reformulation that would work better \
(rewrite the question to use more specific or corpus-friendly terms, \
resolve implicit entity references, etc.)

Reply in JSON with exactly these fields:
{{
  "failure_type": "<one of the types above>",
  "reason": "<1-2 sentence explanation>",
  "reformulated_query": "<improved query string>"
}}"""

PARTIAL_SUCCESS_PROMPT = """\
You are an information retrieval expert. A retrieval system retrieved SOME relevant \
passages for the following question, but ranked them poorly (low nDCG@20).

Question: {question}

Top-5 passages retrieved (mix of relevant and irrelevant):
{passages}

Tasks:
1. Classify the main issue:
   {failure_types}

2. Explain in 1-2 sentences what's causing poor ranking.

3. Suggest a reformulated query that would better surface the relevant passages.

Reply in JSON with exactly these fields:
{{
  "failure_type": "<one of the types above>",
  "reason": "<1-2 sentence explanation>",
  "reformulated_query": "<improved query string>"
}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_topics(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
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


def load_run(path: Path) -> dict[str, list[tuple[int, str, str]]]:
    """Returns {topic_id: [(rank, doc_id, text), ...]} sorted by rank."""
    run = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", 3)
            if len(parts) == 4:
                topic_id, rank, doc_id, text = parts
                run[topic_id].append((int(rank), doc_id, text))
    return {tid: sorted(v) for tid, v in run.items()}


def load_qrels(path: Path) -> dict[str, dict[tuple[str, str], int]]:
    with path.open() as f:
        raw = json.load(f)
    return {
        tid: {tuple(k.split("\x00", 1)): v for k, v in passages.items()}
        for tid, passages in raw.items()
    }


def dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(ranked_rels: list[int], k: int = 20) -> float:
    ideal = sorted(ranked_rels, reverse=True)
    ideal_dcg = dcg(ideal, k)
    return dcg(ranked_rels, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def score_run(run: dict, qrels: dict, k: int = 20) -> dict[str, float]:
    scores = {}
    for topic_id, passages in run.items():
        rels = [qrels.get(topic_id, {}).get((doc_id, text), 0)
                for _, doc_id, text in passages]
        scores[topic_id] = ndcg_at_k(rels, k)
    scores["mean"] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores


def call_claude(client, model: str, prompt: str, min_interval: float = 1.3) -> str:
    for attempt in range(4):
        start = time.monotonic()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return response.content[0].text.strip()
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            if attempt < 3:
                print(f"  API error, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def parse_json_response(text: str) -> dict:
    """Extract JSON from Claude's response, tolerating markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract fields with regex
        ft = re.search(r'"failure_type"\s*:\s*"([^"]+)"', text)
        reason = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
        rq = re.search(r'"reformulated_query"\s*:\s*"([^"]+)"', text)
        return {
            "failure_type": ft.group(1) if ft else "other",
            "reason": reason.group(1) if reason else "",
            "reformulated_query": rq.group(1) if rq else "",
        }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def diagnose_topics(client, model: str, topics: dict, base_run: dict,
                    qrels: dict, ndcg_scores: dict,
                    fail_threshold: float = 0.1,
                    partial_threshold: float = 0.4) -> dict[str, dict]:
    """
    For each weak topic, ask Claude to diagnose and suggest a reformulated query.
    Returns {topic_id: {failure_type, reason, reformulated_query, ndcg}}.
    """
    failures = {
        tid: score for tid, score in ndcg_scores.items()
        if tid != "mean" and score < partial_threshold
    }
    print(f"\nDiagnosing {len(failures)} weak topics "
          f"(nDCG < {partial_threshold}) out of {len(ndcg_scores)-1} total")

    failure_types_str = "\n   ".join(f"- {ft}" for ft in FAILURE_TYPES)
    diagnoses = {}

    for topic_id in sorted(failures.keys(), key=lambda t: failures[t]):
        question = topics.get(topic_id, "")
        ndcg = failures[topic_id]
        passages = base_run.get(topic_id, [])[:5]
        passage_text = "\n".join(
            f"  [{i+1}] (rel={qrels.get(topic_id, {}).get((doc_id, text), 0)}) {text}"
            for i, (_, doc_id, text) in enumerate(passages)
        )

        is_total_fail = ndcg < fail_threshold
        prompt_template = DIAGNOSE_PROMPT if is_total_fail else PARTIAL_SUCCESS_PROMPT
        prompt = prompt_template.format(
            question=question,
            passages=passage_text,
            failure_types=failure_types_str,
        )

        print(f"\n[{topic_id}] nDCG={ndcg:.3f}  Q: {question[:70]}")
        try:
            raw = call_claude(client, model, prompt)
            diag = parse_json_response(raw)
            diag["ndcg"] = ndcg
            diag["question"] = question
            diagnoses[topic_id] = diag
            print(f"  type={diag['failure_type']}  → {diag['reformulated_query'][:80]}")
        except Exception as e:
            print(f"  Diagnosis failed: {e}")
            diagnoses[topic_id] = {
                "failure_type": "other",
                "reason": str(e),
                "reformulated_query": question,
                "ndcg": ndcg,
                "question": question,
            }

    return diagnoses


def generate_improved_run(diagnoses: dict, base_run: dict, bm25_index_dir: str,
                          meta_file: str, output_path: Path, top_k: int = 20):
    """Use reformulated queries to re-retrieve for failing topics, keep base run for others."""
    import pickle
    import bm25s

    print(f"\nLoading BM25S index for re-retrieval ...")
    retriever = bm25s.BM25.load(bm25_index_dir, load_corpus=False)
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    improved = 0

    with output_path.open("w", encoding="utf-8") as fout:
        # For diagnosed topics: use reformulated query
        for topic_id, diag in diagnoses.items():
            ref_query = diag.get("reformulated_query", diag["question"])
            if not ref_query or ref_query == diag["question"]:
                # No improvement suggested — keep original
                for rank, doc_id, text in base_run.get(topic_id, [])[:top_k]:
                    fout.write(f"{topic_id};{rank};{doc_id};{text}\n")
                continue

            tokenized = bm25s.tokenize([ref_query], stopwords="en", stemmer=None)
            results, _ = retriever.retrieve(tokenized, k=top_k)
            for rank, idx in enumerate(results[0], start=1):
                rec = meta[idx]
                fout.write(f"{topic_id};{rank};{rec['doc_id']};{rec['text_snippet']}\n")
            improved += 1

        # For non-diagnosed topics: copy from base run unchanged
        diagnosed_ids = set(diagnoses.keys())
        for topic_id, passages in base_run.items():
            if topic_id not in diagnosed_ids:
                for rank, doc_id, text in passages[:top_k]:
                    fout.write(f"{topic_id};{rank};{doc_id};{text}\n")

    print(f"Improved run: {improved} topics re-retrieved, "
          f"{len(base_run) - improved} kept from base run")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--runs", nargs="+",
                        default=[str(BASE / "data/runs/alemagnani-PO-1.txt")])
    parser.add_argument("--base-run", default=str(BASE / "data/runs/alemagnani-PO-1.txt"),
                        help="Run used for diagnosis context and as fallback")
    parser.add_argument("--qrels", default=str(BASE / "data/eval/qrels_haiku.json"))
    parser.add_argument("--output-dir", default=str(BASE / "data/autoresearch"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--fail-threshold", type=float, default=0.1,
                        help="Topics below this nDCG are 'total failures'")
    parser.add_argument("--partial-threshold", type=float, default=0.4,
                        help="Topics below this nDCG are diagnosed")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Only diagnose, don't generate improved run")
    parser.add_argument("--bm25-index", default=str(BASE / "data/processed/bm25s_index"))
    parser.add_argument("--meta", default=str(BASE / "data/processed/passages_meta.pkl"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topics = load_topics(Path(args.topics))
    qrels = load_qrels(Path(args.qrels))
    base_run = load_run(Path(args.base_run))

    # Score the base run to find weak topics
    print(f"Scoring base run: {args.base_run}")
    ndcg_scores = score_run(base_run, qrels)
    print(f"Base run mean nDCG@20: {ndcg_scores['mean']:.4f}")

    # Also print scores for all provided runs
    print("\nAll run scores:")
    for run_path in args.runs:
        run = load_run(Path(run_path))
        scores = score_run(run, qrels)
        print(f"  {Path(run_path).stem:35s}  nDCG@20 = {scores['mean']:.4f}")

    client = anthropic.Anthropic()

    # Diagnose weak topics
    diagnoses = diagnose_topics(
        client, args.model, topics, base_run, qrels, ndcg_scores,
        fail_threshold=args.fail_threshold,
        partial_threshold=args.partial_threshold,
    )

    # Save diagnoses
    diag_path = output_dir / "diagnoses.json"
    with diag_path.open("w") as f:
        json.dump(diagnoses, f, indent=2, ensure_ascii=False)
    print(f"\nDiagnoses saved to {diag_path}")

    # Print taxonomy summary
    from collections import Counter
    counts = Counter(d["failure_type"] for d in diagnoses.values())
    print("\nFailure taxonomy:")
    for ft, count in counts.most_common():
        print(f"  {ft:20s}  {count}")

    if args.diagnose_only:
        return

    # Generate improved run
    improved_run_path = output_dir / "autoresearch_run.txt"
    generate_improved_run(
        diagnoses, base_run, args.bm25_index, args.meta,
        improved_run_path, top_k=20,
    )

    # Score improved run
    improved_run = load_run(improved_run_path)
    improved_scores = score_run(improved_run, qrels)
    print(f"\nResults:")
    print(f"  Base run nDCG@20:     {ndcg_scores['mean']:.4f}")
    print(f"  Improved run nDCG@20: {improved_scores['mean']:.4f}")
    delta = improved_scores['mean'] - ndcg_scores['mean']
    print(f"  Delta:                {delta:+.4f}")

    # Per-topic delta for diagnosed topics
    print(f"\nPer-topic delta for {len(diagnoses)} diagnosed topics:")
    for tid in sorted(diagnoses.keys()):
        before = ndcg_scores.get(tid, 0)
        after = improved_scores.get(tid, 0)
        q = diagnoses[tid]["question"][:50]
        ft = diagnoses[tid]["failure_type"]
        print(f"  [{tid}] {before:.3f} → {after:.3f} ({after-before:+.3f})  [{ft}]  {q}")

    # Save summary
    summary = {
        "base_run": args.base_run,
        "base_ndcg": ndcg_scores["mean"],
        "improved_ndcg": improved_scores["mean"],
        "delta": delta,
        "n_diagnosed": len(diagnoses),
        "taxonomy": dict(counts),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
