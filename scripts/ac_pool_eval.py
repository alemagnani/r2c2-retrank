#!/usr/bin/env python3
"""
Stage 0 — Evaluate the AC broad pool quality.

Runs all six intermediate evals (E0.1..E0.6) and prints a single report.
- E0.1: Coverage stats (unique passages / docs per topic)
- E0.2: CE-quality distribution (% topics with strong-CE passages)
- E0.3: Spot-check (writes top-3 from 5 random topics to a file for manual review)
- E0.4: Oracle-doc presence (uses scripts/ac_oracle.py output)
- E0.5: Team contribution distribution
- E0.6: Comparison to retrank-PG-1 (what we lose by dropping retrank)

All evals are zero-cost — no LLM calls. E0.4 requires the oracle JSON to be
pre-built (run scripts/ac_oracle.py first).

Usage:
    python scripts/ac_pool_eval.py --pool data/processed/ac_pool_tier1.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_pool(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Returns (meta, topics_dict)."""
    data = json.loads(path.read_text())
    return data.get("_meta", {}), data["topics"]


def parse_pr_run(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split(";", 3)
        if len(parts) < 4:
            continue
        qid, rank_s, doc_id, text = parts
        if "::" in doc_id:
            doc_id = doc_id.split("::")[0]
        try:
            rank = int(rank_s)
        except ValueError:
            continue
        rows.append({"qid": qid.strip(), "rank": rank, "doc_id": doc_id.strip(),
                     "text": text.strip()})
    return rows


def stats_block(values: list[float], name: str) -> str:
    values = sorted(values)
    n = len(values)
    if n == 0:
        return f"  {name}: (empty)"
    return (f"  {name}: median={values[n//2]:>5.1f}  p25={values[n//4]:>5.1f}  "
            f"p75={values[3*n//4]:>5.1f}  min={values[0]:>5.1f}  max={values[-1]:>5.1f}  "
            f"mean={sum(values)/n:>5.1f}  n={n}")


# ─── E0.1 — Coverage ──────────────────────────────────────────────────────────


def eval_e01(pool: dict[str, list[dict]]) -> dict:
    n_passages = [len(p) for p in pool.values()]
    n_unique_docs = [len({e["doc_id"] for e in p}) for p in pool.values()]

    pass_median = statistics.median(n_unique_docs) >= 25
    pass_min = min(n_unique_docs) >= 15
    below_10 = sum(1 for x in n_unique_docs if x < 10) / len(n_unique_docs)
    pass_below = below_10 < 0.10

    print()
    print("─── E0.1 Coverage ─────────────────────────────────────────────")
    print(stats_block(n_passages, "passages/topic"))
    print(stats_block(n_unique_docs, "unique docs/topic"))
    print(f"  topics with < 10 unique docs: {below_10*100:.1f}%")
    overall = pass_median and pass_min and pass_below
    print(f"  median ≥ 25 unique docs:      {'✓' if pass_median else '✗'}")
    print(f"  min ≥ 15 unique docs:         {'✓' if pass_min else '✗'}")
    print(f"  < 10% topics below 10 docs:   {'✓' if pass_below else '✗'}")
    print(f"  Overall E0.1: {'✓ PASS' if overall else '✗ FAIL'}")
    return {"pass": overall, "below_10_pct": below_10,
            "median_unique_docs": statistics.median(n_unique_docs),
            "min_unique_docs": min(n_unique_docs)}


# ─── E0.2 — CE quality ────────────────────────────────────────────────────────


def eval_e02(pool: dict[str, list[dict]]) -> dict:
    max_ce_per_topic = []
    n_strong_per_topic = []      # CE > 3
    n_very_strong_per_topic = [] # CE > 5
    weak_topics = []

    for qid, passages in pool.items():
        ces = [p["ce_score"] for p in passages if p["ce_score"] is not None]
        if not ces:
            max_ce_per_topic.append(float("-inf"))
            n_strong_per_topic.append(0)
            n_very_strong_per_topic.append(0)
            weak_topics.append((qid, "no CE scores at all"))
            continue
        m = max(ces)
        max_ce_per_topic.append(m)
        n_strong_per_topic.append(sum(1 for c in ces if c > 3))
        n_very_strong_per_topic.append(sum(1 for c in ces if c > 5))
        if m < 1:
            weak_topics.append((qid, f"max CE {m:.2f}"))

    n = len(pool)
    pct_with_strong = sum(1 for c in max_ce_per_topic if c > 3) / n
    pct_with_3_strong = sum(1 for c in n_strong_per_topic if c >= 3) / n

    pass_strong = pct_with_strong >= 0.80
    pass_3_strong = pct_with_3_strong >= 0.60

    print()
    print("─── E0.2 CE quality ───────────────────────────────────────────")
    finite_max = [c for c in max_ce_per_topic if c != float("-inf")]
    print(stats_block(finite_max, "max CE / topic"))
    print(stats_block(n_strong_per_topic, "# CE>3 / topic"))
    print(stats_block(n_very_strong_per_topic, "# CE>5 / topic"))
    print(f"  topics with ≥1 CE>3:           {pct_with_strong*100:.1f}%   "
          f"{'✓' if pass_strong else '✗'} (target ≥80%)")
    print(f"  topics with ≥3 CE>3:           {pct_with_3_strong*100:.1f}%   "
          f"{'✓' if pass_3_strong else '✗'} (target ≥60%)")
    if weak_topics:
        print(f"  Weak-retrieval alarms (max CE < 1): {len(weak_topics)} topics")
        for qid, why in weak_topics[:10]:
            print(f"    - {qid}: {why}")
    overall = pass_strong and pass_3_strong
    print(f"  Overall E0.2: {'✓ PASS' if overall else '✗ FAIL'}")
    return {
        "pass": overall,
        "pct_with_ce3": pct_with_strong,
        "pct_with_3xce3": pct_with_3_strong,
        "weak_topics": [qid for qid, _ in weak_topics],
    }


# ─── E0.3 — Spot check ────────────────────────────────────────────────────────


def eval_e03(pool: dict[str, list[dict]], topics: dict[str, str], n_samples: int,
              output_path: Path) -> dict:
    rng = random.Random(42)
    sample_qids = rng.sample(list(pool.keys()), min(n_samples, len(pool)))

    out_lines = []
    for qid in sample_qids:
        question = topics.get(qid, "(unknown)")
        passages = pool[qid]
        unique_docs = len({p["doc_id"] for p in passages})
        out_lines.append(f"\n{'═'*80}\nTopic {qid}: {question}\n{'═'*80}")
        out_lines.append(f"  Pool size: {len(passages)}, unique docs: {unique_docs}")
        out_lines.append(f"\n  Top-3 passages by CE:")
        for p in passages[:3]:
            out_lines.append(f"    [CE={p['ce_score']:>5.2f}] {p['source_run']}#{p['source_rank']} doc={p['doc_id']}")
            out_lines.append(f"      {p['text'][:300]}{'...' if len(p['text'])>300 else ''}")
        out_lines.append(f"\n  Bottom-3 passages by CE:")
        for p in passages[-3:]:
            out_lines.append(f"    [CE={p['ce_score']:>5.2f}] {p['source_run']}#{p['source_rank']} doc={p['doc_id']}")
            out_lines.append(f"      {p['text'][:200]}{'...' if len(p['text'])>200 else ''}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines))
    print()
    print("─── E0.3 Spot check (manual) ──────────────────────────────────")
    print(f"  {n_samples} random topics' details written to:")
    print(f"    {output_path}")
    print(f"  Manual review required — grade subjectively for answerability.")
    return {"sample_qids": sample_qids, "output_path": str(output_path)}


# ─── E0.4 — Oracle doc presence ───────────────────────────────────────────────


def eval_e04(pool: dict[str, list[dict]], oracle_path: Path) -> dict:
    if not oracle_path.exists():
        print()
        print("─── E0.4 Oracle-doc presence ──────────────────────────────────")
        print(f"  SKIPPED — oracle file not found: {oracle_path}")
        print(f"  Run scripts/ac_oracle.py first.")
        return {"pass": None, "skipped": True}

    oracle = json.loads(oracle_path.read_text())

    def normalise(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower())

    def matches(passage_text: str, entities: list[str]) -> int:
        nt = normalise(passage_text)
        return sum(1 for e in entities if e and normalise(e) in nt)

    found_topics = []
    not_found_topics = []
    for qid, passages in pool.items():
        o = oracle.get(qid)
        if o is None or not o.get("answer"):
            # Oracle declined or missing — count as "n/a"
            continue
        entities = o.get("key_entities", []) or []
        if not entities and o.get("answer"):
            entities = [o["answer"]]
        any_match = any(matches(p["text"], entities) > 0 for p in passages)
        if any_match:
            found_topics.append(qid)
        else:
            not_found_topics.append(qid)

    n_total = len(found_topics) + len(not_found_topics)
    pct_found = len(found_topics) / n_total if n_total else 0.0
    pass_thresh = pct_found >= 0.75

    print()
    print("─── E0.4 Oracle-doc presence ──────────────────────────────────")
    print(f"  Oracle answered for: {n_total}/{len(pool)} topics")
    print(f"  Pool contains oracle entity: {len(found_topics)}/{n_total} ({pct_found*100:.1f}%)")
    print(f"  Pool MISSING oracle entity:  {len(not_found_topics)}/{n_total}")
    if not_found_topics:
        print(f"  Topics where pool may not contain answer:")
        for qid in not_found_topics[:15]:
            o = oracle[qid]
            ans_short = (o.get("answer") or "")[:50]
            print(f"    - {qid}: oracle='{ans_short}' (conf={o.get('confidence')})")
    print(f"  Target ≥ 75%: {'✓ PASS' if pass_thresh else '✗ FAIL'}")
    return {
        "pass": pass_thresh, "pct_found": pct_found,
        "n_total": n_total,
        "missing_topics": not_found_topics,
    }


# ─── E0.5 — Team contribution ─────────────────────────────────────────────────


def eval_e05(pool: dict[str, list[dict]]) -> dict:
    team_counts = Counter()
    total = 0
    for passages in pool.values():
        for p in passages:
            team_counts[p["source_team"]] += 1
            total += 1

    pcts = {t: c / total for t, c in team_counts.items()}
    too_low = [t for t, p in pcts.items() if p < 0.10]
    too_high = [t for t, p in pcts.items() if p > 0.70]

    print()
    print("─── E0.5 Team contribution ────────────────────────────────────")
    for t, p in sorted(pcts.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 50)
        print(f"  {t:<20} {p*100:>5.1f}%  {bar}")
    overall = not too_low and not too_high
    if too_low:
        print(f"  ⚠ Underrepresented (<10%): {too_low}")
    if too_high:
        print(f"  ⚠ Dominating (>70%): {too_high}")
    print(f"  Overall E0.5: {'✓ PASS' if overall else '✗ FAIL'}")
    return {"pass": overall, "team_pct": pcts}


# ─── E0.6 — retrank-loss ──────────────────────────────────────────────────────


def eval_e06(pool: dict[str, list[dict]], pr_runs_dir: Path, ce_cache_path: Path) -> dict:
    import pickle
    retrank_path = pr_runs_dir / "retrank-PG-1.txt"
    if not retrank_path.exists():
        print()
        print("─── E0.6 retrank-loss ──────────────────────────────────────────")
        print(f"  SKIPPED — retrank-PG-1.txt not found at {retrank_path}")
        return {"pass": None, "skipped": True}

    ce_cache = pickle.load(open(ce_cache_path, "rb"))

    pool_doc_set = defaultdict(set)
    for qid, passages in pool.items():
        for p in passages:
            pool_doc_set[qid].add(p["doc_id"])

    retrank_only_count = 0
    retrank_only_high_ce = 0
    for row in parse_pr_run(retrank_path):
        if row["doc_id"] in pool_doc_set.get(row["qid"], set()):
            continue
        retrank_only_count += 1
        ce = ce_cache.get((row["qid"], row["doc_id"], row["text"][:200]))
        if ce is not None and ce > 3:
            retrank_only_high_ce += 1

    pass_thresh = retrank_only_high_ce < 100

    print()
    print("─── E0.6 retrank-loss ─────────────────────────────────────────")
    print(f"  retrank-PG-1 unique docs not in our Tier-1 pool: {retrank_only_count}")
    print(f"  ... of which CE > 3 (real loss):                  {retrank_only_high_ce}")
    print(f"  Target real loss < 100: {'✓ PASS' if pass_thresh else '✗ FAIL'}")
    return {"pass": pass_thresh,
            "retrank_only_count": retrank_only_count,
            "retrank_only_high_ce": retrank_only_high_ce}


# ─── Topic loader ─────────────────────────────────────────────────────────────


def load_topics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        topics = {}
        for m in re.finditer(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", text, re.DOTALL):
            topics[m.group(1).strip()] = m.group(2).strip()
        return topics
    data = json.loads(text)
    return {rec["topic_id"]: rec["question"] for rec in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--oracle", default=str(BASE / "data/eval/ac/oracle_answers.json"))
    parser.add_argument("--pr-runs-dir", default=str(BASE / "data/raw/competitor_runs/PRruns"))
    parser.add_argument("--ce-cache", default=str(BASE / "data/eval/all_team_ce_scores.pkl"))
    parser.add_argument("--n-spotcheck", type=int, default=5)
    parser.add_argument("--report-output", default=None,
                        help="Path to write the eval report JSON (defaults to <pool>.eval.json)")
    args = parser.parse_args()

    pool_path = Path(args.pool)
    meta, pool = load_pool(pool_path)
    topics = load_topics(Path(args.topics))

    print()
    print(f"╔═══════════════════════════════════════════════════════════════════╗")
    print(f"║  Stage 0 Pool Quality Report                                       ║")
    print(f"║  Pool file: {pool_path.name:<54} ║")
    print(f"║  Teams: {str(meta.get('teams', '?')):<58} ║")
    print(f"║  Topics: {len(pool):<57} ║")
    print(f"╚═══════════════════════════════════════════════════════════════════╝")

    spotcheck_path = pool_path.with_suffix(".spotcheck.txt")
    results = {
        "pool_file": str(pool_path),
        "meta": meta,
        "n_topics": len(pool),
        "e01": eval_e01(pool),
        "e02": eval_e02(pool),
        "e03": eval_e03(pool, topics, args.n_spotcheck, spotcheck_path),
        "e04": eval_e04(pool, Path(args.oracle)),
        "e05": eval_e05(pool),
        "e06": eval_e06(pool, Path(args.pr_runs_dir), Path(args.ce_cache)),
    }

    print()
    print(f"╔═══════════════════════════════════════════════════════════════════╗")
    print(f"║  SUMMARY                                                            ║")
    print(f"╠═══════════════════════════════════════════════════════════════════╣")
    for name, key in [("E0.1 Coverage", "e01"), ("E0.2 CE quality", "e02"),
                      ("E0.4 Oracle docs", "e04"), ("E0.5 Team distribution", "e05"),
                      ("E0.6 retrank-loss", "e06")]:
        r = results[key]
        if r.get("skipped"):
            status = "○ SKIPPED"
        elif r.get("pass"):
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"║  {name:<28} {status:<37}║")
    print(f"║  E0.3 Spot-check            ○ MANUAL                              ║")
    print(f"╚═══════════════════════════════════════════════════════════════════╝")

    report_path = Path(args.report_output) if args.report_output else \
        pool_path.with_suffix(".eval.json")
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
