#!/usr/bin/env python3
"""PR re-evaluation reflexive analysis.

Mirrors the official scoring step that re-grades each team's PR run
once AC labels are available: passage relevance grade = #relevant
nuggets that passage contributed across all AC runs.

We use OUR self-evaluator's per-question relevant-nugget labels
(combined across our top candidate AC runs) as a proxy for ground truth.

For each team's PR run, compute nDCG@20 over the per-topic graded relevance.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"
PRRUNS = BASE / "data" / "raw" / "competitor_runs" / "PRruns"


def parse_pr_run(path: Path) -> dict[str, list[tuple[int, str]]]:
    """Returns {topic_id: [(rank, doc_id), ...]} sorted by rank."""
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            qid, rank, docid = parts[0], int(parts[1]), parts[2]
            out[qid].append((rank, docid))
    for qid in out:
        out[qid].sort()
    return dict(out)


def collect_passage_grades(*ac_eval_paths: Path) -> dict[str, dict[tuple[str, int], int]]:
    """{topic_id: {(pr_run, rank): n_relevant_nugget_uses}}.

    Aggregated across the supplied AC runs (relevance label = how many
    times a passage was credited as a 'relevant' nugget for the topic).
    """
    grades: dict[str, dict[tuple[str, int], int]] = defaultdict(lambda: defaultdict(int))
    for p in ac_eval_paths:
        d = json.loads(p.read_text())
        for qid_raw, q in d["per_question"].items():
            qid = qid_raw if qid_raw.startswith("D") else f"D{int(qid_raw):04d}" if qid_raw.isdigit() else qid_raw
            # Topic IDs in PR runs are usually plain like '0001'
            qid_short = qid.lstrip("D").zfill(4) if qid.startswith("D") else qid.zfill(4)
            for n in q.get("nuggets", []):
                if not n.get("entailed"):
                    continue
                # 'relevant' = the AC eval marked it as helpful for derivation
                # We approximate: relevant iff this nugget number appears in
                # the question's nuggets_relevant set (q.n_relevant counts them
                # but doesn't give numbers; fall back to entailed-count if
                # explicit relevance not present).
                # Many eval JSONs don't store per-nugget relevance separately
                # so we use 'entailed' as the AC-credited proxy.
                key = (n["pr_run"], int(n["rank"]))
                grades[qid_short][key] += 1
    return {k: dict(v) for k, v in grades.items()}


def ndcg_at_k(grades: dict[tuple[str, int], int], pr_run_name: str,
              ranking: list[tuple[int, str]], k: int = 20) -> float:
    """nDCG@k for one team's PR run on one topic.
    `grades` is the AC-derived relevance for THIS topic."""
    rels = []
    for rank, _doc in ranking[:k]:
        rels.append(grades.get((pr_run_name, rank), 0))
    if not any(rels):
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    # Use our top AC candidates as the label source — strongest signal
    sources = [
        EVAL / "ensemble_top1.json",
        EVAL / "bitem_only_refined_sonnet_A.json",
        EVAL / "ensemble_top3.json",
    ]
    sources = [p for p in sources if p.exists()]
    print(f"\n  Aggregating relevance grades from {len(sources)} AC eval files:")
    for p in sources:
        print(f"    {p.name}")

    grades_per_topic = collect_passage_grades(*sources)
    print(f"  Topics with at least one graded passage: {len(grades_per_topic)}")

    # Discover all team PR runs
    team_runs = sorted(PRRUNS.iterdir())
    if not team_runs:
        print("  no PR runs found"); return

    print(f"\n{'─'*88}")
    print(f"Reflexive nDCG@20 (passage grade = #relevant-nugget uses across our top AC runs)")
    print(f"{'─'*88}\n")
    print(f"{'PR run':<28} {'mean nDCG@20':>14} {'topics with any grade':>22}")
    print("─" * 88)

    summary = []
    for pr_path in team_runs:
        if not pr_path.is_file():
            continue
        run_name = pr_path.name.replace(".txt", "")
        rankings = parse_pr_run(pr_path)
        scores = []
        n_with_grade = 0
        for qid, ranking in rankings.items():
            qid_short = qid.zfill(4)
            g = grades_per_topic.get(qid_short, {})
            # Filter to only this run's contributions
            g_this_run = {k: v for k, v in g.items() if k[0] == run_name}
            if not g_this_run:
                scores.append(0.0)
                continue
            n_with_grade += 1
            scores.append(ndcg_at_k(g_this_run, run_name, ranking))
        mean_ndcg = sum(scores) / len(scores) if scores else 0.0
        summary.append({"run": run_name, "mean_ndcg": mean_ndcg,
                        "topics_with_grade": n_with_grade,
                        "n_topics": len(scores)})
        print(f"  {run_name:<26} {mean_ndcg:>14.4f}  {n_with_grade:>5}/{len(scores)}")

    summary.sort(key=lambda r: -r["mean_ndcg"])
    print(f"\n  Top 5 by reflexive nDCG@20:")
    for r in summary[:5]:
        print(f"    {r['run']:<26}  {r['mean_ndcg']:.4f}  "
              f"({r['topics_with_grade']}/{r['n_topics']} topics graded)")

    Path(EVAL / "pr_reflexive_ndcg.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\n  saved {EVAL / 'pr_reflexive_ndcg.json'}")


if __name__ == "__main__":
    main()
