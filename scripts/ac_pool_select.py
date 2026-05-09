#!/usr/bin/env python3
"""
Stage 0 — Build the AC broad-pool from selected teams' PR runs.

For each of the 65 R2C2 topics, gather all passages from selected teams' PR runs,
dedup by (doc_id, text-prefix), look up cached CE scores, sort by CE descending,
cap at K, write to JSON.

The output JSON is consumed by Stage 1 (candidate answer) and Stage 1.5 (refinement).

Default selection: Tier-1 = BITEM + Error404 + WaterlooClarke (long passages, low
mutual Jaccard — see README §14). retrank is intentionally excluded because our
200-char passages are too short to be useful nugget source material (see README §11).

Usage:
    python scripts/ac_pool_select.py
    python scripts/ac_pool_select.py --teams BITEM Error404 WaterlooClarke ORG
    python scripts/ac_pool_select.py --top-k 30 --output data/processed/ac_pool_smaller.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

DEFAULT_TIER1 = ["BITEM", "Error404", "WaterlooClarke"]
DEFAULT_TIER1_PLUS_TIER2 = DEFAULT_TIER1 + ["ORG", "hit-u", "WasedaR2C2"]
DEFAULT_BITEM_ONLY = ["BITEM"]


def team_of(filename: str) -> str:
    """e.g. 'Error404-PG-3' -> 'Error404'; 'retrank-PG-1.txt' -> 'retrank'"""
    name = filename.removesuffix(".txt")
    return name.rsplit("-", 2)[0]


def parse_pr_run(path: Path) -> list[dict]:
    """Return list of {qid, rank, doc_id, text} dicts for one PR run file."""
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
        rows.append({
            "qid": qid.strip(),
            "rank": rank,
            "doc_id": doc_id.strip(),
            "text": text.strip(),
        })
    return rows


def normalised_prefix(text: str, n: int = 120) -> str:
    """Used for dedup: case-fold, collapse whitespace, take first n chars."""
    return re.sub(r"\s+", " ", text.lower()).strip()[:n]


def build_pool(
    pr_runs_dir: Path,
    selected_teams: list[str],
    ce_cache: dict,
    top_k: int,
    per_team_cap: int | None,
) -> dict[str, list[dict]]:
    """Return {qid: [pool_entry, ...]} sorted by CE desc, capped at top_k."""
    by_topic: dict[str, list[dict]] = defaultdict(list)

    for run_file in sorted(pr_runs_dir.iterdir()):
        if run_file.is_dir():
            continue
        team = team_of(run_file.name)
        if team not in selected_teams:
            continue
        run_name = run_file.name
        for row in parse_pr_run(run_file):
            qid = row["qid"]
            text = row["text"]
            ce = ce_cache.get((qid, row["doc_id"], text[:200]))
            entry = {
                "passage_key": [run_name, row["rank"]],
                "doc_id": row["doc_id"],
                "text": text,
                "source_team": team,
                "source_run": run_name,
                "source_rank": row["rank"],
                "ce_score": ce if ce is not None else None,
                "_dedup_key": (row["doc_id"], normalised_prefix(text)),
            }
            by_topic[qid].append(entry)

    # Dedup per topic: keep entry with highest CE for each (doc_id, text_prefix)
    deduped: dict[str, list[dict]] = {}
    for qid, entries in by_topic.items():
        best_for_key: dict[tuple, dict] = {}
        for e in entries:
            key = e["_dedup_key"]
            cur = best_for_key.get(key)
            if cur is None:
                best_for_key[key] = e
                continue
            # Prefer the one with a real CE score over None
            if cur["ce_score"] is None and e["ce_score"] is not None:
                best_for_key[key] = e
            elif e["ce_score"] is not None and cur["ce_score"] is not None:
                if e["ce_score"] > cur["ce_score"]:
                    best_for_key[key] = e
        deduped[qid] = list(best_for_key.values())

    # Sort by CE desc (None scores last); apply per-team cap if set; cap at top_k
    result: dict[str, list[dict]] = {}
    for qid, entries in deduped.items():
        entries.sort(key=lambda e: (e["ce_score"] if e["ce_score"] is not None else -1e9), reverse=True)

        if per_team_cap is not None:
            team_count: dict[str, int] = defaultdict(int)
            capped = []
            for e in entries:
                if team_count[e["source_team"]] >= per_team_cap:
                    continue
                capped.append(e)
                team_count[e["source_team"]] += 1
            entries = capped

        # Drop the internal dedup key before serialising
        for e in entries:
            e.pop("_dedup_key", None)
        result[qid] = entries[:top_k]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-runs-dir", default=str(BASE / "data/raw/competitor_runs/PRruns"))
    parser.add_argument("--ce-cache", default=str(BASE / "data/eval/all_team_ce_scores.pkl"))
    parser.add_argument("--teams", nargs="+", default=DEFAULT_TIER1,
                        help="Team names to include (default: Tier-1 = BITEM Error404 WaterlooClarke)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Maximum passages per topic in final pool (default 50)")
    parser.add_argument("--per-team-cap", type=int, default=None,
                        help="Optional cap on passages per team per topic (default: no cap)")
    parser.add_argument("--output", default=str(BASE / "data/processed/ac_pool_tier1.json"))
    parser.add_argument("--preset", choices=["tier1", "tier1plus2", "bitem-only"], default=None,
                        help="Use a named preset for --teams (overrides --teams)")
    args = parser.parse_args()

    if args.preset == "tier1":
        args.teams = DEFAULT_TIER1
    elif args.preset == "tier1plus2":
        args.teams = DEFAULT_TIER1_PLUS_TIER2
    elif args.preset == "bitem-only":
        args.teams = DEFAULT_BITEM_ONLY

    print(f"Selected teams:    {args.teams}")
    print(f"Top-K per topic:   {args.top_k}")
    print(f"Per-team cap:      {args.per_team_cap}")
    print()

    print(f"Loading CE cache from {args.ce_cache} ...")
    ce_cache = pickle.load(open(args.ce_cache, "rb"))
    print(f"  {len(ce_cache):,} cached scores")

    print(f"Reading PR runs from {args.pr_runs_dir} ...")
    pool = build_pool(
        Path(args.pr_runs_dir),
        args.teams,
        ce_cache,
        args.top_k,
        args.per_team_cap,
    )
    print(f"  {len(pool)} topics")

    output = {
        "_meta": {
            "teams": args.teams,
            "top_k": args.top_k,
            "per_team_cap": args.per_team_cap,
            "n_topics": len(pool),
            "ce_cache_size": len(ce_cache),
        },
        "topics": pool,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    print(f"\nWrote pool to {output_path}")

    sizes = [len(v) for v in pool.values()]
    sizes.sort()
    print(f"\nPool size distribution:")
    print(f"  min={sizes[0]}  p25={sizes[len(sizes)//4]}  median={sizes[len(sizes)//2]}  "
          f"p75={sizes[3*len(sizes)//4]}  max={sizes[-1]}")
    print(f"  mean={sum(sizes)/len(sizes):.1f}")


if __name__ == "__main__":
    main()
