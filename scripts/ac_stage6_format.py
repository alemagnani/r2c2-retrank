#!/usr/bin/env python3
"""
Stage 6 — Format calibrated outputs into the official AC submission file.

Reads a Stage 5 JSON (per-topic answer + confidence + filtered_nuggets) and
writes the AC XML format expected by NTCIR-19 R2C2:

    <D001>[Answer];[Confidence]
    [NuggetNum];[PRrunname];[PassageRank];[NuggetText]
    ...
    </D001>

Reuses scripts/ac_format.py for the actual XML serialisation.

Usage:
    python scripts/ac_stage6_format.py \\
        --stage5 data/processed/ac_stage5_tier1_broad_A.json \\
        --output data/runs/retrank-AC-1.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from ac_format import ACRecord, Nugget, validate, write_ac_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5", required=True)
    parser.add_argument("--output", required=True, help="AC run file path (e.g. retrank-AC-1.txt)")
    args = parser.parse_args()

    stage5 = json.loads(Path(args.stage5).read_text())
    records: list[ACRecord] = []

    for qid in sorted(stage5["topics"].keys()):
        rec = stage5["topics"][qid]
        answer = (rec.get("answer") or "").strip()
        confidence = int(rec.get("confidence") or 0)
        confidence = max(0, min(100, confidence))

        nuggets: list[Nugget] = []
        for i, n in enumerate(rec.get("filtered_nuggets", []), 1):
            pr_run, rank = n["passage_key"]
            # Strip any .txt suffix for the citation; competitor docs use the
            # bare run name.
            pr_run_clean = pr_run.rstrip()
            if pr_run_clean.endswith(".txt"):
                pr_run_clean = pr_run_clean[:-4]
            nuggets.append(Nugget(
                num=i,
                pr_run_name=pr_run_clean,
                passage_rank=int(rank),
                text=(n["text"] or "").strip(),
            ))

        records.append(ACRecord(
            question_id=qid.zfill(4),
            answer=answer,
            confidence_raw=confidence,
            nuggets=nuggets,
        ))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ac_run(output_path, records)

    warnings = validate(records)
    print(f"Wrote AC run to {output_path}")
    print(f"  Records:     {len(records)}")
    print(f"  Total nuggets: {sum(len(r.nuggets) for r in records)}")
    n_refused = sum(1 for r in records if not r.answer)
    print(f"  Refusals:    {n_refused}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"    - {w}")


if __name__ == "__main__":
    main()
