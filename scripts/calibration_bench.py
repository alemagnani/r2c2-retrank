#!/usr/bin/env python3
"""
Calibration benchmark — compare confidence strategies by HMR without re-running the LLM judge.

Two input modes:

  (1) From a JSON of pre-computed per-question outcomes:
      [
        {"qid": "0001", "correct": true,  "confidence": 0.92, "returned": 4, "relevant": 3},
        ...
      ]

  (2) From multiple AC eval JSON outputs (produced by scripts/ac_eval.py),
      so you can compare 4+ strategies side-by-side on the same topics.

Output: a leaderboard table sorted by HMR, plus per-strategy R_O/R_U/Accuracy/MNP.

Examples:
  python scripts/calibration_bench.py --inputs strategy_*.json
  python scripts/calibration_bench.py --raw labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402


def _load_eval_json(path: Path) -> list[QuestionResult]:
    """Parse the per_question section of a scripts/ac_eval.py output JSON."""
    data = json.loads(path.read_text())
    if "per_question" not in data:
        raise ValueError(f"{path}: not an ac_eval.py output (missing per_question)")
    out: list[QuestionResult] = []
    for qid, q in data["per_question"].items():
        out.append(QuestionResult(
            question_id=qid,
            correct=bool(q["correct"]),
            confidence=float(q["confidence"]),
            nuggets_returned=int(q["n_returned"]),
            nuggets_relevant=int(q["n_relevant"]),
        ))
    return out


def _load_raw_json(path: Path) -> list[QuestionResult]:
    raw = json.loads(path.read_text())
    out: list[QuestionResult] = []
    for r in raw:
        conf = float(r["confidence"])
        if conf > 1:
            conf = conf / 100.0
        out.append(QuestionResult(
            question_id=str(r["qid"]),
            correct=bool(r["correct"]),
            confidence=conf,
            nuggets_returned=int(r.get("returned", 0)),
            nuggets_relevant=int(r.get("relevant", 0)),
        ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", help="ac_eval.py JSON output files (one per strategy)")
    parser.add_argument("--raw", help="JSON list of {qid, correct, confidence, returned, relevant}")
    parser.add_argument("--names", nargs="+", help="Optional display names for each --inputs file")
    args = parser.parse_args()

    bench: list[tuple[str, list[QuestionResult]]] = []

    if args.raw:
        results = _load_raw_json(Path(args.raw))
        bench.append((Path(args.raw).stem, results))

    if args.inputs:
        names = args.names or [Path(p).stem for p in args.inputs]
        if len(names) != len(args.inputs):
            parser.error("--names length must match --inputs length")
        for name, p in zip(names, args.inputs):
            bench.append((name, _load_eval_json(Path(p))))

    if not bench:
        parser.error("provide --inputs or --raw")

    rows = []
    for name, results in bench:
        m = compute_metrics(results)
        rows.append((name, m))

    rows.sort(key=lambda x: x[1].HMR, reverse=True)

    print()
    print(f"{'Strategy':<32} {'N':>4}  {'Acc':>6}  {'MNP':>6}  {'R_O':>6}  {'R_U':>6}  {'HMR':>6}")
    print("─" * 80)
    for name, m in rows:
        print(f"{name:<32} {m.n_questions:>4}  "
              f"{m.accuracy:>6.3f}  {m.mean_nugget_precision:>6.3f}  "
              f"{m.R_O:>6.3f}  {m.R_U:>6.3f}  {m.HMR:>6.3f}")
    print()


if __name__ == "__main__":
    main()
