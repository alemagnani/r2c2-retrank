#!/usr/bin/env python3
"""HMR sensitivity to the refusal confidence value.

The trivial always-refuse baseline at conf=0 reaches HMR=1.0; at
conf=0.05 it reaches 0.974. How sharp is this pathology? We sweep
the refusal-encoding confidence over a range and report HMR for:
   (a) the trivial always-refuse system,
   (b) our submitted BITEM-Sonnet with its refusals re-encoded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

EVAL = BASE / "data" / "eval" / "ac_runs"


def hmr_for_trivial_refuse(conf: float, n: int = 65) -> float:
    res = [QuestionResult(question_id=f"Q{i}", correct=False,
                          confidence=conf, nuggets_returned=0,
                          nuggets_relevant=0) for i in range(n)]
    return compute_metrics(res).HMR


def hmr_with_refusal_conf_replaced(path: Path, new_conf: float) -> float:
    d = json.loads(path.read_text())
    res = []
    for qid, q in d["per_question"].items():
        ans = (q.get("answer") or "").strip().lower()
        is_refuse = (not ans) or ans in {"i don't know", "i dont know"} \
                    or float(q.get("confidence", 0)) <= 0.10
        c = new_conf if is_refuse else float(q.get("confidence", 0))
        res.append(QuestionResult(
            question_id=qid, correct=bool(q.get("correct")), confidence=c,
            nuggets_returned=int(q.get("n_returned", 0)),
            nuggets_relevant=int(q.get("n_relevant", 0)),
        ))
    return compute_metrics(res).HMR


def main():
    confs = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00]
    print(f"\n{'─'*72}")
    print(f"Refusal-confidence sensitivity sweep")
    print(f"{'─'*72}\n")
    print(f"{'conf':>6} {'always-refuse HMR':>20} {'BITEM-Sonnet HMR':>20}")
    print("─" * 72)
    rows = []
    bitem_path = EVAL / "bitem_only_refined_sonnet_A.json"
    for c in confs:
        a = hmr_for_trivial_refuse(c)
        b = hmr_with_refusal_conf_replaced(bitem_path, c)
        print(f"  {c:>5.2f} {a:>20.4f} {b:>20.4f}")
        rows.append({"conf": c, "trivial_HMR": a, "bitem_HMR": b})

    Path(EVAL / "refusal_conf_sensitivity.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    print(f"\n  saved {EVAL / 'refusal_conf_sensitivity.json'}")


if __name__ == "__main__":
    main()
