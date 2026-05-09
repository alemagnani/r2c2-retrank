#!/usr/bin/env python3
"""Oracle upper bounds for HMR / Acc.

Three oracles, each cumulative:
  (1) Oracle abstention: refuse exactly the topics our pipeline gets wrong.
      Conf=0.05 on those, leave correct topics' conf unchanged.
      Tells us the headroom available from perfect refusal alone.
  (2) Oracle calibration: leave correctness pattern unchanged, but set
      conf=1.0 on every correct topic and conf=0.0 on every wrong one.
      Tells us the headroom available from perfect calibration alone.
  (3) Oracle answer: every topic correct (conf=0.95).
      Sanity check — should give R_O=1, R_U=0.95, HMR≈0.974.

Reported relative to BITEM-Sonnet (main system).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

EVAL = BASE / "data" / "eval" / "ac_runs"


def load(p: Path) -> list[QuestionResult]:
    d = json.loads(p.read_text())
    out = []
    for qid, q in d["per_question"].items():
        out.append(QuestionResult(
            question_id=qid,
            correct=bool(q.get("correct")),
            confidence=float(q.get("confidence", 0)),
            nuggets_returned=int(q.get("n_returned", 0)),
            nuggets_relevant=int(q.get("n_relevant", 0)),
        ))
    return out


def report(label: str, res: list[QuestionResult]):
    m = compute_metrics(res)
    n_cnf_w = sum(1 for r in res if not r.correct and r.confidence >= 0.5)
    n_refuse = sum(1 for r in res if r.confidence <= 0.10)
    print(f"  {label:<55} Acc={m.accuracy:.3f} R_O={m.R_O:.3f} R_U={m.R_U:.3f} "
          f"HMR={m.HMR:.3f} Cnf-W={n_cnf_w} Refuse={n_refuse}")


def main():
    src = EVAL / "bitem_only_refined_sonnet_A.json"
    base = load(src)

    print(f"\n{'─'*108}")
    print(f"Oracle upper bounds, anchored to BITEM-Sonnet (main system)")
    print(f"{'─'*108}\n")

    report("Baseline: BITEM-Sonnet as submitted", base)

    # Oracle 1: refuse exactly the wrong topics; correct keep their confidence
    oracle_abst = [
        QuestionResult(question_id=r.question_id,
                       correct=False if not r.correct else r.correct,
                       confidence=0.05 if not r.correct else r.confidence,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    report("Oracle abstention: refuse exactly the topics we get wrong", oracle_abst)

    # Oracle 2: perfect calibration on existing correctness
    oracle_cal = [
        QuestionResult(question_id=r.question_id, correct=r.correct,
                       confidence=1.0 if r.correct else 0.0,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    report("Oracle calibration: conf=1.0 if correct, conf=0.0 if wrong", oracle_cal)

    # Oracle 2b: same, with conf=0.05 on wrong (matches the trivial-baseline
    # encoding our pipeline uses)
    oracle_cal_05 = [
        QuestionResult(question_id=r.question_id, correct=r.correct,
                       confidence=1.0 if r.correct else 0.05,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    report("Oracle calibration (refuse @ 0.05 instead of 0)", oracle_cal_05)

    # Oracle 3: every topic correct, conf=0.95
    oracle_ans = [
        QuestionResult(question_id=r.question_id, correct=True,
                       confidence=0.95,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    report("Oracle answer: every topic correct, conf=0.95", oracle_ans)

    # Combined: oracle answer with perfect calibration (conf=1.0)
    oracle_all = [
        QuestionResult(question_id=r.question_id, correct=True,
                       confidence=1.0,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    report("Oracle answer + perfect calibration (theoretical max)", oracle_all)

    # Save
    out = {
        "baseline_HMR": compute_metrics(base).HMR,
        "oracle_abstention_HMR": compute_metrics(oracle_abst).HMR,
        "oracle_calibration_perfect_HMR": compute_metrics(oracle_cal).HMR,
        "oracle_calibration_conf005_HMR": compute_metrics(oracle_cal_05).HMR,
        "oracle_answer_HMR": compute_metrics(oracle_ans).HMR,
        "oracle_answer_plus_calibration_HMR": compute_metrics(oracle_all).HMR,
    }
    Path(EVAL / "oracle_upper_bounds.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"\n  saved {EVAL / 'oracle_upper_bounds.json'}")


if __name__ == "__main__":
    main()
