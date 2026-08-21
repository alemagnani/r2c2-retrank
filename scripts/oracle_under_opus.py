#!/usr/bin/env python3
"""Oracle bounds re-run using Opus correctness labels.

The original oracle_upper_bounds.py used the Sonnet self-evaluator's
"correct" labels. The harsh-review concern: if our system has Cnf-W=0
under Sonnet labels, the oracle bound is tautological under self-eval.

Repeat the oracle bound calculation using Opus 4.7's labels for
correctness instead. If the bound still equals our system's HMR (under
Opus judge), the finding is robust. If not, the bound is a self-eval
artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

EVAL = BASE / "data" / "eval" / "ac_runs"


def load_results_from(path: Path) -> list[QuestionResult]:
    d = json.loads(path.read_text())
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
    return m


def run_for_judge(judge_label: str, eval_path: Path):
    """Run the oracle bounds for one judge's correctness labels."""
    base = load_results_from(eval_path)
    print(f"\n{'═'*92}")
    print(f"Oracle upper bounds under {judge_label} judge labels")
    print(f"{'═'*92}\n")

    baseline_m = report("BITEM-Sonnet (submitted)", base)

    oracle_abst = [
        QuestionResult(question_id=r.question_id,
                       correct=r.correct,
                       confidence=0.05 if not r.correct else r.confidence,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    abst_m = report("Oracle abstention (refuse exactly the topics we get wrong)", oracle_abst)

    oracle_cal = [
        QuestionResult(question_id=r.question_id, correct=r.correct,
                       confidence=1.0 if r.correct else 0.0,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    cal_m = report("Oracle calibration: conf=1.0 if correct, conf=0.0 if wrong", oracle_cal)

    oracle_cal_05 = [
        QuestionResult(question_id=r.question_id, correct=r.correct,
                       confidence=1.0 if r.correct else 0.05,
                       nuggets_returned=r.nuggets_returned,
                       nuggets_relevant=r.nuggets_relevant)
        for r in base
    ]
    cal05_m = report("Oracle calibration (refuse @ 0.05 instead of 0)", oracle_cal_05)

    return {
        "judge": judge_label,
        "baseline_HMR": baseline_m.HMR,
        "oracle_abstention_HMR": abst_m.HMR,
        "oracle_calibration_HMR": cal_m.HMR,
        "oracle_calibration_05_HMR": cal05_m.HMR,
        "baseline_Cnf_W": sum(1 for r in base
                              if not r.correct and r.confidence >= 0.5),
    }


def main():
    sonnet_path = EVAL / "bitem_only_refined_sonnet_A.json"
    opus_path = EVAL / "bitem_only_refined_sonnet_A_opus_judge.json"
    if not opus_path.exists():
        print(f"  ERROR: Opus judge file not found at {opus_path}"); return

    sonnet = run_for_judge("Sonnet", sonnet_path)
    opus = run_for_judge("Opus", opus_path)

    print(f"\n{'═'*92}")
    print(f"Comparison: does the oracle-abstention = baseline finding survive judge change?")
    print(f"{'═'*92}\n")
    for k in ["baseline_HMR", "oracle_abstention_HMR",
              "oracle_calibration_HMR", "oracle_calibration_05_HMR"]:
        print(f"  {k:<35} Sonnet: {sonnet[k]:.3f}  Opus: {opus[k]:.3f}  "
              f"Δ = {opus[k] - sonnet[k]:+.3f}")
    print(f"\n  Cnf-W: Sonnet {sonnet['baseline_Cnf_W']}  Opus {opus['baseline_Cnf_W']}\n")

    abst_eq_baseline_sonnet = abs(sonnet["oracle_abstention_HMR"] - sonnet["baseline_HMR"]) < 1e-6
    abst_eq_baseline_opus = abs(opus["oracle_abstention_HMR"] - opus["baseline_HMR"]) < 1e-6
    print(f"  Oracle-abstention = baseline under Sonnet judge: {abst_eq_baseline_sonnet}")
    print(f"  Oracle-abstention = baseline under Opus judge:   {abst_eq_baseline_opus}")
    if abst_eq_baseline_sonnet and abst_eq_baseline_opus:
        print(f"\n  → Finding survives judge change: oracle-abstention bound equals baseline under "
              f"BOTH judges.\n    The 'we are at the abstention ceiling' claim is robust.")
    else:
        print(f"\n  → Finding is judge-dependent: re-investigate.")

    Path(EVAL / "oracle_under_opus.json").write_text(
        json.dumps({"sonnet": sonnet, "opus": opus}, indent=2)
    )
    print(f"\n  saved {EVAL / 'oracle_under_opus.json'}")


if __name__ == "__main__":
    main()
