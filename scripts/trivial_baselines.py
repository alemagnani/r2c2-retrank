#!/usr/bin/env python3
"""Trivial baselines to anchor HMR magnitudes.

Without HMR baselines, "0.974 is the best" is meaningless to a reviewer.
Run several trivial baselines to provide context.

  Baseline 1 — always-refuse: empty answer + conf=0.05 on every topic.
                              R_O=1 (vacuous), R_U=1 (vacuous), Acc=0, HMR=1.
                              [the trivial degenerate ceiling]
  Baseline 2 — always-refuse-conf-zero: empty + conf=0. HMR formula
                              edge-case sanity check.
  Baseline 3 — confident-wrong: every answer is "Yes" with conf=1.0.
                              Worst-case overconfidence.
  Baseline 4 — neutral-wrong: every answer is "Yes" with conf=0.5.
                              R_O=0.5 if all wrong.
  Baseline 5 — random-from-pool: pick a random passage's first phrase
                              as the answer. Conf=0.5.
                              Lower bound for "any retrieval works".

We evaluate baselines using the same self-evaluator as for the real pipelines.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402


def synthesize_results(actual_correct_count: int, n_topics: int,
                        confidence: float, n_returned: int = 0,
                        n_relevant: int = 0) -> list[QuestionResult]:
    """Build a synthetic per-question results list.

    `actual_correct_count` of the n_topics will be marked correct;
    the rest are wrong. All have the given confidence and nugget counts.
    """
    out = []
    for i in range(n_topics):
        out.append(QuestionResult(
            question_id=f"Q{i:04d}",
            correct=(i < actual_correct_count),
            confidence=confidence,
            nuggets_returned=n_returned,
            nuggets_relevant=n_relevant,
        ))
    return out


def main():
    n = 65
    print(f"\n{'─'*88}")
    print(f"Trivial baselines on n={n} topics, for HMR magnitude anchoring")
    print(f"{'─'*88}\n")
    print(f"{'Baseline':<48} {'Acc':>5} {'MNP':>5} {'R_O':>5} {'R_U':>5} {'HMR':>5}")
    print("─" * 88)

    baselines = [
        ("Always refuse (conf=0.05, empty answer, no nuggets)",
         0, 0.05, 0, 0),
        ("Always answer with conf=1.0, all wrong",
         0, 1.0, 0, 0),
        ("Always answer with conf=0.5, all wrong",
         0, 0.5, 0, 0),
        ("Always answer with conf=1.0, all RIGHT (oracle)",
         n, 1.0, 0, 0),
        ("Always answer with conf=0.5, all RIGHT (oracle)",
         n, 0.5, 0, 0),
        ("Random guess: 50% accuracy, conf=0.5",
         n // 2, 0.5, 0, 0),
        ("Random guess: 50% accuracy, conf=0.05 (very humble)",
         n // 2, 0.05, 0, 0),
        ("Random guess: 50% accuracy, conf=1.0 (very confident)",
         n // 2, 1.0, 0, 0),
        ("90% accurate (like our best), conf=0.95",
         int(n * 0.90), 0.95, 0, 0),
        ("90% accurate, conf=0.50 (under-confident)",
         int(n * 0.90), 0.50, 0, 0),
    ]

    for label, n_correct, conf, n_ret, n_rel in baselines:
        results = synthesize_results(n_correct, n, conf, n_ret, n_rel)
        m = compute_metrics(results)
        print(f"  {label:<48} {m.accuracy:>5.3f} {m.mean_nugget_precision:>5.3f} "
              f"{m.R_O:>5.3f} {m.R_U:>5.3f} {m.HMR:>5.3f}")

    # Compare to our submitted candidates
    print(f"\n{'─'*88}")
    print(f"Top submitted runs (for reference)")
    print(f"{'─'*88}")
    for fname, label in [
        ("ensemble_top1.json", "ensemble_top1"),
        ("ensemble_top3.json", "ensemble_top3"),
        ("bitem_only_refined_sonnet_A.json", "bitem_only_refined_sonnet"),
    ]:
        d = json.loads((BASE / "data/eval/ac_runs" / fname).read_text())
        m = d["metrics"]
        print(f"  {label:<48} {m['accuracy']:>5.3f} {m['mean_nugget_precision']:>5.3f} "
              f"{m['R_O']:>5.3f} {m['R_U']:>5.3f} {m['HMR']:>5.3f}")

    # Save
    out = []
    for label, n_correct, conf, n_ret, n_rel in baselines:
        results = synthesize_results(n_correct, n, conf, n_ret, n_rel)
        m = compute_metrics(results)
        out.append({
            "name": label,
            **m.as_dict(),
        })
    Path(BASE / "data/eval/ac_runs/trivial_baselines.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
