#!/usr/bin/env python3
"""Held-out split analysis: dev=30, test=35.

Re-rank candidates by HMR on dev-only, freeze the choice, report
HMR + accuracy on test-only. Compares to "if we'd selected on full
65, what would test-only HMR be?" to expose any overfitting.

We use a deterministic split (sorted topic IDs, first 30 = dev).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from eval.hmr import QuestionResult, compute_metrics  # noqa: E402

EVAL = BASE / "data" / "eval" / "ac_runs"
DEV_N = 30  # first 30 topics


def load_results(p: Path) -> list[QuestionResult]:
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


def split_metrics(results, dev_qids, test_qids):
    dev = [r for r in results if r.question_id in dev_qids]
    test = [r for r in results if r.question_id in test_qids]
    return compute_metrics(dev), compute_metrics(test), len(dev), len(test)


def main():
    candidates = sorted({
        p.stem for p in EVAL.glob("*.json")
        if not p.stem.endswith("_opus_judge")
        and not p.stem.startswith((
            "trivial", "bootstrap", "abstention_sweep", "loo_",
            "judge_specific", "oracle_", "refusal_", "calibration_",
            "pr_", "mcnemar_", "nugget_first"))
    })

    # Determine the topic ID universe from the first candidate
    sample = json.loads((EVAL / f"{candidates[0]}.json").read_text())
    all_qids = sorted(sample["per_question"].keys())
    dev_qids = set(all_qids[:DEV_N])
    test_qids = set(all_qids[DEV_N:])
    print(f"\nHeld-out split: dev={len(dev_qids)} topics ({sorted(dev_qids)[0]}-{sorted(dev_qids)[-1]}), "
          f"test={len(test_qids)} topics ({sorted(test_qids)[0]}-{sorted(test_qids)[-1]})\n")

    rows = []
    for cand in candidates:
        p = EVAL / f"{cand}.json"
        try:
            r = load_results(p)
            if not r:
                continue
            full = compute_metrics(r)
            dev_m, test_m, n_dev, n_test = split_metrics(r, dev_qids, test_qids)
            rows.append({
                "candidate": cand,
                "HMR_full": full.HMR,
                "Acc_full": full.accuracy,
                "HMR_dev": dev_m.HMR,
                "Acc_dev": dev_m.accuracy,
                "HMR_test": test_m.HMR,
                "Acc_test": test_m.accuracy,
                "n_dev": n_dev,
                "n_test": n_test,
            })
        except Exception:
            continue

    rows.sort(key=lambda r: -r["HMR_full"])
    print(f"{'Candidate':<48} {'HMR_full':>8} {'HMR_dev':>8} {'HMR_test':>8} "
          f"{'Acc_test':>8} {'rank_full':>9} {'rank_dev':>8} {'rank_test':>9}")
    print("─" * 116)

    # Compute ranks on each split
    by_full = sorted(rows, key=lambda r: -r["HMR_full"])
    by_dev = sorted(rows, key=lambda r: -r["HMR_dev"])
    by_test = sorted(rows, key=lambda r: -r["HMR_test"])
    rank_full = {r["candidate"]: i + 1 for i, r in enumerate(by_full)}
    rank_dev = {r["candidate"]: i + 1 for i, r in enumerate(by_dev)}
    rank_test = {r["candidate"]: i + 1 for i, r in enumerate(by_test)}

    for r in rows[:18]:
        c = r["candidate"]
        print(f"  {c[:46]:<46} {r['HMR_full']:>8.3f} {r['HMR_dev']:>8.3f} "
              f"{r['HMR_test']:>8.3f} {r['Acc_test']:>8.3f} "
              f"{rank_full[c]:>9} {rank_dev[c]:>8} {rank_test[c]:>9}")

    # The honest test: pick top-K by dev, evaluate on test
    print(f"\n{'─'*70}")
    print("Honest selection: pick top-K by DEV-ONLY HMR, evaluate on TEST")
    print(f"{'─'*70}")
    for k in [1, 3, 5]:
        top_k_dev = [r["candidate"] for r in by_dev[:k]]
        test_hmrs = [r["HMR_test"] for r in rows if r["candidate"] in top_k_dev]
        test_accs = [r["Acc_test"] for r in rows if r["candidate"] in top_k_dev]
        full_hmrs = [r["HMR_full"] for r in rows if r["candidate"] in top_k_dev]
        print(f"\n  Top-{k} by dev: {top_k_dev}")
        print(f"    Mean HMR_test  = {sum(test_hmrs)/len(test_hmrs):.4f}")
        print(f"    Mean Acc_test  = {sum(test_accs)/len(test_accs):.4f}")
        print(f"    Mean HMR_full  = {sum(full_hmrs)/len(full_hmrs):.4f}")

    # Compare to "if we'd just used full-set selection"
    print(f"\n{'─'*70}")
    print("For reference: top-K by FULL HMR (the overfitted selection)")
    print(f"{'─'*70}")
    for k in [1, 3]:
        top_k_full = [r["candidate"] for r in by_full[:k]]
        test_hmrs = [r["HMR_test"] for r in rows if r["candidate"] in top_k_full]
        print(f"\n  Top-{k} by full: {top_k_full}")
        print(f"    Mean HMR_test  = {sum(test_hmrs)/len(test_hmrs):.4f}")

    # Rank stability: how many top-3 by dev are also top-3 by full?
    top3_full = set(r["candidate"] for r in by_full[:3])
    top3_dev = set(r["candidate"] for r in by_dev[:3])
    top3_test = set(r["candidate"] for r in by_test[:3])
    print(f"\n{'─'*70}")
    print(f"Top-3 set agreement:")
    print(f"  full ∩ dev:  {len(top3_full & top3_dev)}/3 — {sorted(top3_full & top3_dev)}")
    print(f"  full ∩ test: {len(top3_full & top3_test)}/3 — {sorted(top3_full & top3_test)}")
    print(f"  dev  ∩ test: {len(top3_dev  & top3_test)}/3 — {sorted(top3_dev  & top3_test)}")

    Path(EVAL / "heldout_split.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    print(f"\n  saved {EVAL / 'heldout_split.json'}")


if __name__ == "__main__":
    main()
