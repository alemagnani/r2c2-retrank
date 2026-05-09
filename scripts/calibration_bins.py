#!/usr/bin/env python3
"""Confidence-calibration reliability bins.

For each candidate, bin per-topic predictions by confidence and report
empirical accuracy in that bin. Well-calibrated systems should have
per-bin accuracy ≈ bin midpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"

BINS = [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]


def reliability(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    qs = list(d["per_question"].values())
    out = []
    for lo, hi in BINS:
        in_bin = [q for q in qs if lo <= float(q.get("confidence", 0)) < hi]
        n = len(in_bin)
        acc = sum(1 for q in in_bin if q.get("correct")) / n if n else 0.0
        mean_conf = sum(float(q.get("confidence", 0)) for q in in_bin) / n if n else 0.0
        out.append({"lo": lo, "hi": hi, "n": n, "acc": acc, "mean_conf": mean_conf,
                    "ece_contrib": abs(acc - mean_conf) * n})
    return out


def main():
    candidates = [
        ("ensemble_top1", "ensemble_top1"),
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet"),
        ("bitem_only_opus_A", "BITEM-Opus"),
        ("bitem_only_broad_A", "BITEM-broad"),
        ("refined_sonnet_A", "Tier-1 Sonnet"),
        ("broad_A", "Tier-1 broad"),
    ]

    print(f"\n{'─'*88}")
    print(f"Confidence calibration: per-bin accuracy")
    print(f"{'─'*88}\n")

    out_data = {}
    for stem, label in candidates:
        p = EVAL / f"{stem}.json"
        if not p.exists():
            continue
        bins = reliability(p)
        n_total = sum(b["n"] for b in bins)
        ece = sum(b["ece_contrib"] for b in bins) / n_total if n_total else 0.0
        print(f"{label:<24}  ECE = {ece:.3f}")
        print(f"  {'bin':<14} {'n':>3} {'acc':>5} {'meanConf':>9}  gap")
        for b in bins:
            if b["n"] == 0:
                continue
            gap = b["acc"] - b["mean_conf"]
            print(f"  [{b['lo']:.2f},{b['hi']:.2f}) {b['n']:>4} {b['acc']:>5.2f} "
                  f"{b['mean_conf']:>9.3f}  {gap:+.2f}")
        out_data[label] = {"ece": ece, "bins": bins}
        print()

    Path(EVAL / "calibration_bins.json").write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False)
    )
    print(f"  Saved to {EVAL / 'calibration_bins.json'}")


if __name__ == "__main__":
    main()
