#!/usr/bin/env python3
"""Reliability diagram for confidence calibration.

Bin per-topic predictions by confidence, plot empirical accuracy vs
bin midpoint. Perfect calibration = diagonal y=x.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"
OUT = BASE / "docs" / "figures" / "fig_reliability.pdf"

BINS = [(0.00, 0.10), (0.10, 0.30), (0.30, 0.50),
        (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]


def reliability(p: Path):
    d = json.loads(p.read_text())
    qs = list(d["per_question"].values())
    out = []
    for lo, hi in BINS:
        in_bin = [q for q in qs if lo <= float(q.get("confidence", 0)) < hi]
        n = len(in_bin)
        if n == 0:
            out.append((lo, hi, n, None, None))
            continue
        acc = sum(1 for q in in_bin if q.get("correct")) / n
        mc = sum(float(q.get("confidence", 0)) for q in in_bin) / n
        out.append((lo, hi, n, mc, acc))
    return out


def main():
    candidates = [
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet (main)"),
        ("bitem_only_broad_A", "BITEM-broad"),
        ("refined_sonnet_A", "Tier-1 Sonnet"),
        ("bitem_only_opus_A", "BITEM-Opus"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f"]
    markers = ["o", "s", "^", "D"]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="perfect calibration")

    for (stem, label), color, marker in zip(candidates, colors, markers):
        p = EVAL / f"{stem}.json"
        if not p.exists():
            continue
        bins = reliability(p)
        confs = [b[3] for b in bins if b[3] is not None]
        accs = [b[4] for b in bins if b[4] is not None]
        ns = [b[2] for b in bins if b[3] is not None]
        sizes = [max(20, n * 4) for n in ns]
        ax.scatter(confs, accs, s=sizes, color=color, marker=marker,
                   alpha=0.75, edgecolors="black", linewidth=0.6, label=label)
        ax.plot(confs, accs, color=color, alpha=0.4, linewidth=1)

    ax.set_xlabel("Mean confidence in bin", fontsize=11)
    ax.set_ylabel("Empirical accuracy in bin", fontsize=11)
    ax.set_title("Reliability diagram — confidence calibration\n"
                 "Marker size ∝ n in bin. Pipelines are bimodal (refusals at ~0.05; "
                 "confident answers at ≥ 0.8).", fontsize=10, loc="left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main()
