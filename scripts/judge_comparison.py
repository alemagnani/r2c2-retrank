#!/usr/bin/env python3
"""Live head-to-head: Sonnet judge vs Opus judge HMR per candidate.

Reads all `*_opus_judge.json` and pairs them with their Sonnet-judge counterpart.
Prints a sorted table. Re-run anytime to get latest.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE / "data" / "eval" / "ac_runs"


def load_metrics(p: Path) -> dict:
    return json.loads(p.read_text())["metrics"]


def main():
    pairs = []
    for opus_path in sorted(EVAL_DIR.glob("*_opus_judge.json")):
        # Find the Sonnet-judge counterpart (same prefix, no "_opus_judge")
        prefix = opus_path.stem.replace("_opus_judge", "")
        candidates = [
            EVAL_DIR / f"{prefix}.json",
        ]
        sonnet_path = next((p for p in candidates if p.exists()), None)
        if sonnet_path is None:
            continue
        try:
            opus_m = load_metrics(opus_path)
            sonnet_m = load_metrics(sonnet_path)
        except (KeyError, json.JSONDecodeError):
            continue
        pairs.append({
            "name": prefix,
            "sonnet_hmr": sonnet_m["HMR"],
            "opus_hmr": opus_m["HMR"],
            "sonnet_acc": sonnet_m["accuracy"],
            "opus_acc": opus_m["accuracy"],
            "sonnet_ro": sonnet_m["R_O"],
            "opus_ro": opus_m["R_O"],
            "sonnet_mnp": sonnet_m["mean_nugget_precision"],
            "opus_mnp": opus_m["mean_nugget_precision"],
        })

    pairs.sort(key=lambda r: -r["sonnet_hmr"])

    print(f"\n{'═'*92}")
    print(f"  Judge cross-validation — Sonnet vs Opus (sorted by Sonnet-judge HMR)")
    print(f"  {len(pairs)} candidates evaluated under both judges")
    print(f"{'═'*92}\n")

    print(f"{'Rank':<5}{'Candidate':<42}{'Sonnet HMR':>11}{'Opus HMR':>11}{'Δ':>9}{'AccS→O':>10}")
    print("─" * 92)

    abs_deltas = []
    rank_changes = 0
    for i, p in enumerate(pairs, 1):
        delta = p["opus_hmr"] - p["sonnet_hmr"]
        abs_deltas.append(abs(delta))
        ar = "→"
        if p["opus_acc"] > p["sonnet_acc"]: ar = "↑"
        elif p["opus_acc"] < p["sonnet_acc"]: ar = "↓"
        print(f"{i:<5}{p['name']:<42}"
              f"{p['sonnet_hmr']:>11.4f}{p['opus_hmr']:>11.4f}"
              f"{delta:>+9.4f}"
              f"  {p['sonnet_acc']:.3f}{ar}{p['opus_acc']:.3f}")

    if abs_deltas:
        print()
        print(f"  Mean |Δ HMR|:    {sum(abs_deltas)/len(abs_deltas):.4f}")
        print(f"  Max  |Δ HMR|:    {max(abs_deltas):.4f}")
        print(f"  # Δ ≥ 0.01:      {sum(1 for d in abs_deltas if d >= 0.01)}/{len(abs_deltas)}")

        # Rank correlation: how preserved is the ordering under each judge?
        sonnet_rank = sorted(pairs, key=lambda r: -r["sonnet_hmr"])
        opus_rank   = sorted(pairs, key=lambda r: -r["opus_hmr"])
        sonnet_idx = {p["name"]: i for i, p in enumerate(sonnet_rank)}
        opus_idx   = {p["name"]: i for i, p in enumerate(opus_rank)}
        rank_diff = [abs(sonnet_idx[n] - opus_idx[n]) for n in sonnet_idx]
        print(f"  Mean rank shift: {sum(rank_diff)/len(rank_diff):.2f}")
        print(f"  Max rank shift:  {max(rank_diff)}")


if __name__ == "__main__":
    main()
