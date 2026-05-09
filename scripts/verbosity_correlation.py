#!/usr/bin/env python3
"""Test the verbosity-bias hypothesis.

Hypothesis: Stage B's correctness verdict is influenced by answer length.
Shorter answers are penalised even when they are factually correct (because
the eval can't ground them via the answer string + nuggets together).

Compute Spearman ρ between answer length and Stage B `correct` for:
  (a) Sonnet judge — should show positive correlation if hypothesis holds
  (b) Opus judge — for comparison

If Sonnet shows higher length-correctness correlation than Opus, that's
quantitative evidence of Sonnet judge's verbosity bias.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE / "data" / "eval" / "ac_runs"


def spearman(xs: list[float], ys: list[float]) -> float:
    """Quick Spearman ρ implementation (no scipy)."""
    n = len(xs)
    if n < 2:
        return 0.0
    def ranks(vals):
        sorted_v = sorted(enumerate(vals), key=lambda p: p[1])
        rks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_v[j + 1][1] == sorted_v[i][1]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rks[sorted_v[k][0]] = avg
            i = j + 1
        return rks

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def per_topic_data(eval_path: Path) -> list[tuple[int, bool]]:
    """Returns list of (answer_length, correct) per topic."""
    d = json.loads(eval_path.read_text())
    out = []
    for qid, q in d["per_question"].items():
        ans = (q.get("answer") or "").strip()
        out.append((len(ans), bool(q.get("correct", False))))
    return out


def main():
    candidates = [
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet pipeline"),
        ("bitem_only_opus_A", "BITEM-Opus pipeline"),
        ("bitem_only_broad_A", "BITEM broad"),
        ("ensemble_top1", "ensemble top-1"),
        ("refined_sonnet_A", "Tier-1 refined Sonnet"),
    ]

    print(f"\n{'─'*88}")
    print(f"Spearman ρ between answer length and Stage B correctness, per judge")
    print(f"{'─'*88}\n")
    print(f"{'Candidate':<32} {'Sonnet ρ':>10} {'Opus ρ':>10} {'Δ':>10} "
          f"{'Acc':>5} {'mean_len':>8}")
    print("─" * 88)

    rows = []
    for name, label in candidates:
        sonnet_path = EVAL_DIR / f"{name}.json"
        opus_path = EVAL_DIR / f"{name}_opus_judge.json"
        if not (sonnet_path.exists() and opus_path.exists()):
            continue
        s_data = per_topic_data(sonnet_path)
        o_data = per_topic_data(opus_path)
        # Same answers under both (only judge differs)
        # Use sonnet's lengths (should match opus's)
        lens = [t[0] for t in s_data]
        s_corr = [int(t[1]) for t in s_data]
        o_corr = [int(t[1]) for t in o_data]
        rho_s = spearman(lens, s_corr)
        rho_o = spearman(lens, o_corr)
        acc_s = sum(s_corr) / len(s_corr) if s_corr else 0
        mean_len = sum(lens) / len(lens) if lens else 0
        rows.append((label, rho_s, rho_o, acc_s, mean_len))
        print(f"{label:<32} {rho_s:>10.3f} {rho_o:>10.3f} {(rho_s - rho_o):>+10.3f} "
              f"{acc_s:>5.3f} {mean_len:>8.0f}")

    # Aggregate: are Sonnet judges generally more length-biased than Opus?
    diffs = [r[1] - r[2] for r in rows]
    print()
    print(f"  Mean Δρ (Sonnet ρ − Opus ρ): {sum(diffs)/len(diffs):+.3f}")
    if sum(diffs) / len(diffs) > 0.05:
        print(f"  → Sonnet judge appears MORE correlated with answer length "
              f"than Opus judge: SUPPORTS verbosity-bias hypothesis")
    elif sum(diffs) / len(diffs) < -0.05:
        print(f"  → Opus judge appears MORE correlated with length: "
              f"REFUTES verbosity-bias on Sonnet")
    else:
        print(f"  → Length-correctness correlations are similar across judges; "
              f"verbosity-bias is symmetric or absent")

    # Also: cross-judge agreement on per-topic correctness
    print(f"\n{'─'*88}")
    print(f"Inter-judge per-topic correctness agreement")
    print(f"{'─'*88}")
    for name, label in candidates:
        sonnet_path = EVAL_DIR / f"{name}.json"
        opus_path = EVAL_DIR / f"{name}_opus_judge.json"
        if not (sonnet_path.exists() and opus_path.exists()):
            continue
        s_data = per_topic_data(sonnet_path)
        o_data = per_topic_data(opus_path)
        agree = sum(1 for s, o in zip(s_data, o_data) if s[1] == o[1])
        s_yes_o_no = sum(1 for s, o in zip(s_data, o_data) if s[1] and not o[1])
        s_no_o_yes = sum(1 for s, o in zip(s_data, o_data) if not s[1] and o[1])
        print(f"  {label:<32} agree {agree}/65 ({agree/65*100:.0f}%)  "
              f"Sonnet-only-correct: {s_yes_o_no}, Opus-only-correct: {s_no_o_yes}")

    # Length distribution of Sonnet-only-correct vs Opus-only-correct
    print(f"\n{'─'*88}")
    print(f"Median answer length: agreed-correct vs disagreed (Sonnet-only or Opus-only)")
    print(f"{'─'*88}")
    for name, label in candidates:
        sonnet_path = EVAL_DIR / f"{name}.json"
        opus_path = EVAL_DIR / f"{name}_opus_judge.json"
        if not (sonnet_path.exists() and opus_path.exists()):
            continue
        s_data = per_topic_data(sonnet_path)
        o_data = per_topic_data(opus_path)
        agreed_correct = [s[0] for s, o in zip(s_data, o_data) if s[1] and o[1]]
        sonnet_only = [s[0] for s, o in zip(s_data, o_data) if s[1] and not o[1]]
        opus_only = [s[0] for s, o in zip(s_data, o_data) if not s[1] and o[1]]
        if agreed_correct:
            ac_med = median(agreed_correct)
        else:
            ac_med = 0
        so_med = median(sonnet_only) if sonnet_only else 0
        oo_med = median(opus_only) if opus_only else 0
        print(f"  {label:<32}  agreed-correct: {ac_med:.0f}, sonnet-only: "
              f"{so_med:.0f} (n={len(sonnet_only)}), opus-only: "
              f"{oo_med:.0f} (n={len(opus_only)})")


if __name__ == "__main__":
    main()
