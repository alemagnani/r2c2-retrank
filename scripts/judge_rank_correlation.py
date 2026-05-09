#!/usr/bin/env python3
"""Spearman + Kendall rank correlation between Sonnet-judge and
Opus-judge HMR rankings across all candidates that have both files.

Direct quantitative answer to "how judge-dependent are our rankings?"
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2: return 0.0
    def ranks(vals):
        s = sorted(enumerate(vals), key=lambda p: p[1])
        rks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and s[j + 1][1] == s[i][1]: j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1): rks[s[k][0]] = avg
            i = j + 1
        return rks
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2: return 0.0
    conc = disc = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx == 0 and dy == 0: continue
            if dx == 0: ties_x += 1; continue
            if dy == 0: ties_y += 1; continue
            if (dx > 0) == (dy > 0): conc += 1
            else: disc += 1
    n0 = n * (n - 1) // 2
    den = ((n0 - ties_x) * (n0 - ties_y)) ** 0.5
    return (conc - disc) / den if den else 0.0


def main():
    sonnet_files = sorted(p for p in EVAL.glob("*.json")
                          if not p.stem.endswith("_opus_judge")
                          and not p.stem.startswith(("trivial", "bootstrap",
                              "abstention_sweep", "loo_", "judge_specific",
                              "oracle_", "refusal_", "calibration_", "pr_",
                              "mcnemar_", "nugget_first")))
    pairs = []
    for s in sonnet_files:
        o = EVAL / f"{s.stem}_opus_judge.json"
        if o.exists():
            try:
                ds = json.loads(s.read_text())["metrics"]["HMR"]
                do = json.loads(o.read_text())["metrics"]["HMR"]
                pairs.append((s.stem, ds, do))
            except Exception:
                pass
    pairs.sort(key=lambda r: -r[1])

    print(f"\n{'─'*88}")
    print(f"Sonnet vs Opus judge HMR across {len(pairs)} candidates")
    print(f"{'─'*88}\n")
    print(f"{'Candidate':<48} {'Sonnet':>8} {'Opus':>8} {'Δ':>8}")
    for name, s, o in pairs:
        print(f"  {name[:46]:<46} {s:>8.4f} {o:>8.4f} {o-s:>+8.4f}")

    xs = [p[1] for p in pairs]; ys = [p[2] for p in pairs]
    rho = spearman(xs, ys)
    tau = kendall_tau_b(xs, ys)
    print(f"\n  n = {len(pairs)}")
    print(f"  Spearman ρ  = {rho:.4f}")
    print(f"  Kendall τ-b = {tau:.4f}")
    print(f"  Mean |Δ|    = {sum(abs(o - s) for _, s, o in pairs) / len(pairs):.4f}")
    print(f"  Max  |Δ|    = {max(abs(o - s) for _, s, o in pairs):.4f}")

    # Top-K stability: how often does the top-K under Sonnet equal top-K under Opus?
    sonnet_ranked = [p[0] for p in sorted(pairs, key=lambda r: -r[1])]
    opus_ranked = [p[0] for p in sorted(pairs, key=lambda r: -r[2])]
    print(f"\n  Top-K membership stability (Sonnet ∩ Opus):")
    for k in [1, 3, 5, 10]:
        if k > len(pairs): continue
        ss = set(sonnet_ranked[:k]); os_ = set(opus_ranked[:k])
        print(f"    Top-{k}: {len(ss & os_)}/{k} candidates shared")

    out = {"n": len(pairs), "spearman": rho, "kendall_tau_b": tau,
           "mean_abs_delta": sum(abs(o - s) for _, s, o in pairs) / len(pairs),
           "max_abs_delta": max(abs(o - s) for _, s, o in pairs),
           "pairs": [{"name": n, "sonnet": s, "opus": o} for n, s, o in pairs]}
    Path(EVAL / "judge_rank_correlation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved {EVAL / 'judge_rank_correlation.json'}")


if __name__ == "__main__":
    main()
