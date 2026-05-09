[← Back to results index](README.md)

---

# Statistical Robustness

A diligent reviewer should ask whether HMR Δ values of 0.01–0.10 are
distinguishable at n=65. Short answer: **most headline differences are not detected at α=0.05; rankings ARE robust to dropping any single topic.**

## Leave-one-topic-out: 9/9 sign-stable

Re-evaluating nine headline pairs after dropping each topic in turn:

| Comparison | Full Δ HMR | Sign-stable |
|---|---:|---:|
| BITEM Sonnet refinement vs BITEM broad | +0.026 | **65/65** |
| BITEM-only vs Tier-1 (both Sonnet refined) | +0.051 | **65/65** |
| BITEM-Sonnet vs Tier-1 broad | +0.136 | **65/65** |
| confidence-saturation vs BITEM-Sonnet | +0.012 | **65/65** |
| BITEM-Sonnet vs S1=Haiku | +0.127 | **65/65** |
| BITEM-Sonnet vs S4=Haiku | +0.050 | **65/65** |
| BITEM-Sonnet vs all-Haiku | +0.170 | **65/65** |
| BITEM-Sonnet vs BITEM-Opus pipeline | +0.176 | **65/65** |
| BITEM-only vs Error404-only | +0.278 | **65/65** |

**No comparison flips on any leave-out.** Magnitudes vary by ±0.05 typical, ±0.10 max, but sign is preserved every time. The rankings are not driven by 1–2 topics.

## Cross-judge rank correlation (33 candidates)

Across all 33 candidate runs evaluated under both Sonnet 4.6 and Opus 4.7 judges:

- **Spearman ρ = 0.574** (moderate)
- **Kendall τ-b = 0.432**
- Mean |Δ HMR| = 0.060; max |Δ| = 0.228

| Top-K | Members shared (Sonnet ∩ Opus) |
|---:|---|
| 1 | **1/1** |
| 3 | **3/3** |
| 5 | 4/5 |
| 10 | 5/10 |

**Top end is judge-robust; mid-tier is judge-fragile.** Our submission decision (top-3) is preserved exactly. Ranking claims that depend on rank ≥ 5 are not safe from a single judge.

## Bootstrap 95% CIs (paired, 2000 iterations)

### Findings that survive (CIs exclude zero)

| Comparison | Δ HMR | 95% CI |
|---|---:|---|
| ★ top-1 ensemble vs BITEM baseline | +0.012 | [+0.008, +0.016] |
| ★ s1-Haiku vs all-Sonnet | −0.127 | [−0.296, −0.034] |
| ★ s4-Haiku vs all-Sonnet | −0.050 | [−0.118, −0.004] |
| ★ all-Haiku vs all-Sonnet | −0.170 | [−0.334, −0.080] |
| ★ Opus pipeline vs Sonnet pipeline | −0.176 | [−0.372, −0.075] |

### Findings within noise (CIs cross zero)

| Comparison | Δ HMR | 95% CI |
|---|---:|---|
| BITEM vs Tier-1 (broad) | −0.111 | [−0.458, +0.044] |
| Sonnet refinement vs broad on Tier-1 | +0.085 | [−0.045, +0.439] |
| CE refinement vs broad | −0.024 | [−0.309, +0.313] |
| lexical refinement vs broad | +0.007 | [−0.041, +0.110] |
| top-3 ensemble vs BITEM baseline | +0.007 | [−0.002, +0.014] |
| retrank+E404 (refined vs broad) | −0.172 | [−0.464, +0.010] |

**This is the right level of caveat.** RQ10 (LLM tier) and improvement-#7 (Opus pipeline) are robustly significant. RQ2 (pool composition) and RQ6 (refinement) are within noise — we describe them as *observed differences*, not *detected effects*.

## McNemar exact test on per-topic correctness

Bootstrap on Δ HMR has limited power at n=65. McNemar tests whether two
paired binary classifiers (correct vs incorrect on the same topics)
**flip on different topics**.

| Comparison | A-only | B-only | p |
|---|---:|---:|---:|
| ensemble_top1 vs BITEM-Sonnet | 0 | 0 | 1.00 |
| BITEM-Sonnet vs BITEM-broad (refinement) | 1 | 0 | 1.00 |
| BITEM-Sonnet vs Tier-1 Sonnet (pool) | 2 | 1 | 1.00 |
| BITEM-Sonnet vs BITEM-Opus pipeline | 4 | 1 | 0.375 |
| BITEM-Sonnet vs all-Haiku | 4 | 0 | 0.125 |
| BITEM-only vs Error404-only (single team) | 3 | 1 | 0.625 |
| BITEM-only vs WaterlooClarke-only | 6 | 1 | 0.125 |

**No comparison reaches p < 0.05.** At n=65 with most topics correct in
both arms, McNemar can't detect significance. **This is consistent with
the R_O bottleneck thesis**: systems agree on which topics are
answerable; they disagree on how confident to be about the wrong ones.
HMR differences come from confidence calibration, not from a different
correctness distribution.

## Cohen's κ (inter-judge agreement)

Per-topic Stage-B correctness verdict between Sonnet 4.6 and Opus 4.7
across nine candidate runs:

- **Mean κ = 0.84** ("almost perfect" by Landis-Koch)
- Top-3 candidates: κ = 1.00 (perfect agreement)
- Worst case (Tier-1 refined-CE): κ = 0.68 (still "substantial")

## Trivial baseline anchors

| Baseline | Acc | R_O | R_U | HMR |
|---|---:|---:|---:|---:|
| Always refuse, conf=0 | 0 | 1.00 | 1.00 | **1.000** |
| Always refuse, conf=0.05 | 0 | 0.95 | 1.00 | **0.974** |
| Random guess (50%), conf=0.5 | 0.49 | 0.50 | 0.50 | 0.500 |
| Always answer, conf=1.0, all wrong | 0 | 0.00 | 1.00 | 0.000 |
| 90% acc oracle, conf=0.5 | 0.90 | 1.00 | 0.50 | 0.667 |
| **Our submitted (ensemble_top1)** | **0.94** | 0.95 | 1.00 | **0.974** |

The always-refuse-at-0.05 baseline ties our best HMR. Distinguished
from our system on accuracy (0.94 vs 0) and on Cnf-W (0 vs 0).
**Read HMR jointly with accuracy and refusal rate — never alone.**

## Reproduce

```bash
python scripts/bootstrap_ci.py
python scripts/mcnemar_tests.py
python scripts/inter_judge_agreement.py
python scripts/trivial_baselines.py
```
