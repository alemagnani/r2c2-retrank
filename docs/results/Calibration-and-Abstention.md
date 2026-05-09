[← Back to results index](README.md)

---

# Calibration & Abstention

## Two calibration variants

The pipeline produces a `c_self` from Stage 1 and `match_score ∈ {0,1,2}`
from Stage 4. We test two ways to combine them:

| Variant | Rule |
|---|---|
| **A (refusal-aware self-report)** | `c_final = 5 if refused or no nuggets`, `min(c_self, 25) if ms=0`, `min(c_self, 60) if ms=1`, `c_self if ms=2` |
| **B (ensemble agreement)** | confidence proportional to fraction of agreeing samples; refuse below 40% |

Variant A wins consistently. We use it for all four submitted runs.

## Abstention rules (post-hoc on Tier-1 refined-Sonnet)

| Rule | Abst. rate | Acc | HMR |
|---|---:|---:|---:|
| **R2** (refuse if ms<2) | 9.2% | 0.908 | **0.956** |
| **R1** (refuse if ms=0) | 7.7% | 0.923 | 0.953 |
| R4 (refuse if c_self<70) | 7.7% | 0.908 | 0.942 |
| R0 (Variant A baseline) | 4.6% | 0.923 | 0.911 |

R1 is the practical sweet spot: **+0.042 HMR over baseline with no accuracy loss**. R2 maxes HMR but trades 1.5pp of accuracy.

## Verifier match-score is a usable correctness signal

Across the four Tier-1 refinement variants:
- ms=2 (full match) on 90.8–92.3% of topics
- Within ms=2: candidate-answer accuracy > 95%
- Within ms=0: candidate-answer accuracy < 30%

The verifier doesn't know the truth; it only knows whether the entailed
nuggets re-derive the candidate answer. But that proxy is a strong
correctness signal because Stage 1 and Stage 4 use independent reasoning
paths over the same evidence.

## Calibration bin reliability

| Candidate | ECE | [0,0.2) | [0.2,0.4) | [0.6,0.8) | [0.8,1.0] |
|---|---:|---|---|---|---|
| ensemble_top1 | 0.003 | 0/4 | — | — | 61/61 |
| BITEM-Sonnet | 0.026 | 0/4 | — | — | 61/61 |
| BITEM-Opus | 0.024 | 0/1 | 2/7 | — | 56/57 |
| BITEM-broad | 0.031 | 0/4 | 1/2 | — | 59/59 |
| Tier-1 Sonnet | 0.034 | 0/3 | 1/3 | 1/1 | 58/58 |
| Tier-1 broad | 0.028 | 0/3 | 2/3 | — | 58/59 |

Pipelines are **highly bimodal**: confidences cluster at ~0.05 (refusals)
and ≥ 0.80 (answers). ECE is small because both modes are well-calibrated.
**Caveat:** the "correct" label and Stage 3 bogus filter both involve
Sonnet, so the high-confidence bin reliability may partly reflect
same-family-judge agreement; we re-evaluate under Opus
(see [Judge Cross-Validation](Judge-Cross-Validation.md)).

## Reproduce

```bash
python scripts/calibration_bins.py
python scripts/refusal_decomposition.py
```
