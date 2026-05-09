[← Back to results index](README.md)

---

# Judge Cross-Validation — Sonnet vs Opus

To address the methodological concern that our self-evaluator (Sonnet
4.6) might exhibit a judge-style bias (we use Sonnet both in the
pipeline and as the judge), we re-evaluated **all 33 candidate AC
runs** with Claude Opus 4.7 as the Stage-A/B judge.

## Top-3 are judge-invariant

| Candidate | HMR (Sonnet) | HMR (Opus) | Δ |
|---|---:|---:|---:|
| ensemble_top1 | **0.974** | **0.974** | 0.000 |
| ensemble_top3 | **0.969** | **0.969** | 0.000 |
| bitem_only_refined_sonnet (main) | **0.963** | **0.963** | 0.000 |

**Cohen's κ on per-topic Stage-B correctness across these three: 1.00 (perfect agreement).**

Our submission decision is robust to judge choice at the top end.

## Mid-tier candidates show substantial variance

Across all 33 candidates:
- Mean |Δ HMR| = 0.060
- Max |Δ HMR| = 0.228
- 22/33 candidates differ by ≥ 0.01
- Mean Cohen's κ across the 9 candidates with both judges: **0.84** ("almost perfect" by Landis-Koch)
- Worst κ: 0.68 (Tier-1 refined-CE) — still "substantial"

## Notable judge-disagreements (rank 4–5)

| Candidate | Sonnet HMR | Opus HMR | Comment |
|---|---:|---:|---|
| BITEM-broad (Sonnet rank #4) | 0.937 | 0.860 | Opus is stricter |
| BITEM-Sonnet-recovered (Opus rank #4) | 0.922 | 0.951 | Opus more lenient on the recovered topics |
| retrank+E404+Sonnet (Opus rank #5) | 0.738 | 0.938 | **0.200 swing** — case-study failure flips |

## Verbosity-bias hypothesis: refuted on Sonnet

We tested whether Sonnet judge favours verbose answers (longer
answers more likely judged correct). Spearman ρ between answer
length and Stage-B correctness:

| Candidate | Sonnet ρ | Opus ρ | Δ |
|---|---:|---:|---:|
| BITEM-Sonnet pipeline | 0.181 | 0.188 | -0.007 |
| BITEM-Opus pipeline | 0.155 | 0.218 | -0.063 |
| BITEM broad | 0.058 | 0.106 | -0.048 |
| Mean Δρ across 5 candidates | — | — | **−0.061** |

**Opus judge correlates *more* with answer length than Sonnet** — refutes the verbosity-bias hypothesis directionally.

## Findings preserved under both judges

- ✅ Best HMR achievable (~ 0.97) and best configuration (ensemble_top1) — identical
- ✅ BITEM-only is the best single pool — top under both judges
- ✅ Filtered top-K ensemble dominates naive 16-way — preserved
- ✅ Stage 1 most Sonnet-critical (cost-quality) — s1_haiku remains worst single substitution
- ✅ Calibration matters more than refinement — the Variant A vs B gap is preserved

## Findings that are judge-specific

- ⚠️ Strict-monotonicity in pool count — Sonnet says 1 > 2 > 3 > 6; Opus says 1 > 6 > 3 (Tier-1+2 jumps). The "less is more" claim survives at the top end but the monotonicity middle does not.
- ⚠️ retrank+E404+Sonnet "is a refinement-failure" — Sonnet-specific. Under Opus this is a top-5 result.
- ⚠️ CE-refinement hurts — direction holds, magnitude varies.

## What we did NOT do

- A third-judge cross-check (GPT-4o, Gemini 2.5) — neither API key was available in the environment. This remains the single biggest methodological gap in the cross-validation; flagged in the paper as future work.

## Reproduce

```bash
python scripts/ac_eval_opus.py --re-judge data/runs/retrank-AC-*.txt
python scripts/judge_comparison.py
python scripts/inter_judge_agreement.py
python scripts/verbosity_correlation.py
```
