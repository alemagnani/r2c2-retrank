[← Back to results index](README.md)

---

# Cross-Pool Aggregation — when ensembling helps and what it really is

## The result

We aggregated answers from all 16 base candidate runs per topic,
clustered by Haiku-judged semantic equivalence, and used majority
voting weighted by self-eval HMR.

| K (top-K voters) | HMR |
|---:|---:|
| **1** | **0.974** |
| 2 | 0.917 |
| **3** | **0.969** |
| 4 | 0.873 |
| 5 | 0.857 |
| 12 | 0.918 |
| 16 (all) | 0.857 |

The curve is **non-monotonic**. Naive 16-way ensembling is *worse*
than the best single pipeline because weak voters drag down consensus
quality. Top-3 (BITEM-Sonnet + BITEM-broad + Tier-1-refined) is a
genuine ensemble that beats the baseline. Top-2 dips because the
second voter (BITEM-broad) is highly correlated with top-1.

## What "top-1 ensemble" actually is

**It is not an ensemble.** A single voter trivially agrees with itself,
so the ensemble script saturates confidence at 1.00 on every non-refused
topic. This is **post-hoc confidence saturation** of the best single
pipeline — the underlying answers and nuggets are identical to
BITEM-only-refined-Sonnet's. The HMR uplift (+0.011) comes purely from
re-calibration; there is no new information.

We submit it (as `retrank-AC-1`) and label it correctly throughout
the paper.

## Synthesis: passage-level pooling vs answer-level aggregation

| Level | Effect | Why |
|---|---|---|
| Passage-level pooling (RQ on [Pool Choice](Pool-Choice.md)) | mostly hurts | Stage 2 sees noisier input; mixed segmentation |
| Answer-level aggregation (this page) | helps with high-quality voters only | each voter ran on a *coherent* pool and produced a *coherent* answer; aggregation is at the output, no input contamination |

The contrast is not contradictory: ensembling **outcomes** avoids the
contamination that ensembling **inputs** causes.

## Caveats

- Bootstrap CI on top-1 vs BITEM-Sonnet baseline: [+0.008, +0.016] — significant but tiny.
- Bootstrap CI on top-3 vs baseline: [-0.002, +0.014] — within noise.
- All voters are our own pipelines. A true cross-team ensemble is left as future work.

## Reproduce

```bash
python scripts/ac_ensemble.py        # produces ensemble_top* and ensemble_all
python scripts/ac_eval.py --ac-run data/runs/retrank-AC-ensemble-top1.txt --output data/eval/ac_runs/ensemble_top1.json
```
