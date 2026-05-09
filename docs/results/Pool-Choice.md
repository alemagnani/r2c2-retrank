[← Back to results index](README.md)

---

# Pool Choice — Quality of strongest component, not count

The naive read of our data is "less is more": a single-team pool beats a multi-team pool.
The defensible read is more nuanced.

## Single-team-pool comparison

| Single-team pool | Broad HMR | Refined-Sonnet HMR |
|---|---:|---:|
| **BITEM-only** | **0.937** | **0.963** |
| WaterlooClarke-only | 0.833 | 0.870 |
| Error404-only | 0.730 | 0.684 |

A high-quality single-team pool wins. A low-quality single-team pool
loses. **Count is not the explanatory variable; quality of the
strongest component is.**

## Multi-team-pool comparison (broad pool, no Stage 1.5)

| Pool | # teams | Acc | R_O | HMR |
|---|---:|---:|---:|---:|
| BITEM-only | 1 | 0.923 | 0.910 | 0.937 |
| retrank+Error404 | 2 | 0.923 | 0.870 | 0.910 |
| Tier-1 (BITEM+E404+Waterloo) | 3 | 0.923 | 0.730 | 0.826 |
| Tier-1+2 (6 teams) | 6 | 0.877 | 0.741 | 0.840 |

Read this carefully: the column "# teams" is *descriptive*, not the
explanatory variable. We retracted the strict-monotonicity claim
because:
- Bootstrap 95% CI on BITEM-vs-Tier-1 is `[-0.458, +0.044]` (crosses zero)
- Under the Opus judge, the ordering becomes `1 > 6 > 3` — Tier-1+2 (6 teams) jumps above Tier-1 (3 teams)

## Confounds we cannot rule out

The pools differ in *many* dimensions besides which teams contribute:

- Passage length (BITEM ~ 2,400 chars; retrank ~ 200 chars)
- Segmentation strategy (sliding-window vs paragraph vs hybrid)
- Underlying retrieval architecture (BM25, dense, hybrid, biencoder)
- Depth of supplied ranking
- Amount of near-duplication

A clean "diversity" isolation experiment would require **fixed-budget pooling** (equal passages from each pool, controlled segmentation, controlled rank depth). We flag that as future work; with the current data we cannot attribute the effect to *diversity* per se.

## Why does pooling sometimes hurt? One mechanism

Stage 2 nugget extraction sees a noisier input context when the pool
mixes segmentation styles. The candidate-answer-targeted prompt
becomes harder to satisfy with passages of mixed quality. The end
result is more Cnf-W topics → lower R_O → lower HMR. See
[The R_O Bottleneck](The-RO-Bottleneck.md).

## Practitioner takeaway

If you can identify a high-quality PG-route team's run, prefer it as
a standalone pool. If not, a multi-team pool is a safer default than
gambling on a random single-team pool. The right axis is **quality of
the best component**, not count.

## Reproduce
```bash
# our 5 pool variants
ls data/processed/ac_pool_*.json

# eval per pool
python scripts/refusal_decomposition.py
```
