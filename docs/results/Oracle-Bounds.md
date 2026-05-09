[← Back to results index](README.md)

---

# Oracle Upper Bounds

What's the most an HMR-optimising pipeline could possibly score on the BITEM-only pool, holding the underlying retrieval and answer set fixed?

## Anchored to BITEM-Sonnet (main system)

| System | Acc | R_O | R_U | HMR |
|---|---:|---:|---:|---:|
| **BITEM-Sonnet (submitted)** | 0.938 | 0.950 | 0.976 | **0.963** |
| Oracle abstention (refuse exactly the topics we get wrong) | 0.938 | 0.950 | 0.976 | **0.963** |
| Oracle calibration (wrong @ conf=0.05) | 0.938 | 0.950 | 1.000 | 0.974 |
| Oracle calibration (wrong @ conf=0.0) | 0.938 | 1.000 | 1.000 | 1.000 |
| Oracle answer (every topic correct @ 0.95) | 1.000 | 1.000 | 0.950 | 0.974 |
| Theoretical max (all correct @ 1.0) | 1.000 | 1.000 | 1.000 | 1.000 |

## The headline finding

**Our submitted system is already at the oracle-abstention upper bound.**

Because Cnf-W (confident-wrong) is already 0, perfect refusal of our wrong topics cannot add anything. The system already refuses (or correctly answers) the topics that would otherwise damage R_O. This corroborates the [R_O bottleneck thesis](The-RO-Bottleneck.md) operationally: the design choices that move the metric have already been spent on R_O.

## What's left as headroom

The remaining 0.037 gap to HMR=1.0 lives in two places:

1. **R_U** (currently 0.976). We are slightly under-confident on correct answers (some correct topics carry conf < 1.0). Bringing R_U → 1.0 alone would lift HMR to 0.974. The "confidence-saturation recalibration" submission (a.k.a. "ensemble_top1") essentially does this — see [Cross-Pool Aggregation](Cross-Pool-Aggregation.md).
2. **Accuracy** (currently 0.938). Four topics are refused. Three are genuinely unanswerable from the BITEM-only pool; one is a JSON parse failure that we deliberately left as-is because attempts to recover it [hurt HMR](../paper_draft.pdf) (see "Negative Results: Three Improvements That Hurt").

## Implication for the engineering frontier

We can't out-engineer the abstention oracle on BITEM-only. The genuinely promising directions for further HMR uplift on this pool are:

- **Better R_U** (already largely captured by post-hoc saturation, +0.011 HMR).
- **Different pool entirely** to recover the 4 refused topics (but our [Tier-1 experiments](Pool-Choice.md) show pooling-based recovery introduces new Cnf-W).
- **Better understanding of what topics are "genuinely unanswerable"** — calibration on the meta-question of refusal itself.

## Reproduce

```bash
python scripts/oracle_upper_bounds.py
```

Source: `data/eval/ac_runs/oracle_upper_bounds.json`.
