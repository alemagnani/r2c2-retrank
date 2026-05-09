[← Back to results index](README.md)

---

# Cost vs Quality — per-stage Sonnet/Haiku ablation

Where does Sonnet-class reasoning matter? Where can Haiku replace it?

## Per-stage substitution (BITEM-only, single substitution per row)

| Variant | S1 | S1.5 | S2 | S4 | Acc | R_O | HMR | Δ HMR |
|---|:-:|:-:|:-:|:-:|---:|---:|---:|---:|
| **all-Sonnet (baseline)** | S | S | S | S | 0.938 | 0.950 | **0.963** | — |
| s4-Haiku | S | S | S | H | 0.923 | 0.870 | 0.913 | −0.050 |
| s2-Haiku | S | S | H | S | 0.908 | 0.850 | 0.899 | −0.064 |
| s1-Haiku | H | S | S | S | 0.892 | 0.750 | 0.836 | **−0.127** |
| all-Haiku | H | H | H | H | 0.877 | 0.708 | 0.792 | −0.171 |

**Stage 1 (candidate answer) is the most Sonnet-critical** stage.
Stage 4 (verifier) is the least. The mechanism: Stage 1 reads 50 long
passages and synthesises a single confident answer (most reasoning-
intensive); Stage 4 receives a short curated nugget list and asks
"what answer do these support?" (much narrower task).

## Hybrid configuration (measured, not extrapolated)

We tested the specific hybrid `S1=S, S1.5=H, S2=S, S3=S, S4=H`:

| Configuration | Acc | R_O | HMR | Δ vs all-Sonnet |
|---|---:|---:|---:|---:|
| all-Sonnet | 0.938 | 0.950 | 0.963 | — |
| `S/H/S/S/H` (hybrid) | 0.923 | **0.634** | **0.765** | **−0.198** |

**The additive extrapolation fails badly.** Naively summing the
per-stage drops would predict Δ ≈ −0.08; the measured drop is −0.20.
**Per-stage Haiku substitutions interact non-additively** because
Stage 1.5 demotes the same grounding-fact passages that Stage 4's
Haiku verifier then fails to recover from — two losses on the same
topics.

## Recommended cost-saving substitution

Stage 4 alone (`S/S/S/H`): **−0.05 HMR for a genuinely cheap verifier**. This
is the only single-stage substitution we recommend without reservation.

## Reproduce

```bash
# Single-stage Haiku ablations (already cached):
ls data/processed/ac_stage1_bitem_only_s*.json

# Hybrid run:
bash scripts/run_hybrid_S_H_S_H.sh
```

## Why the additive fail matters for cascade design

Cost-quality cascades in the LLM literature often assume per-stage
quality is independently composable. **At our operating point this
fails** because the same hard topics are sensitive to *multiple*
upstream and downstream stages, and reductions compound on those topics.
The bootstrap CI on the all-Haiku Δ (−0.170, [-0.334, -0.080]) excludes
zero; the per-stage substitutions also cross statistical detectability
at this sample size — see [Statistical Robustness](Statistical-Robustness.md).
