[← Back to results index](README.md)

---

# Final Submission — the four AC runs

`data/runs/retrank-AC.zip` (96 KB, 4 files, all lint-clean).

## The four runs

| File | Configuration | HMR (S) | HMR (O) | Acc | Role |
|---|---|---:|---:|---:|---|
| **retrank-AC-1.txt** | `ensemble_top1` | 0.974 | 0.974 | 0.938 | Headline: confidence-saturation recalibration of the main pipeline |
| **retrank-AC-2.txt** | `bitem_only_refined_sonnet` | 0.963 | 0.963 | 0.938 | **Main system** — clean single-pool baseline |
| **retrank-AC-3.txt** | `bitem_only_refined_sonnet_recovered` | 0.922 | **0.951** | **0.954** | Accuracy-favouring hedge — highest accuracy of any candidate |
| **retrank-AC-4.txt** | `tier1plus2_refined_sonnet` (6-team pool) | 0.874 | **0.937** | 0.877 | Judge-divergent hedge — rank 4 under Opus; different pool |

## Why these four?

**AC-1** is our best self-eval HMR — we submit it though we are
clear in the paper that the +0.011 over AC-2 is post-hoc re-calibration.

**AC-2** is the system the rest of the paper analyses: clean,
reproducible, simplest pipeline with the strongest evidence behind
it. **This is the one we'd cite if we had to pick one.**

**AC-3** is the same pipeline as AC-2 but with the JSON-parse refusal-recovery
prompt fix from the negative-results section of the paper. It was originally
classed as a Sonnet-judge negative result (HMR dropped 0.963 → 0.922), but
under the Opus judge it is **rank 4 (0.951)** and it has the **highest accuracy
of any candidate (0.954)** with two refusals instead of four. If the official
ranking weights Accuracy or MNP heavily, this candidate is genuinely
competitive with our main system, not just a hedge.

**AC-4** uses a different *pool* entirely: a six-team pool (BITEM, Error404,
WaterlooClarke, ORG, hit-u, WasedaR2C2). It has the largest judge-positive
swing of any submission candidate: Sonnet 0.874 → Opus 0.937. If the
official judge is closer in temperament to Opus 4.7 than to Sonnet 4.6,
this candidate jumps from middle-of-pack to rank 4. Different methodology
(6 teams pooled) from the BITEM-only basis of AC-1, AC-2, AC-3.

We chose this set rather than an "all top-3 by self-eval" bundle
because the top-3 share too much state (essentially the same
answers, same nuggets, different confidence rules) — they're not
independent bets.

## Lint pass

All four files pass `scripts/lint_ac_run.py`:

- 65 `<Dxxxx>...</Dxxxx>` blocks each
- All `(rank, run-name)` tuples valid
- All confidence values in [0, 100]
- No malformed lines
- Refusal cells re-coded as "I don't know" before packaging (in case the official scorer rejects empty answers)

## Submission

Email `data/runs/retrank-AC.zip` to **`ntcir19r2c2org@list.waseda.jp`** before **May 15, 2026**.

## Reproduce

```bash
# Lint:
python scripts/lint_ac_run.py data/runs/retrank-AC-*.txt

# Re-package:
cd /tmp/ac_pkg
zip retrank-AC.zip retrank-AC-*.txt
```
