[← Back to results index](README.md)

---

# Final Submission — the four AC runs

`data/runs/retrank-AC.zip` (96 KB, 4 files, all lint-clean).

## The four runs

| File | Configuration | HMR (S) | HMR (O) | Role |
|---|---|---:|---:|---|
| **retrank-AC-1.txt** | `ensemble_top1` | 0.974 | 0.974 | Headline: post-hoc confidence saturation of the main pipeline |
| **retrank-AC-2.txt** | `bitem_only_refined_sonnet` | 0.963 | 0.963 | **Main system** — clean single-pool baseline |
| **retrank-AC-3.txt** | `bitem_only_opus` | 0.787 | 0.792 | Diversity hedge (Opus pipeline at Stage 1) |
| **retrank-AC-4.txt** | `refined_sonnet` (Tier-1) | 0.911 | 0.818 | Different pool entirely — insurance against grounding-diversity rewards |

## Why these four?

**AC-1** is our best self-eval HMR — we submit it though we are
clear in the paper that the +0.011 over AC-2 is post-hoc re-calibration.

**AC-2** is the system the rest of the paper analyses: clean,
reproducible, simplest pipeline with the strongest evidence behind
it. **This is the one we'd cite if we had to pick one.**

**AC-3** is methodologically different from the other three (Opus
Stage 1 emits terser answers). We submit it as a diversity hedge
even though our judges penalise it; if the official judge is more
lenient on terse grounding, this could surprise.

**AC-4** uses a different *pool* (Tier-1 = BITEM + Error404 +
WaterlooClarke). All other AC runs source passages from BITEM-only.
If the official judge rewards grounding diversity that BITEM-only
cannot reach, this is our hedge against that.

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
