[← Back to results index](README.md)

---

# Pipeline Overview — 7-stage answer-first design

Our pipeline is **answer-first with verification**: a single candidate
answer is proposed early, then used to focus subsequent nugget extraction
and confidence estimation.

## Stages

| # | Stage | Model | Function |
|---|---|---|---|
| 0 | Pool selection | CE (`ms-marco-MiniLM-L-12-v2`) | Aggregate top-K passages from selected teams' PR runs; dedupe; rank by CE; cap at 50 |
| 1 | Candidate answer | Sonnet 4.6 | Read pool, emit `(answer, c_self)` |
| 1.5 | Answer-conditional refinement | Sonnet pointwise (or CE / lexical) | Re-rank pool conditional on candidate answer; keep top 30 |
| 2 | Targeted nugget extraction | Sonnet 4.6 | Emit 3–8 atomic claims supporting the candidate, each with a single-passage citation |
| 3 | Self Stage-A bogus filter | Sonnet 4.6 | Run entailment check on every (passage, nugget) pair; drop bogus |
| 4 | Verification | Sonnet 4.6 | Derive answer using *only* entailed nuggets; produce `match_score ∈ {0,1,2}` |
| 5 | Confidence & refusal | (no LLM) | Combine `c_self`, `match_score`, refusal logic into final confidence |
| 6 | Format | (no LLM) | Emit official AC XML |

## Why answer-first?

A direct head-to-head against nugget-first on 20 topics shows similar accuracy but **−0.29 HMR** for nugget-first, almost entirely from R_O. See [Nugget-First vs Answer-First](Nugget-First-vs-Answer-First.md).

Mechanism: answer-first knows what to look for and can recognise when
the pool does not contain it; nugget-first confidently generalises
from incomplete evidence.

## Self-evaluator (`scripts/ac_eval.py`)

Mirrors the official two-stage scoring locally:

- **Stage A (bogus identification):** for each (passage, nugget), is it entailed?
- **Stage B (answer evaluation):** given question + answer + entailed nuggets, is the answer correct? which nuggets helped derive it?

Default judge: Sonnet 4.6. Cross-validation judge: Opus 4.7.
SHA256-keyed deterministic caches; resumable.

## File map

```
scripts/
  ac_stage0_pool.py          # build per-pool pool JSONs
  ac_stage1_candidate.py     # candidate answer
  ac_stage15_refine.py       # answer-conditional re-rank
  ac_stage2_nuggets.py       # targeted nugget extraction
  ac_stage3_filter.py        # bogus filter
  ac_stage4_verify.py        # independent verifier
  ac_stage5_calibrate.py     # confidence + refusal
  ac_stage6_format.py        # AC XML output
  ac_eval.py                 # two-stage self-evaluator
  ac_ensemble.py             # cross-pool aggregation
src/
  eval/hmr.py                # HMR / R_O / R_U math (unit-tested)
data/
  runs/                      # AC submission text files + retrank-AC.zip
  eval/ac_runs/              # per-candidate self-eval JSON
  processed/                 # caches (gitignored — regenerable)
```

See [Reproducing the Pipeline](Reproducing-the-Pipeline.md) for end-to-end commands.
