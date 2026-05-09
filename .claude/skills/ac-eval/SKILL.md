---
name: ac-eval
description: Evaluate a NTCIR-19 R2C2 AC (Answering with Confidence) run file using a Sonnet judge that mirrors the official scoring methodology. Reports Accuracy, Mean Nugget Precision, R_O, R_U, and HMR. Use this whenever the user asks to "score", "evaluate", "judge", or "benchmark" an AC run file (e.g. `retrank-AC-1.txt`), or to compare confidence-calibration strategies across multiple AC runs.
---

# ac-eval — R2C2 AC self-evaluation

## When to use

Invoke this skill in any of these situations:

- **Single AC run scoring**: user has a `*-AC-*.txt` file and wants metrics
- **Comparing confidence strategies**: user wants to compare multiple AC runs side by side
- **Pre-submission sanity check**: before zipping `retrank-AC.zip` for submission, run this on each of the 4 AC files
- **Replaying with cached judgments**: the user iterated on prompts and wants a fresh evaluation that re-uses the prior cache

## What it does

The skill wraps two scripts that together implement the official R2C2 AC scoring pipeline (BREV-RAG 2025 paper, §3.1):

1. **`scripts/ac_eval.py`** — runs Stage A (bogus nugget identification) + Stage B (answer evaluation) using a Sonnet judge, then computes the four metrics. Caches all LLM judgments in JSON.
2. **`scripts/calibration_bench.py`** — takes the JSON outputs and prints a leaderboard sorted by HMR.

## Standard invocation

For a single AC run:

```bash
python scripts/ac_eval.py \
  --ac-run data/runs/retrank-AC-1.txt \
  --pr-runs-dir data/raw/competitor_runs/PRruns \
  --topics data/raw/r2c2topics.txt \
  --output data/eval/ac/retrank-AC-1.json
```

To compare 4 AC runs:

```bash
for i in 1 2 3 4; do
  python scripts/ac_eval.py \
    --ac-run data/runs/retrank-AC-$i.txt \
    --pr-runs-dir data/raw/competitor_runs/PRruns \
    --topics data/raw/r2c2topics.txt \
    --output data/eval/ac/retrank-AC-$i.json
done
python scripts/calibration_bench.py --inputs data/eval/ac/retrank-AC-*.json
```

## Inputs

- `--ac-run`: file in the AC submission format (`<D...>` blocks). See `scripts/ac_format.py` for the parser.
- `--pr-runs-dir`: directory containing all PR run files cited by the AC nuggets. Must include both our own and any other team's runs whose passages the AC run cites. Default: `data/raw/competitor_runs/PRruns`.
- `--topics`: official R2C2 topics file (XML or JSON). Default: `data/raw/r2c2topics.txt`.
- `--output`: path for the per-question + metrics JSON.
- `--model`: judge model. Default `claude-sonnet-4-6`. Don't change without reason — the methodology assumes a strong judge.

## Outputs

A JSON file with this shape:

```json
{
  "metrics": {
    "accuracy": 0.74,
    "mean_nugget_precision": 0.62,
    "R_O": 0.41,
    "R_U": 0.78,
    "HMR": 0.54,
    "n_questions": 65
  },
  "per_question": {
    "0001": {
      "question": "...",
      "answer": "...",
      "confidence": 0.85,
      "n_returned": 4,
      "n_entailed": 3,
      "n_relevant": 2,
      "correct": true,
      "answer_reason": "...",
      "nuggets": [...]
    },
    ...
  }
}
```

Console summary table is printed at the end with the headline metrics.

## Cost guardrails

A full pass on one AC run × 65 questions × ~5 nuggets each ≈ ~330 LLM calls (Stage A) + 65 calls (Stage B). At Sonnet rates, roughly $4–6 per AC run. Re-runs are essentially free thanks to caching at `data/eval/ac_cache/`.

If the user asks for cheaper iteration, suggest running on a subset first:

```bash
# Truncate AC run to first 5 questions for a quick smoke test (~$0.30)
head -n 50 data/runs/retrank-AC-1.txt > /tmp/mini-AC-1.txt
python scripts/ac_eval.py --ac-run /tmp/mini-AC-1.txt ...
```

## What this skill does NOT do

- It does not produce AC outputs — that's a separate pipeline (extract nuggets + synthesize answer + estimate confidence). This skill only *evaluates* AC outputs.
- It does not replicate Step (6) of the official pipeline (PR re-evaluation using AC labels). That requires AC runs from all teams; we'll have it post-Aug 1.
- It does not include Brier or ECE — by design. The R2C2 task chose HMR specifically because Brier/ECE don't separate overconfidence from underconfidence.

## Validation

The math is unit-tested in `tests/test_hmr.py` (11 cases including paper edge examples). The end-to-end pipeline has an integration test in `tests/test_ac_eval_integration.py` that uses a mocked LLM and verifies metrics match hand-computed expected values.

Run all tests with:

```bash
python -m pytest tests/test_hmr.py tests/test_ac_format.py tests/test_ac_eval_integration.py -v
```
