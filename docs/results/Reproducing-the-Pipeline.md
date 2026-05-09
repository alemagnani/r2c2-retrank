[← Back to results index](README.md)

---

# Reproducing the Pipeline

End-to-end commands to regenerate the submission and main results from scratch.

## Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt    # anthropic, sentence-transformers, bm25s, faiss-cpu, ...
export ANTHROPIC_API_KEY=sk-ant-...
```

Estimated cost end-to-end: **~$73 in API calls** for one pipeline pass over the 16 base configurations + ensembles + judge cross-validation. Caches are SHA256-keyed and resumable, so subsequent runs are essentially free.

## 1. Build pools

The pre-built pools are in `data/processed/`. To regenerate from raw PR runs (in `data/raw/competitor_runs/PRruns/` — gitignored, downloaded post-deadline):

```bash
python scripts/ac_stage0_pool.py --teams BITEM \
    --output data/processed/ac_pool_bitem_only.json
python scripts/ac_stage0_pool.py --teams BITEM Error404 WaterlooClarke \
    --output data/processed/ac_pool_tier1.json
# ... etc for tier1plus2, retrank-only, retrank+Error404
```

## 2. Run main pipeline (BITEM-only refined-Sonnet)

```bash
POOL=bitem_only

# Stage 1: candidate answer
python scripts/ac_stage1_candidate.py \
    --pool data/processed/ac_pool_${POOL}.json \
    --output data/processed/ac_stage1_${POOL}.json

# Stage 1.5: answer-conditional refinement
python scripts/ac_stage15_refine.py \
    --pool data/processed/ac_pool_${POOL}.json \
    --stage1 data/processed/ac_stage1_${POOL}.json \
    --algo sonnet \
    --output data/processed/ac_pool_${POOL}_refined_sonnet.json

# Stage 2: targeted nugget extraction
python scripts/ac_stage2_nuggets.py \
    --pool data/processed/ac_pool_${POOL}_refined_sonnet.json \
    --stage1 data/processed/ac_stage1_${POOL}.json \
    --output data/processed/ac_stage2_${POOL}_refined_sonnet.json

# Stage 3: bogus filter
python scripts/ac_stage3_filter.py \
    --stage2 data/processed/ac_stage2_${POOL}_refined_sonnet.json \
    --output data/processed/ac_stage3_${POOL}_refined_sonnet.json

# Stage 4: verifier
python scripts/ac_stage4_verify.py \
    --stage3 data/processed/ac_stage3_${POOL}_refined_sonnet.json \
    --output data/processed/ac_stage4_${POOL}_refined_sonnet.json

# Stage 5: calibrate
python scripts/ac_stage5_calibrate.py \
    --stage4 data/processed/ac_stage4_${POOL}_refined_sonnet.json \
    --variant A \
    --output data/processed/ac_stage5_${POOL}_refined_sonnet_A.json

# Stage 6: format
python scripts/ac_stage6_format.py \
    --stage5 data/processed/ac_stage5_${POOL}_refined_sonnet_A.json \
    --output data/runs/retrank-AC-${POOL}_refined_sonnet-A.txt

# Self-evaluate
python scripts/ac_eval.py \
    --ac-run data/runs/retrank-AC-${POOL}_refined_sonnet-A.txt \
    --output data/eval/ac_runs/${POOL}_refined_sonnet_A.json
```

Expected: HMR 0.963, accuracy 0.94, refusal 6%, Cnf-W 0.

## 3. Single-team-pool ablation

```bash
bash scripts/run_single_team_pools.sh
```

## 4. Hybrid Sonnet/Haiku

```bash
bash scripts/run_hybrid_S_H_S_H.sh
```

## 5. Cross-pool ensembles

```bash
python scripts/ac_ensemble.py
```

## 6. Judge cross-validation

```bash
python scripts/ac_eval_opus.py --re-judge data/runs/retrank-AC-*.txt
python scripts/judge_comparison.py
python scripts/inter_judge_agreement.py
```

## 7. Statistical analyses

```bash
python scripts/refusal_decomposition.py
python scripts/calibration_bins.py
python scripts/bootstrap_ci.py
python scripts/mcnemar_tests.py
python scripts/trivial_baselines.py
python scripts/verbosity_correlation.py
python scripts/ro_bottleneck_figure.py
python scripts/pr_reflexive_eval.py
```

## 8. Mini nugget-first head-to-head

```bash
python scripts/nugget_first_mini.py
```

## 9. Package and lint final submission

```bash
mkdir -p /tmp/ac_pkg
cp data/runs/retrank-AC-ensemble-top1.txt                /tmp/ac_pkg/retrank-AC-1.txt
cp data/runs/retrank-AC-bitem_only_refined_sonnet-A.txt  /tmp/ac_pkg/retrank-AC-2.txt
cp data/runs/retrank-AC-bitem_only_opus-A.txt            /tmp/ac_pkg/retrank-AC-3.txt
cp data/runs/retrank-AC-refined_sonnet-A.txt             /tmp/ac_pkg/retrank-AC-4.txt
python scripts/fix_empty_answers.py /tmp/ac_pkg/retrank-AC-*.txt
python scripts/lint_ac_run.py /tmp/ac_pkg/retrank-AC-*.txt
cd /tmp/ac_pkg && zip -r ../retrank-AC.zip retrank-AC-*.txt
```

## 10. Compile paper

```bash
cd docs
pdflatex paper_draft.tex
bibtex paper_draft
pdflatex paper_draft.tex
pdflatex paper_draft.tex
```

## Cache hygiene

Every Sonnet/Opus call is cached in `data/processed/*.cache.json` keyed by SHA256 of the prompt. To force re-computation on one stage, just delete that stage's cache file. Do **not** invalidate the entire cache — it represents ~$73 of API calls.
