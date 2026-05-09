#!/bin/bash
# Hybrid LLM ablation: S1=Sonnet, S1.5=Haiku, S2=Sonnet, S3=Sonnet, S4=Haiku.
# Tests the cost-quality recipe in §RQ10 of the paper.
set -e
cd /home/alessandro/workspace/r2c2-retrank

POOL=bitem_only
TAG=${POOL}_hybrid_SHSH
HAIKU=claude-haiku-4-5-20251001
SONNET=claude-sonnet-4-6

# Stage 1 (Sonnet) — reuse existing ac_stage1_bitem_only.json (already cached Sonnet)
# Stage 1.5 with Haiku
python scripts/ac_stage15_refine.py \
    --pool data/processed/ac_pool_${POOL}.json \
    --stage1 data/processed/ac_stage1_${POOL}.json \
    --algo sonnet --model "$HAIKU" \
    --output data/processed/ac_pool_${TAG}_refined.json 2>&1 | tail -3

# Stage 2 (Sonnet, default)
python scripts/ac_stage2_nuggets.py \
    --pool data/processed/ac_pool_${TAG}_refined.json \
    --stage1 data/processed/ac_stage1_${POOL}.json \
    --model "$SONNET" \
    --output data/processed/ac_stage2_${TAG}.json 2>&1 | tail -3

# Stage 3 (Sonnet bogus filter — paper holds Stage 3 = Sonnet for all variants)
python scripts/ac_stage3_filter.py \
    --stage2 data/processed/ac_stage2_${TAG}.json \
    --model "$SONNET" \
    --output data/processed/ac_stage3_${TAG}.json 2>&1 | tail -3

# Stage 4 with Haiku (the cheap-verifier substitution)
python scripts/ac_stage4_verify.py \
    --stage3 data/processed/ac_stage3_${TAG}.json \
    --model "$HAIKU" \
    --output data/processed/ac_stage4_${TAG}.json 2>&1 | tail -3

# Stage 5 (no LLM)
python scripts/ac_stage5_calibrate.py \
    --stage4 data/processed/ac_stage4_${TAG}.json \
    --variant A \
    --output data/processed/ac_stage5_${TAG}_A.json 2>&1 | tail -3

# Stage 6 (no LLM)
python scripts/ac_stage6_format.py \
    --stage5 data/processed/ac_stage5_${TAG}_A.json \
    --output data/runs/retrank-AC-${TAG}-A.txt 2>&1 | tail -3

# Eval
python scripts/ac_eval.py \
    --ac-run data/runs/retrank-AC-${TAG}-A.txt \
    --output data/eval/ac_runs/${TAG}_A.json 2>&1 | grep -E "Accuracy|Mean Nugget|R_O|R_U|HMR|N questions"

echo "═══ DONE ═══"
