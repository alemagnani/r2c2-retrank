#!/bin/bash
# Watchdog: checks explorer, restarts if dead or stuck on auth errors.

export ANTHROPIC_API_KEY="$(grep -o 'ANTHROPIC_API_KEY=.*' ~/.bashrc | head -1 | cut -d'"' -f2)"
cd /home/alessandro/workspace/r2c2-retrank

LOG=/home/alessandro/workspace/r2c2-retrank/data/watchdog.log
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

explorer_running=$(ps aux | grep "explore_pipelines.py" | grep -v grep | wc -l)

# Read explorer progress
n_done=$(python3 -c "import json; r=json.load(open('data/pipeline_exploration/results.json')); print(len(r))" 2>/dev/null || echo "?")
top1=$(python3 -c "import json; r=sorted(json.load(open('data/pipeline_exploration/results.json')),key=lambda x:-x['mean_ndcg']); print(f'{r[0][\"name\"]}={r[0][\"mean_ndcg\"]:.4f}')" 2>/dev/null || echo "?")

echo "[$TIMESTAMP] explorer=${explorer_running} | ${n_done}/100 best=${top1}" >> $LOG

# Check for auth errors in explorer
if [ "$explorer_running" -gt 0 ]; then
    auth_errors=$(tail -5 data/pipeline_exploration/explore_run5.log 2>/dev/null | grep -c "authentication")
    auth_errors=${auth_errors:-0}
    if [ "$auth_errors" -gt 2 ]; then
        echo "[$TIMESTAMP] WARNING: explorer has auth errors, restarting..." >> $LOG
        pkill -f "explore_pipelines.py"
        sleep 3
        explorer_running=0
    fi
fi

# Restart explorer if dead
if [ "$explorer_running" -eq 0 ]; then
    if python3 -c "import json; r=json.load(open('data/pipeline_exploration/results.json')); exit(0 if len(r)>=100 else 1)" 2>/dev/null; then
        echo "[$TIMESTAMP] Explorer finished normally — not restarting." >> $LOG
    else
        echo "[$TIMESTAMP] Explorer not running — restarting..." >> $LOG
        python3 -u scripts/explore_pipelines.py \
          --topics-file data/synthetic_val/synthetic_topics_val65.json \
          --qrels data/synthetic_autoresearch_val65/qrels_loop_claude_haiku_4_5_20251001.json \
          --loop-run data/synthetic_autoresearch_val65/final_run.txt \
          --output-dir data/pipeline_exploration \
          --max-configs 100 \
          --propose-every 5 \
          --seed 99 \
          --model claude-haiku-4-5-20251001 \
          >> data/pipeline_exploration/explore_run5.log 2>&1 &
        echo "[$TIMESTAMP] Explorer restarted PID=$!" >> $LOG
    fi
fi
