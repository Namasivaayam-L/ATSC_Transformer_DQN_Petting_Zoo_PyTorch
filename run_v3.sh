#!/usr/bin/env bash
# results_v3: cologne8 (8 agents) + cologne3, 3600s episodes, 20 eps, 5 seeds.
# Tests transformer coordination on larger topology with longer horizons.
set -euo pipefail

PYTHON=".venv/bin/python"
OUTDIR="results_v3"
LOGDIR="results_v3/_logs"
mkdir -p "$LOGDIR"

COMBOS=(
  # cologne8 baselines
  "fixed_time cologne8 dwt"
  "max_pressure cologne8 dwt"
  "mplight cologne8 dwt"
  # cologne8 RL agents × 3 rewards
  "idqn cologne8 dwt"
  "idqn cologne8 pressure"
  "idqn cologne8 queue"
  "trf_coord cologne8 dwt"
  "trf_coord cologne8 pressure"
  "trf_coord cologne8 queue"
  "trf_coord_equal cologne8 dwt"
  "trf_coord_equal cologne8 pressure"
  "trf_coord_equal cologne8 queue"
  # cologne3 baselines (longer episodes)
  "fixed_time cologne3 dwt"
  "max_pressure cologne3 dwt"
  "mplight cologne3 dwt"
  # cologne3 RL agents
  "idqn cologne3 dwt"
  "trf_coord cologne3 dwt"
  "trf_coord_equal cologne3 dwt"
)

TOTAL=${#COMBOS[@]}
echo "=== results_v3: $TOTAL combos × 5 seeds, cologne8 + cologne3, 3600s ==="
echo "Start: $(date)"

FAILED=0
COMPLETED=0
GLOBAL_T0=$SECONDS

for combo in "${COMBOS[@]}"; do
  read -r agent env reward <<< "$combo"
  tag="${agent}_${env}_${reward}"
  rdir="$OUTDIR/$tag"
  logfile="$LOGDIR/${tag}.log"

  # Skip if done
  if [ -f "$rdir/aggregate.json" ]; then
    nseeds=$(python3 -c "import json; print(len(json.load(open('$rdir/aggregate.json')).get('per_seed',[])))" 2>/dev/null || echo 0)
    if [ "$nseeds" -ge 5 ]; then
      echo "[SKIP] $tag — $nseeds seeds"
      COMPLETED=$((COMPLETED + 1))
      continue
    fi
  fi

  echo "[RUN] $tag — $(date '+%H:%M:%S') [$((COMPLETED+1))/$TOTAL]"
  t0=$SECONDS

  CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    agent="$agent" env="$env" reward="$reward" \
    seeds=5 num_episodes=20 num_seconds=3600 device=cuda resume=true \
    run_dir="$rdir" \
    > "$logfile" 2>&1

  rc=$?
  elapsed=$(( SECONDS - t0 ))
  if [ $rc -eq 0 ]; then
    echo "[DONE] $tag — ${elapsed}s"
    COMPLETED=$((COMPLETED + 1))
  else
    echo "[FAIL] $tag — rc=$rc (${elapsed}s)"
    FAILED=$((FAILED + 1))
    tail -3 "$logfile" 2>/dev/null
  fi
done

total_elapsed=$(( SECONDS - GLOBAL_T0 ))
echo ""
echo "=== Batch complete: $COMPLETED/$TOTAL done, $FAILED failed, ${total_elapsed}s ==="
echo "End: $(date)"
