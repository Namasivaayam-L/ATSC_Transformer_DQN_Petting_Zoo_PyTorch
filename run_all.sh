#!/usr/bin/env bash
# Run all 24 experiment combos × 5 seeds sequentially.
# SUMO port conflicts prevent parallelism — run one combo at a time.
# Results stored in results_v2/ with clear naming.
# Usage: bash run_all.sh
set -euo pipefail

PYTHON=".venv/bin/python"
OUTDIR="results_v2"
LOGDIR="results_v2/_logs"
mkdir -p "$LOGDIR"

COMBOS=(
  "fixed_time grid4x4 dwt"
  "max_pressure grid4x4 dwt"
  "mplight grid4x4 dwt"
  "idqn grid4x4 dwt"
  "idqn grid4x4 pressure"
  "idqn grid4x4 queue"
  "trf_coord grid4x4 dwt"
  "trf_coord grid4x4 pressure"
  "trf_coord grid4x4 queue"
  "trf_coord_equal grid4x4 dwt"
  "trf_coord_equal grid4x4 pressure"
  "trf_coord_equal grid4x4 queue"
  "fixed_time cologne3 dwt"
  "max_pressure cologne3 dwt"
  "mplight cologne3 dwt"
  "idqn cologne3 dwt"
  "idqn cologne3 pressure"
  "idqn cologne3 queue"
  "trf_coord cologne3 dwt"
  "trf_coord cologne3 pressure"
  "trf_coord cologne3 queue"
  "trf_coord_equal cologne3 dwt"
  "trf_coord_equal cologne3 pressure"
  "trf_coord_equal cologne3 queue"
)

TOTAL=${#COMBOS[@]}
echo "=== Running $TOTAL combos × 5 seeds (sequential) ==="
echo "Start: $(date)"
echo ""

FAILED=0
COMPLETED=0
GLOBAL_T0=$SECONDS

for combo in "${COMBOS[@]}"; do
  read -r agent env reward <<< "$combo"
  tag="${agent}_${env}_${reward}"
  rdir="$OUTDIR/$tag"
  logfile="$LOGDIR/${tag}.log"

  # Skip if already complete
  if [ -f "$rdir/aggregate.json" ]; then
    nseeds=$(python3 -c "import json; print(len(json.load(open('$rdir/aggregate.json')).get('per_seed',[])))" 2>/dev/null || echo 0)
    if [ "$nseeds" -ge 5 ]; then
      echo "[SKIP] $tag — $nseeds seeds done"
      COMPLETED=$((COMPLETED + 1))
      continue
    fi
  fi

  echo "[RUN] $tag — $(date '+%H:%M:%S') [$((COMPLETED+1))/$TOTAL]"
  t0=$SECONDS

  # 20 episodes for speed; RL learning curves still meaningful
  # resume=true: picks up from latest.pt if it exists
  CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    agent="$agent" env="$env" reward="$reward" \
    seeds=5 num_episodes=20 device=cuda resume=true \
    run_dir="$rdir" \
    > "$logfile" 2>&1

  rc=$?
  elapsed=$(( SECONDS - t0 ))
  if [ $rc -eq 0 ]; then
    echo "[DONE] $tag — ${elapsed}s"
    COMPLETED=$((COMPLETED + 1))
  else
    echo "[FAIL] $tag — rc=$rc (${elapsed}s) — see $logfile"
    FAILED=$((FAILED + 1))
    # Print last error line
    tail -3 "$logfile" 2>/dev/null
  fi
done

total_elapsed=$(( SECONDS - GLOBAL_T0 ))
echo ""
echo "=== Batch complete: $COMPLETED/$TOTAL done, $FAILED failed, ${total_elapsed}s total ==="
echo "End: $(date)"

# Verify
echo ""
echo "Verifying aggregates:"
GOOD=0
for combo in "${COMBOS[@]}"; do
  read -r agent env reward <<< "$combo"
  rdir="$OUTDIR/${agent}_${env}_${reward}"
  if [ -f "$rdir/aggregate.json" ]; then
    nseeds=$(python3 -c "import json; print(len(json.load(open('$rdir/aggregate.json')).get('per_seed',[])))" 2>/dev/null || echo 0)
    echo "  ✓ ${agent}_${env}_${reward} — $nseeds seeds"
    GOOD=$((GOOD + 1))
  else
    echo "  ✗ ${agent}_${env}_${reward} — MISSING"
  fi
done
echo "$GOOD/$TOTAL aggregates present."
