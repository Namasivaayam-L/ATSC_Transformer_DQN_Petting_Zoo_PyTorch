#!/usr/bin/env bash
# run_ep_count.sh — trf_coord episode count convergence study.
# 10 combos x 5 ep_counts x 5 seeds = 250 runs.
# SUMO port conflicts prevent parallelism — run sequentially.
# Safe to kill and resume at any time.
set -euo pipefail

PYTHON=".venv/bin/python"
OUTDIR="results_ep_count"
LOGDIR="$OUTDIR/_logs"
mkdir -p "$LOGDIR"

EP_COUNTS=(50 100 200 400 500)
COMBOS=()
for eps in "${EP_COUNTS[@]}"; do
  COMBOS+=("trf_coord grid4x4 dwt $eps")
  COMBOS+=("trf_coord_equal grid4x4 dwt $eps")
  COMBOS+=("trf_coord cologne3 dwt $eps")
  COMBOS+=("trf_coord_equal cologne3 dwt $eps")
  COMBOS+=("trf_coord cologne8 dwt $eps")
  COMBOS+=("trf_coord_equal cologne8 dwt $eps")
  COMBOS+=("trf_coord ingolstadt7 dwt $eps")
  COMBOS+=("trf_coord_equal ingolstadt7 dwt $eps")
  COMBOS+=("trf_coord ingolstadt21 dwt $eps")
  COMBOS+=("trf_coord_equal ingolstadt21 dwt $eps")
done

TOTAL=${#COMBOS[@]}
echo "=== Episode count study: $TOTAL runs x 5 seeds ==="
echo "Start: $(date)"
echo ""

FAILED=0
COMPLETED=0
GLOBAL_T0=$SECONDS

for combo in "${COMBOS[@]}"; do
  read -r agent env reward eps <<< "$combo"
  tag="${agent}_${env}_${reward}_eps${eps}"
  rdir="$OUTDIR/$tag"
  logfile="$LOGDIR/${tag}.log"

  # Skip if aggregate.json already exists with 5 seeds
  if [ -f "$rdir/aggregate.json" ]; then
    nseeds=$(python3 -c "
import json; d=json.load(open('$rdir/aggregate.json'));
print(len(d.get('per_seed',[])))" 2>/dev/null || echo 0)
    if [ "$nseeds" -ge 5 ]; then
      echo "[SKIP] $tag — $nseeds seeds"
      COMPLETED=$((COMPLETED + 1))
      continue
    fi
  fi

  echo "[RUN $((COMPLETED+1))/$TOTAL] $tag — $(date '+%H:%M:%S')"
  t0=$SECONDS

  CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    agent="$agent" env="$env" reward="$reward" \
    seeds=5 num_episodes=$eps device=cuda resume=true \
    run_dir="$rdir" \
    > "$logfile" 2>&1

  rc=$?
  elapsed=$(( SECONDS - t0 ))
  if [ $rc -eq 0 ]; then
    echo "  [DONE] ${elapsed}s"
    COMPLETED=$((COMPLETED + 1))
  else
    echo "  [FAIL] rc=$rc — see $logfile"
    FAILED=$((FAILED + 1))
    tail -3 "$logfile" 2>/dev/null
  fi
done

total_elapsed=$(( SECONDS - GLOBAL_T0 ))
echo ""
echo "=== Done: $COMPLETED/$TOTAL completed, $FAILED failed, ${total_elapsed}s ==="
echo "End: $(date)"

# Verification
echo ""
echo "Verifying aggregates:"
GOOD=0
for combo in "${COMBOS[@]}"; do
  read -r agent env reward eps <<< "$combo"
  rdir="$OUTDIR/${agent}_${env}_${reward}_eps${eps}"
  if [ -f "$rdir/aggregate.json" ]; then
    nseeds=$(python3 -c "import json; print(len(json.load(open('$rdir/aggregate.json')).get('per_seed',[])))" 2>/dev/null || echo 0)
    echo "  ${agent}_${env}_eps${eps} — $nseeds seeds"
    GOOD=$((GOOD + 1))
  else
    echo "  ${agent}_${env}_eps${eps} — MISSING"
  fi
done
echo "$GOOD/$TOTAL aggregates present."
