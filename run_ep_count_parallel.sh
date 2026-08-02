#!/usr/bin/env bash
# run_ep_count_parallel.sh — trf_coord episode count convergence study.
# 10 combos x 5 ep_counts x 5 seeds = 250 runs.
# Runs up to MAX_PARALLEL combos simultaneously (safe — no SUMO port conflicts).
# Safe to kill and resume at any time.

PYTHON=".venv/bin/python"
OUTDIR="results_ep_count"
LOGDIR="$OUTDIR/_logs"
MAX_PARALLEL=8
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

# --- Monitor mode (run in separate terminal) ---
if [ "${1:-}" = "monitor" ]; then
  echo "=== MONITOR MODE — Live episode progress ==="
  echo "Press Ctrl+C to stop"
  echo ""
  while true; do
    clear
    echo "=== Episode Count Study — $(date '+%H:%M:%S') ==="
    echo ""
    for combo in "${COMBOS[@]}"; do
      read -r agent env reward eps <<< "$combo"
      tag="${agent}_${env}_${reward}_eps${eps}"
      rdir="$OUTDIR/$tag"
      for pf in "$rdir"/seed_*/progress_seed*.jsonl; do
        [ -f "$pf" ] || continue
        seed=$(basename "$(dirname "$pf")")
        last=$(tail -1 "$pf" 2>/dev/null)
        if [ -n "$last" ]; then
          ep=$(echo "$last" | python3 -c "import sys,json; print(json.load(sys.stdin)['ep'])" 2>/dev/null)
          tt=$(echo "$last" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['tt']:.1f}\")" 2>/dev/null)
          reward_val=$(echo "$last" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['reward']:.1f}\")" 2>/dev/null)
          printf "%-45s %s ep %3d/%d  tt=%ss  reward=%s\n" "$tag" "$seed" "$ep" "$eps" "$tt" "$reward_val"
        fi
      done
    done
    sleep 5
  done
  exit 0
fi

TOTAL=${#COMBOS[@]}
echo "=== Episode count study (parallel=$MAX_PARALLEL): $TOTAL combos x 5 seeds ==="
echo "Start: $(date)"

# --- Build work list (skip already-completed combos) ---
WORK=()
SKIPPED=0
for combo in "${COMBOS[@]}"; do
  read -r agent env reward eps <<< "$combo"
  tag="${agent}_${env}_${reward}_eps${eps}"
  rdir="$OUTDIR/$tag"

  if [ -f "$rdir/aggregate.json" ]; then
    nseeds=$(python3 -c "
import json; d=json.load(open('$rdir/aggregate.json'));
print(len(d.get('per_seed',[])))" 2>/dev/null || echo 0)
    if [ "$nseeds" -ge 5 ]; then
      echo "[SKIP] $tag — $nseeds seeds"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
  fi
  WORK+=("$combo")
done

REMAINING=${#WORK[@]}
echo "[INFO] $SKIPPED skipped, $REMAINING remaining"
echo ""

if [ "$REMAINING" -eq 0 ]; then
  echo "All combos already complete!"
  exit 0
fi

# --- Run combos in parallel ---
GLOBAL_T0=$SECONDS
FAILED=0
COMPLETED=0
RUNNING=0
declare -A PID_MAP  # pid -> combo name

run_one() {
  local combo="$1"
  read -r agent env reward eps <<< "$combo"
  local tag="${agent}_${env}_${reward}_eps${eps}"
  local rdir="$OUTDIR/$tag"
  local logfile="$LOGDIR/${tag}.log"

  echo "[START] $tag — $(date '+%H:%M:%S')"
  CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    agent="$agent" env="$env" reward="$reward" \
    seeds=5 num_episodes=$eps device=cuda resume=true \
    run_dir="$rdir" \
    > "$logfile" 2>&1

  return $?
}

idx=0
for combo in "${WORK[@]}"; do
  idx=$((idx + 1))

  # If at capacity, wait for one job to finish
  while [ "$RUNNING" -ge "$MAX_PARALLEL" ]; do
    # wait -n returns pid of finished job (bash 4.3+)
    wait -n -p FINISHED_PID 2>/dev/null
    rc=$?
    finished_combo="${PID_MAP[$FINISHED_PID]:-unknown}"
    unset "PID_MAP[$FINISHED_PID]"
    RUNNING=$((RUNNING - 1))
    COMPLETED=$((COMPLETED + 1))
    if [ $rc -eq 0 ]; then
      echo "[DONE] $finished_combo (rc=0) — $COMPLETED/$REMAINING done, $RUNNING running"
    else
      echo "[FAIL] $finished_combo (rc=$rc) — see log"
      FAILED=$((FAILED + 1))
    fi
  done

  # Launch new job
  run_one "$combo" &
  pid=$!
  PID_MAP[$pid]="$combo"
  RUNNING=$((RUNNING + 1))
  echo "[LAUNCH $idx/$REMAINING] pid=$pid — $RUNNING running"
done

echo ""
echo "=== All $REMAINING jobs launched. Waiting for last $RUNNING... ==="

# Wait for all remaining jobs
for pid in "${!PID_MAP[@]}"; do
  wait "$pid"
  rc=$?
  combo="${PID_MAP[$pid]}"
  COMPLETED=$((COMPLETED + 1))
  if [ $rc -eq 0 ]; then
    echo "[DONE] $combo (rc=0) — $COMPLETED/$REMAINING done"
  else
    echo "[FAIL] $combo (rc=$rc)"
    FAILED=$((FAILED + 1))
  fi
done

total_elapsed=$(( SECONDS - GLOBAL_T0 ))
echo ""
echo "=== Done: $COMPLETED/$REMAINING completed, $FAILED failed, ${total_elapsed}s ==="
echo "End: $(date)"

# --- Verification ---
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
