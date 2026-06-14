# 99 — Decision Gates (Hard Checkpoints)

Two gates protect the <2-month timeline. Treat them as go/no-go; do not proceed past a failed gate by
"hoping it works out later."

## GATE A — end of Phase 1 (≈ Week 2): Does the rebuilt core learn?
- **Criterion:** On `grid4x4`, rebuilt IDQN beats Fixed-time on average travel time across 5 seeds with
  clearly-better CIs, and updates-per-episode are in the hundreds+.
- **PASS →** proceed to Phase 2 (transformer).
- **FAIL →** STOP. Debug the RL core (most likely culprits: training cadence, target network, eval-mode
  leakage, observation shape). Nothing downstream can succeed until this passes. Do not start the
  transformer.

## GATE B — Week 4: Does the spatial transformer train?
- **Criterion:** `trf_coord` trains stably and at least matches IDQN within ~2 weeks of tuning.
- **PASS →** spatial coordination is the contribution; proceed to Phase 3 with `trf_coord`.
- **FALL BACK →** switch to `trf_temporal` (history-attention). Lower risk, still publishable. You lose
  ~3 days, not the deadline. Proceed to Phase 3 with `trf_temporal`.
- **Rule:** make this call ON Week 4. Do not let transformer tuning consume Weeks 5–6 (reserved for
  baselines/ablations).

## Standing rule
If at any point the full method×scenario×reward×seed matrix will not fit the compute budget before the
deadline, cut SECONDARY ablations first, then drop `cologne3` to a single reward, before ever cutting
seeds below 5 or dropping a mandatory baseline (Fixed-time, Max-Pressure, IDQN, MPLight). Log every cut
explicitly in the paper's experimental-setup section — never silently truncate.
