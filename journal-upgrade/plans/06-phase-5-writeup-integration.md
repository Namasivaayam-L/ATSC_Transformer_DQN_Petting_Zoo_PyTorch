# Phase 5 — Write-up Integration (Week 8 + buffer)

## Context recap
Fold the real results into the manuscript and fix the correctness/clarity defects flagged in the
original paper. This phase also absorbs schedule overruns.

## Prerequisites
- Phase 4 figures/tables generated.

## Tasks

### 5.1 — Rewrite Methods
- Describe the spatial-coordination transformer precisely: tokenisation, neighbour graph, multi-head
  attention, Q-head, and the Double-DQN/target-network training (which the original CLAIMED via the
  theta-prime in its loss equation but did not implement).
- State the true observation and action shapes (the original conflated an 8-dim state with a length-4
  action and misused "max-pooling" for "argmax"). Correct all of this.

### 5.2 — Fix the equations
- Bellman optimality (original Eq. 1) is missing the summation over (s', r): restore the expectation/sum.
- DQN loss (original Eq. 3) is missing the expectation/mean over the minibatch: write it correctly with
  the target-network parameters.

### 5.3 — Rewrite Results around absolute numbers
- Lead every table with average travel time ± 95% CI. Report secondary metrics. Replace qualitative
  plot descriptions with the rliable figures and significance statements.

### 5.4 — Fix figure hygiene
- The original reused the IDENTICAL caption ("architecture of our Transformer-based DQN model…") for
  Figs. 3, 4, 7, 8, 9. Give every figure a unique, accurate caption. Remove placeholder figures.

### 5.5 — Reconcile narrative with scope
- Remove or reframe the computer-vision (DETR/dlib) sections — CV is cut. Move any CV mention to a brief
  future-work/deployment paragraph only.
- Fix the OpenCV-vs-dlib tooling contradiction by simply not making real-time CV claims.

### 5.6 — Limitations & ethics
- Add sim-to-real gap, no real-world deployment claims, fairness across directions, and the scope of the
  scenarios (grid4x4 + cologne3).

### 5.7 — Reproducibility statement
- Point to the cleaned repo, configs, seeds, and the one-command figure regeneration.

## Files created/edited
- The manuscript (LaTeX). New: `paper/` if the manuscript is brought into this tree, else edit in place.

## ACCEPTANCE GATE (Phase 5)
The manuscript's every numeric claim is backed by a logged result; equations are correct; every figure
has a unique accurate caption; CV claims are removed/reframed; a reproducibility statement is present.
Run a similarity check (see backlog) before declaring done.

## Dependencies
Requires Phase 4. Also depends on backlog item 01 (publication-status) being resolved so the framing
(new submission vs. substantial extension) is correct.
