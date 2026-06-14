# Old buggy TRF-DQN driver. SUPERSEDED by Phase 1 (agents/idqn.py + train.py, CleanRL-style).
# Kept temporarily to make the import surface explicit during the Phase 0 refactor.
# The hardcoded sys.path.append has been removed (Plan: memory/04-diagnosis-and-suggestions.md).
# Do NOT import from this module — use `from agents import idqn` and the Hydra entry point `train.py`.
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
