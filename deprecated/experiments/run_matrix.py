"""Full experiment matrix runner — Phase 3.

Orchestrates: methods × scenarios × reward functions × seeds.

Usage:
    # Full matrix (overnight):
    python experiments/run_matrix.py

    # Single method/scenario:
    python experiments/run_matrix.py agent=trf_coord env=grid4x4 reward=dwt seeds=5

    # Equal-parameter ablation:
    python experiments/run_matrix.py +ablation=equal_params
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class RunSpec:
    agent: str
    env: str
    reward: str
    seed: int
    num_episodes: int = 200


EXPERIMENT_MATRIX = {
    "agents": ["fixed_time", "max_pressure", "mplight", "idqn", "trf_coord"],
    "envs": ["grid4x4", "cologne3"],
    "rewards": ["dwt", "pressure", "queue"],
    "seeds": list(range(5)),
}


def _build_hydra_overrides(spec: RunSpec) -> list[str]:
    """Build Hydra CLI overrides for a single run."""
    overrides = [
        f"agent={spec.agent}",
        f"env={spec.env}",
        f"reward={spec.reward}",
        f"seed={spec.seed}",
        f"num_episodes={spec.num_episodes}",
    ]
    return overrides


def run_single(spec: RunSpec, dry_run: bool = False) -> dict:
    """Execute a single training run."""
    overrides = _build_hydra_overrides(spec)
    cmd = [sys.executable, "train.py"] + overrides

    run_dir = Path("outputs") / f"{spec.agent}_{spec.env}_{spec.reward}_s{spec.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running: {' '.join(cmd)}")
    print(f"  Output:  {run_dir}")

    if dry_run:
        return {"status": "dry_run", "cmd": cmd, "run_dir": str(run_dir)}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)

    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=3600
    )

    log_file = run_dir / "train.log"
    log_file.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)

    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "run_dir": str(run_dir),
        "log": str(log_file),
    }


def run_matrix(
    config: Optional[DictConfig] = None,
    agents: Optional[list[str]] = None,
    envs: Optional[list[str]] = None,
    rewards: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    num_episodes: int = 200,
    dry_run: bool = False,
):
    """Run the full experiment matrix."""
    agents = agents or EXPERIMENT_MATRIX["agents"]
    envs = envs or EXPERIMENT_MATRIX["envs"]
    rewards = rewards or EXPERIMENT_MATRIX["rewards"]
    seeds = seeds or EXPERIMENT_MATRIX["seeds"]

    specs = [
        RunSpec(agent=a, env=e, reward=r, seed=s, num_episodes=num_episodes)
        for a, e, r, s in itertools.product(agents, envs, rewards, seeds)
    ]

    total = len(specs)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT MATRIX: {total} runs")
    print(f"  Agents:   {agents}")
    print(f"  Envs:     {envs}")
    print(f"  Rewards:  {rewards}")
    print(f"  Seeds:    {seeds}")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    results = []
    for i, spec in enumerate(specs):
        print(f"\n[{i+1}/{total}] {spec.agent} / {spec.env} / {spec.reward} / seed={spec.seed}")
        try:
            result = run_single(spec, dry_run=dry_run)
            result["spec"] = vars(spec)
            results.append(result)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 3600s")
            results.append({"status": "timeout", "spec": vars(spec)})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"status": "error", "error": str(e), "spec": vars(spec)})

    # Summary
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'='*60}")
    print(f"RESULTS: {completed} completed, {failed} failed, {total - completed - failed} other")
    print(f"{'='*60}")

    return results


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    run_matrix(config)


if __name__ == "__main__":
    main()
