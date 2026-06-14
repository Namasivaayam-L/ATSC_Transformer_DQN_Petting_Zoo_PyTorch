"""Frozen evaluation loop — runs an agent (or rule-based baseline) on a SUMO
scenario for N seeds, never calls `learn()`, never updates weights.

This is the permanent fix for contaminated-evaluation bug #5 (the old
`dqn/main.py` reused the training loop with epsilon=0 — kept learning).

Phase 0 acceptance gate uses this with `agent=fixed_time` to prove the
pipeline logs the 5 metrics end-to-end with zero hardcoded paths.

Phase 1+ will reuse this with `agent=idqn` and a checkpoint loader.

Usage:
    uv run python -m eval.evaluate env=grid4x4 agent=fixed_time seeds=5
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Ensure repo root is importable when invoked as `python -m eval.evaluate`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")

from utils.seeding import seed_everything  # noqa: E402
from eval.metrics import (  # noqa: E402
    EpisodeMetrics,
    aggregate_over_seeds,
    build_episode_metrics,
)

# -- Optional: TensorBoard (Task 0.3) ---------------------------------------
_TRACKER_WRITERS: Dict[str, Any] = {}


def _init_tracker(cfg: DictConfig) -> None:
    name = cfg.get("tracker", "tensorboard")
    if name == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(cfg.run_dir, "tb")
        os.makedirs(log_dir, exist_ok=True)
        _TRACKER_WRITERS["tb"] = SummaryWriter(log_dir=log_dir)
    elif name == "wandb":
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))
            _TRACKER_WRITERS["wandb"] = wandb
        except Exception as e:  # noqa: BLE001
            print(f"[evaluate] wandb init failed ({e}); falling back to tensorboard")
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(cfg.run_dir, "tb")
            os.makedirs(log_dir, exist_ok=True)
            _TRACKER_WRITERS["tb"] = SummaryWriter(log_dir=log_dir)


def _log_metrics(prefix: str, m: EpisodeMetrics, step: int) -> None:
    if "tb" in _TRACKER_WRITERS:
        w = _TRACKER_WRITERS["tb"]
        for k, v in m.as_dict().items():
            w.add_scalar(f"{prefix}/{k}", v, step)
    if "wandb" in _TRACKER_WRITERS:
        _TRACKER_WRITERS["wandb"].log({f"{prefix}/{k}": v for k, v in m.as_dict().items()}, step=step)


# -- Agent adapters --------------------------------------------------------

class RuleBasedAgent:
    """Fixed-time cyclic controller. Used by the Phase 0 acceptance gate.

    Selects one of `env.action_spaces[agent].n` phases in round-robin
    every `cycle_seconds / num_phases` of simulated time, then holds.
    """

    def __init__(self, n_phases: int, cycle_seconds: int = 30):
        self.n_phases = int(n_phases)
        self.cycle_seconds = int(cycle_seconds)
        self._counter = 0
        self._last_switch = 0.0

    def reset(self) -> None:
        self._counter = 0
        self._last_switch = 0.0

    def act(self, obs: np.ndarray, sim_time: float) -> int:
        if sim_time - self._last_switch >= self.cycle_seconds / max(self.n_phases, 1):
            self._counter = (self._counter + 1) % self.n_phases
            self._last_switch = sim_time
        return self._counter


def _make_agent(cfg: DictConfig, n_phases: int):
    name = cfg.agent.name
    if name == "fixed_time":
        return RuleBasedAgent(n_phases=n_phases, cycle_seconds=cfg.agent.cycle_seconds)
    if name in ("idqn", "trf_coord", "trf_temporal"):
        raise NotImplementedError(
            f"agent={name} arrives in Phase 1/2. Phase 0 gate uses agent=fixed_time."
        )
    raise ValueError(f"Unknown agent: {name}")


# -- SUMO env adapter ------------------------------------------------------

def _build_resco_env(cfg: DictConfig, output_dir: str, seed: int):
    """Instantiate the RESCO grid4x4 / cologne3 env from the vendored fork.

    The vendored `sumo_rl/environment/resco_envs.py` exposes **factory
    functions** that return a PettingZoo `parallel_env`. The factory wires
    the net.xml / route.xml paths internally; we only forward overrides
    like `use_gui`, `num_seconds`, signal timings, and an output dir.
    """
    from sumo_rl.environment import resco_envs

    scenario = cfg.env.scenario
    if not hasattr(resco_envs, scenario):
        raise ValueError(
            f"Unknown scenario '{scenario}'. Available: "
            f"{[n for n in dir(resco_envs) if not n.startswith('_')]}"
        )

    factory = getattr(resco_envs, scenario)
    return factory(
        parallel=True,
        out_csv_name=os.path.join(output_dir, "csv"),
        use_gui=cfg.env.use_gui,
        num_seconds=cfg.num_seconds,
        yellow_time=cfg.yellow_time,
        min_green=cfg.min_green,
        max_green=cfg.max_green,
        reward_fn=cfg.reward.fn,
        sumo_warnings=False,
        additional_sumo_cmd=cfg.env.additional_sumo_cmd or None,
        single_agent=False,
    )


# -- Per-seed run ----------------------------------------------------------

def run_one_seed(cfg: DictConfig, seed: int, run_root: str) -> EpisodeMetrics:
    """One frozen evaluation seed: build env, run all episodes, return agg metrics.

    Uses the PettingZoo `parallel_env` interface:
        reset()  -> (obs_dict, info)
        step(act)-> (obs, reward, term, trunc, info)
    All agents step in lockstep. Episode ends when all agents are terminated
    or truncated.
    """
    seed_everything(seed)
    out_dir = os.path.join(run_root, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    env = _build_resco_env(cfg, out_dir, seed=seed)

    possible_agents = list(env.possible_agents)
    agents = {ts: _make_agent(cfg, n_phases=env.action_spaces[ts].n) for ts in possible_agents}
    for ag in agents.values():
        if hasattr(ag, "reset"):
            ag.reset()

    # Multiple episodes per seed for stability
    ep_metrics: List[EpisodeMetrics] = []
    for ep in range(int(cfg.eval_episodes)):
        obs, _info = env.reset()
        sim_time = 0.0
        done = False
        step_count = 0
        max_steps = int(cfg.num_seconds) * 5  # generous cap (5x for safety)
        while not done and step_count < max_steps:
            actions = {}
            for ts in possible_agents:
                # Use the env's current observation if available; fall back to zeros.
                cur_obs = obs.get(ts, None) if obs else None
                if cur_obs is None:
                    cur_obs = np.zeros(1, dtype=np.float32)
                actions[ts] = ag_act(agents[ts], cur_obs, sim_time)
            result = env.step(actions)
            if len(result) == 5:
                obs, _rew, term, trunc, _info = result
            else:
                # Some PettingZoo versions return 4-tuples
                obs, _rew, done_dict, _info = result
                term = {ts: done_dict.get(ts, False) for ts in possible_agents}
                trunc = {ts: False for ts in possible_agents}
            term_all = all(term.values()) if term else False
            trunc_all = all(trunc.values()) if trunc else False
            done = term_all or trunc_all
            sim_time = getattr(env, "sim_step", sim_time + cfg.delta_time)
            step_count += 1
        # Explicitly save CSV after each episode (sumo_rl only saves on next reset)
        # Navigate: parallel_wrapper -> aec_to_parallel_wrapper -> SumoEnvironmentPZ -> SumoEnvironment
        try:
            pz_env = env.unwrapped
            sumo_env = pz_env.env
            sumo_env.save_csv(sumo_env.out_csv_name, sumo_env.episode)
        except Exception:
            pass  # best-effort; CSV may not be saved for single-episode runs
        ep_dir = os.path.join(out_dir)
        ep_metrics.append(build_episode_metrics(ep_dir))

    env.close()
    return _mean_metrics(ep_metrics)


def ag_act(agent, obs, sim_time):
    """Polymorphic action selection (rule-based or learned)."""
    if isinstance(agent, RuleBasedAgent):
        return agent.act(obs, sim_time)
    # learned agents (Phase 1+) accept (obs, epsilon=0) or (obs, sim_time)
    return agent.act(obs, epsilon=0.0)


def _mean_metrics(metrics: List[EpisodeMetrics]) -> EpisodeMetrics:
    """Average the per-episode metrics within a single seed."""
    if not metrics:
        return EpisodeMetrics(0, 0, 0, 0, 0)
    keys = ["avg_travel_time", "avg_waiting_time", "mean_queue_length",
            "throughput", "total_time_loss", "n_vehicles_completed", "sim_time"]
    out = {k: float(np.nanmean([getattr(m, k) for m in metrics])) for k in keys}
    return EpisodeMetrics(**out)


def _find_first(directory: str, name: str) -> str:
    """Find the first `name` under `directory` (or `directory` itself)."""
    candidate = os.path.join(directory, name)
    if os.path.isfile(candidate):
        return candidate
    if not os.path.isdir(directory):
        return candidate
    for root, _, files in os.walk(directory):
        if name in files:
            return os.path.join(root, name)
    return candidate


# -- Hydra entry point -----------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.run_dir = os.getcwd()  # Hydra already created the run dir
    os.makedirs(cfg.run_dir, exist_ok=True)
    print("[evaluate] resolved config:\n", OmegaConf.to_yaml(cfg))

    _init_tracker(cfg)
    seeds = list(range(int(cfg.seeds)))
    per_seed: List[EpisodeMetrics] = []
    for s in seeds:
        print(f"[evaluate] running seed {s} ...")
        m = run_one_seed(cfg, seed=s, run_root=cfg.run_dir)
        per_seed.append(m)
        _log_metrics("eval/seed", m, step=s)
        print(f"[evaluate]   seed {s}: travel_time={m.avg_travel_time:.2f}s, "
              f"wait={m.avg_waiting_time:.2f}s, throughput={int(m.throughput)}")

    agg = aggregate_over_seeds(per_seed)
    with open(os.path.join(cfg.run_dir, "aggregate.json"), "w") as f:
        json.dump(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "per_seed": [asdict(m) for m in per_seed],
                "aggregate": agg,
            },
            f,
            indent=2,
        )
    print("\n[evaluate] aggregate (mean ± 95% bootstrap CI) over", len(seeds), "seeds:")
    for k, v in agg.items():
        print(f"  {k:>20s} = {v['mean']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]  n={v['n']}")

    if "tb" in _TRACKER_WRITERS:
        _TRACKER_WRITERS["tb"].close()


if __name__ == "__main__":
    main()
