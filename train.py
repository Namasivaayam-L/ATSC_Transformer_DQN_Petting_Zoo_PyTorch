"""Training entry point for IDQN and Spatial Coordination Transformer.

Hydra-driven.  Fixes bug #1 (training cadence) by updating every `train_freq`
env steps after `learning_starts`, NOT only at episode termination.

Usage:
    uv run python train.py agent=idqn   env=grid4x4 reward=dwt seed=0
    uv run python train.py agent=trf_coord env=grid4x4 reward=dwt seed=0
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")

from agents.idqn import IDQNAgent  # noqa: E402
from agents.trf_coord import TrfCoordAgent  # noqa: E402
from eval.metrics import EpisodeMetrics, aggregate_over_seeds, build_episode_metrics  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402

AgentType = Union[IDQNAgent, TrfCoordAgent]


def _init_tracker(cfg: DictConfig):
    tracker = cfg.get("tracker", "tensorboard")
    log_dir = os.path.join(cfg.run_dir, "tb")
    os.makedirs(log_dir, exist_ok=True)
    if tracker == "wandb":
        try:
            import wandb
            wandb.init(project=cfg.wandb_project,
                       config=OmegaConf.to_container(cfg, resolve=True))
            return {"name": "wandb", "wandb": wandb}
        except Exception:
            pass
    from torch.utils.tensorboard import SummaryWriter
    return {"name": "tensorboard", "writer": SummaryWriter(log_dir=log_dir)}


def _log(tracker: dict, tag: str, value: float, step: int) -> None:
    if tracker["name"] == "tensorboard":
        tracker["writer"].add_scalar(tag, value, step)
    elif tracker["name"] == "wandb":
        tracker["wandb"].log({tag: value}, step=step)


def _build_env(cfg: DictConfig, output_dir: str, seed: int):
    from sumo_rl.environment import resco_envs
    scenario = cfg.env.scenario
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


def _build_adjacency(env, cfg: DictConfig) -> Dict[str, List[str]]:
    """Build the adjacency graph from the SUMO net file."""
    from env.road_graph import build_adjacency
    u = env.unwrapped.env
    net_file = u._net
    ts_ids = list(env.possible_agents)
    return build_adjacency(net_file, ts_ids)


def _create_agents(
    cfg: DictConfig,
    possible_agents: List[str],
    obs_dim: int,
    act_dim: int,
    adj: Dict[str, List[str]],
    device: str = "cpu",
) -> Dict[str, AgentType]:
    """Create one agent per traffic signal."""
    agent_name = cfg.agent.name

    if agent_name == "idqn":
        hidden = tuple(cfg.agent.network.hidden_sizes)
        return {
            ts: IDQNAgent(
                obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden,
                lr=cfg.learning_rate, gamma=cfg.gamma, tau=cfg.tau,
                target_update_interval=cfg.target_update_interval,
                buffer_size=cfg.buffer_size, batch_size=cfg.batch_size,
                learning_starts=cfg.learning_starts, train_freq=cfg.train_freq,
                dueling=cfg.get("dueling", False), device=device,
            )
            for ts in possible_agents
        }

    if agent_name == "trf_coord":
        return {
            ts: TrfCoordAgent(
                obs_dim=obs_dim, act_dim=act_dim,
                adj=adj, agent_id=ts,
                max_neighbours=cfg.agent.network.get("neighbour_radius", 8),
                embedding_dim=cfg.agent.network.embedding_dim,
                num_heads=cfg.agent.network.num_heads,
                num_enc_layers=cfg.agent.network.num_enc_layers,
                lr=cfg.learning_rate, gamma=cfg.gamma, tau=cfg.tau,
                target_update_interval=cfg.target_update_interval,
                buffer_size=cfg.buffer_size, batch_size=cfg.batch_size,
                learning_starts=cfg.learning_starts, train_freq=cfg.train_freq,
                device=device,
            )
            for ts in possible_agents
        }

    raise ValueError(f"Unknown agent: {agent_name}")


def _select_action(agent: AgentType, obs_dict: Dict[str, np.ndarray], ts: str) -> int:
    """Select action — handle both flat (IDQN) and tokenised (trf_coord) agents."""
    if isinstance(agent, TrfCoordAgent):
        return agent.act(obs_dict)
    else:
        return agent.act(obs_dict[ts].flatten().astype(np.float32))


def _store_transition(
    agent: AgentType,
    obs_dict: Dict[str, np.ndarray],
    next_obs_dict: Dict[str, np.ndarray],
    ts: str, action: int, reward: float, done: bool,
) -> None:
    """Store a transition — flat for IDQN, tokenised for trf_coord."""
    if isinstance(agent, TrfCoordAgent):
        tokens = agent._build_tokens(obs_dict)
        next_tokens = agent._build_tokens(next_obs_dict)
        agent.buffer.add(tokens, action, reward, next_tokens, done)
    else:
        flat_cur = obs_dict[ts].flatten().astype(np.float32)
        flat_next = next_obs_dict[ts].flatten().astype(np.float32)
        agent.buffer.add(flat_cur, action, reward, flat_next, done)


def train_one_seed(cfg: DictConfig, seed: int, run_root: str) -> EpisodeMetrics:
    seed_everything(seed)
    out_dir = os.path.join(run_root, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    tracker = _init_tracker(cfg)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    env = _build_env(cfg, out_dir, seed=seed)
    possible_agents = list(env.possible_agents)

    # Infer obs/act dims from the ACTUAL env output
    sample_obs_dict, _ = env.reset()
    sample_obs = sample_obs_dict[possible_agents[0]]
    obs_dim = int(np.prod(sample_obs.shape))
    act_dim = int(env.action_spaces[possible_agents[0]].n)

    # Build adjacency graph for transformer agents (without closing env)
    adj: Dict[str, List[str]] = {}
    agent_name = cfg.agent.name
    if agent_name == "trf_coord":
        from env.road_graph import build_adjacency
        u = env.unwrapped.env
        adj = build_adjacency(u._net, possible_agents)
        print(f"[train] Adjacency graph built: {len(adj)} nodes")
        for k, v in adj.items():
            print(f"  {k}: {v}")

    agents = _create_agents(cfg, possible_agents, obs_dim, act_dim, adj)
    global_step = 0
    ep_metrics: List[EpisodeMetrics] = []

    for ep in range(int(cfg.num_episodes)):
        obs_dict, _info = env.reset()
        done = False
        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_updates = 0

        while not done:
            actions = {}
            for ts in possible_agents:
                actions[ts] = _select_action(agents[ts], obs_dict, ts)

            result = env.step(actions)
            if len(result) == 5:
                next_obs_dict, rewards, term_dict, trunc_dict, _info = result
            else:
                next_obs_dict, rewards, done_dict, _info = result
                term_dict = {ts: done_dict.get(ts, False) for ts in possible_agents}
                trunc_dict = {ts: False for ts in possible_agents}

            for ts in possible_agents:
                r = float(rewards[ts])
                d = bool(term_dict.get(ts, False) or trunc_dict.get(ts, False))
                ep_reward += r
                _store_transition(agents[ts], obs_dict, next_obs_dict, ts, actions[ts], r, d)
                loss = agents[ts].learn()
                if loss is not None:
                    ep_loss_sum += loss
                    ep_updates += 1

            obs_dict = next_obs_dict
            done = all(term_dict.get(ts, False) or trunc_dict.get(ts, False)
                       for ts in possible_agents)
            global_step += 1

        for ts in possible_agents:
            agents[ts].on_episode_end()

        avg_loss = ep_loss_sum / max(ep_updates, 1)
        _log(tracker, "train/ep_reward", ep_reward, ep)
        _log(tracker, "train/ep_loss", avg_loss, ep)
        _log(tracker, "train/epsilon", agents[possible_agents[0]].epsilon, ep)
        _log(tracker, "train/updates_per_ep", ep_updates, ep)

        print(f"  ep {ep:>4d} | reward {ep_reward:>8.1f} | loss {avg_loss:.4f} | "
              f"eps {agents[possible_agents[0]].epsilon:.3f} | updates {ep_updates}", flush=True)

        save_interval = cfg.get("save_interval", 50)
        if (ep + 1) % save_interval == 0 or ep == cfg.num_episodes - 1:
            for ts in possible_agents:
                agents[ts].save(os.path.join(ckpt_dir, f"{ts}_ep{ep}.pt"))

        ep_m = build_episode_metrics(out_dir)
        ep_metrics.append(ep_m)

    env.close()
    for t in tracker.values():
        if hasattr(t, "close"):
            t.close()

    return _mean_metrics(ep_metrics)


def _mean_metrics(metrics: List[EpisodeMetrics]) -> EpisodeMetrics:
    if not metrics:
        return EpisodeMetrics(0, 0, 0, 0, 0)
    keys = ["avg_travel_time", "avg_waiting_time", "mean_queue_length",
            "throughput", "total_time_loss", "n_vehicles_completed", "sim_time"]
    out = {k: float(np.nanmean([getattr(m, k) for m in metrics])) for k in keys}
    return EpisodeMetrics(**out)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.run_dir = os.getcwd()
    os.makedirs(cfg.run_dir, exist_ok=True)
    print("[train] resolved config:\n", OmegaConf.to_yaml(cfg), flush=True)

    seeds = list(range(int(cfg.seeds)))
    per_seed: List[EpisodeMetrics] = []
    for s in seeds:
        print(f"\n[train] ===== seed {s} =====", flush=True)
        m = train_one_seed(cfg, s, run_root=cfg.run_dir)
        per_seed.append(m)
        print(f"[train] seed {s}: travel_time={m.avg_travel_time:.2f}s, "
              f"wait={m.avg_waiting_time:.2f}s, throughput={int(m.throughput)}", flush=True)

    agg = aggregate_over_seeds(per_seed)
    with open(os.path.join(cfg.run_dir, "aggregate.json"), "w") as f:
        json.dump(
            {"config": OmegaConf.to_container(cfg, resolve=True),
             "per_seed": [asdict(m) for m in per_seed],
             "aggregate": agg},
            f, indent=2,
        )
    print("\n[train] aggregate (mean ± 95% bootstrap CI):")
    for k, v in agg.items():
        print(f"  {k:>20s} = {v['mean']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]  n={v['n']}")


if __name__ == "__main__":
    main()
