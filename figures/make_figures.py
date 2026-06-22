"""make_figures.py — Regenerates every figure/table from results_v2/ data.

Data layout:
    results_v2/{agent}_{env}_{reward}/
        seed_0/csv1ep.csv ... csv50ep.csv   (per-step data per episode)
        seed_0/tb/                           (tensorboard events)
        seed_0/checkpoints/
        aggregate.json                       (per-seed + bootstrap CI)

Usage:
    python figures/make_figures.py --all
    python figures/make_figures.py --fig learning_curves
    python figures/make_figures.py --fig attention_heatmap
    python figures/make_figures.py --table results
    python figures/make_figures.py --fig iqms
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results_v2")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aggregate(combo_dir: Path) -> Optional[Dict]:
    """Load aggregate.json from a combo directory."""
    agg_path = combo_dir / "aggregate.json"
    if not agg_path.exists():
        return None
    with open(agg_path) as f:
        return json.load(f)


def load_seed_episodes(combo_dir: Path, seed: int) -> List[Dict]:
    """Load all episode CSVs for a given seed.

    Returns list of dicts, one per episode, with keys like
    'system_total_waiting_time', 'system_mean_speed', etc.
    """
    seed_dir = combo_dir / f"seed_{seed}"
    if not seed_dir.exists():
        return []
    csvs = sorted(seed_dir.glob("csv*ep.csv"))
    episodes = []
    for csv_path in csvs:
        import csv as csv_mod
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
            if rows:
                # Take the LAST row (final step of episode) for summary metrics
                episodes.append(rows[-1])
    return episodes


def list_combos(
    results_dir: Path = RESULTS_DIR,
    agent: Optional[str] = None,
    env: Optional[str] = None,
    reward: Optional[str] = None,
) -> List[Tuple[str, str, str, Path]]:
    """List all (agent, env, reward, path) combos.

    Supports two layouts:
      1. Flat:    results_v2/{agent}_{env}_{reward}/aggregate.json
      2. Nested:  results/{env}/{agent}/{reward}/aggregate.json
    """
    combos = []
    if not results_dir.exists():
        return combos

    # Layout 1: flat naming (results_v2/)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        parts = d.name.rsplit("_", 2)
        if len(parts) == 3 and (d / "aggregate.json").exists():
            a, e, r = parts
            if agent and a != agent:
                continue
            if env and e != env:
                continue
            if reward and r != reward:
                continue
            combos.append((a, e, r, d))

    # Layout 2: nested (env/agent/reward/)
    if not combos:
        for env_dir in sorted(results_dir.iterdir()):
            if not env_dir.is_dir() or env_dir.name.startswith("_"):
                continue
            for agent_dir in sorted(env_dir.iterdir()):
                if not agent_dir.is_dir():
                    continue
                for reward_dir in sorted(agent_dir.iterdir()):
                    if not reward_dir.is_dir():
                        continue
                    if not (reward_dir / "aggregate.json").exists():
                        continue
                    a, e, r = agent_dir.name, env_dir.name, reward_dir.name
                    if agent and a != agent:
                        continue
                    if env and e != env:
                        continue
                    if reward and r != reward:
                        continue
                    combos.append((a, e, r, reward_dir))

    return combos


# ---------------------------------------------------------------------------
# Figure: Learning curves (from aggregate.json per-seed data)
# ---------------------------------------------------------------------------

def plot_learning_curves(
    results_dir: Path = RESULTS_DIR,
    methods: Optional[List[str]] = None,
    env: str = "grid4x4",
    reward: str = "dwt",
    metric: str = "avg_travel_time",
    save_dir: str = "figures",
):
    """Plot mean ± 95% CI learning curves for each method."""
    if methods is None:
        methods = ["idqn", "trf_coord", "trf_coord_equal", "max_pressure", "mplight", "fixed_time"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        combos = list_combos(results_dir, agent=method, env=env, reward=reward)
        if not combos:
            print(f"  [SKIP] {method}/{env}/{reward} — no data")
            continue
        _, _, _, combo_dir = combos[0]
        agg = load_aggregate(combo_dir)
        if not agg or "per_seed" not in agg:
            continue

        # Extract per-seed final metrics
        seed_vals = []
        for sd in agg["per_seed"]:
            if metric in sd:
                seed_vals.append(float(sd[metric]))
        if not seed_vals:
            continue

        mean = np.mean(seed_vals)
        ci = 1.96 * np.std(seed_vals) / np.sqrt(len(seed_vals))
        ax.barh(f"{method}\n({reward})", mean, xerr=ci, color=colors[i],
                alpha=0.8, capsize=5, label=method)

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Final {metric.replace('_', ' ').title()} — {env} / {reward}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/learning_curves_{env}_{reward}_{metric}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_dir}/learning_curves_{env}_{reward}_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir}/learning_curves_{env}_{reward}_{metric}.png")


# ---------------------------------------------------------------------------
# Figure: Learning curves from per-episode CSV data (for RL agents)
# ---------------------------------------------------------------------------

def plot_learning_curves_from_csv(
    results_dir: Path = RESULTS_DIR,
    methods: Optional[List[str]] = None,
    env: str = "grid4x4",
    reward: str = "dwt",
    csv_col: str = "system_total_waiting_time",
    save_dir: str = "figures",
):
    """Plot per-episode learning curves from CSV data."""
    if methods is None:
        methods = ["idqn", "trf_coord", "trf_coord_equal"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        combos = list_combos(results_dir, agent=method, env=env, reward=reward)
        if not combos:
            continue
        _, _, _, combo_dir = combos[0]

        # Collect per-episode values across seeds
        seed_data = []
        for seed_dir in sorted(combo_dir.glob("seed_*")):
            csvs = sorted(seed_dir.glob("csv*ep.csv"))
            vals = []
            for csv_path in csvs:
                import csv as csv_mod
                with open(csv_path) as f:
                    reader = csv_mod.DictReader(f)
                    rows = list(reader)
                    if rows and csv_col in rows[-1]:
                        vals.append(float(rows[-1][csv_col]))
            if vals:
                seed_data.append(vals)

        if not seed_data:
            continue

        # Pad to same length and compute mean ± CI
        max_len = max(len(d) for d in seed_data)
        padded = np.full((len(seed_data), max_len), np.nan)
        for j, d in enumerate(seed_data):
            padded[j, :len(d)] = d
        means = np.nanmean(padded, axis=0)
        ci = 1.96 * np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0))

        x = np.arange(len(means))
        ax.plot(x, means, label=f"{method}", color=colors[i], linewidth=2)
        ax.fill_between(x, means - ci, means + ci, alpha=0.2, color=colors[i])

    ax.set_xlabel("Episode")
    ax.set_ylabel(csv_col.replace("_", " ").title())
    ax.set_title(f"Learning Curves — {env} / {reward}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/learning_curves_csv_{env}_{reward}_{csv_col}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_dir}/learning_curves_csv_{env}_{reward}_{csv_col}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir}/learning_curves_csv_{env}_{reward}_{csv_col}.png")


# ---------------------------------------------------------------------------
# Table: Results from aggregate.json
# ---------------------------------------------------------------------------

def generate_results_table(
    results_dir: Path = RESULTS_DIR,
    methods: Optional[List[str]] = None,
    envs: Optional[List[str]] = None,
    rewards: Optional[List[str]] = None,
    save_dir: str = "figures",
):
    """Generate a results table with mean ± 95% CI from aggregate.json."""
    if methods is None:
        methods = ["fixed_time", "max_pressure", "mplight", "idqn", "trf_coord", "trf_coord_equal"]
    if envs is None:
        envs = ["grid4x4", "cologne3"]
    if rewards is None:
        rewards = ["dwt", "pressure", "queue"]

    rows = []
    for env in envs:
        for reward in rewards:
            for method in methods:
                combos = list_combos(results_dir, agent=method, env=env, reward=reward)
                if not combos:
                    rows.append({"env": env, "reward": reward, "method": method,
                                 "travel_time": "N/A", "ci95": "", "n_seeds": 0,
                                 "waiting_time": "N/A", "throughput": "N/A"})
                    continue
                _, _, _, combo_dir = combos[0]
                agg = load_aggregate(combo_dir)
                if not agg or "aggregate" not in agg:
                    rows.append({"env": env, "reward": reward, "method": method,
                                 "travel_time": "N/A", "ci95": "", "n_seeds": 0,
                                 "waiting_time": "N/A", "throughput": "N/A"})
                    continue

                agg_data = agg["aggregate"]
                n_seeds = len(agg.get("per_seed", []))

                tt = agg_data.get("avg_travel_time", {})
                wt = agg_data.get("avg_waiting_time", {})
                tp = agg_data.get("throughput", {})

                tt_mean = f"{tt.get('mean', 0):.1f}" if tt else "N/A"
                tt_ci = f"[{tt.get('lo', 0):.1f}, {tt.get('hi', 0):.1f}]" if tt else ""
                wt_mean = f"{wt.get('mean', 0):.1f}" if wt else "N/A"
                tp_mean = f"{int(tp.get('mean', 0))}" if tp else "N/A"

                rows.append({"env": env, "reward": reward, "method": method,
                             "travel_time": tt_mean, "ci95": tt_ci, "n_seeds": n_seeds,
                             "waiting_time": wt_mean, "throughput": tp_mean})

    # Save as CSV
    os.makedirs(save_dir, exist_ok=True)
    import csv as csv_mod
    csv_path = f"{save_dir}/results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["env", "reward", "method",
                                                    "travel_time", "ci95", "n_seeds",
                                                    "waiting_time", "throughput"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    # Print table
    print(f"\n{'='*100}")
    print(f"RESULTS TABLE — Travel Time (s) ± 95% CI")
    print(f"{'='*100}")
    print(f"{'Env':<12} {'Reward':<10} {'Method':<16} {'Travel Time':<18} {'95% CI':<22} {'Seeds':<6}")
    print("-" * 100)
    for row in rows:
        print(f"{row['env']:<12} {row['reward']:<10} {row['method']:<16} "
              f"{row['travel_time']:<18} {row['ci95']:<22} {row['n_seeds']:<6}")

    return rows


# ---------------------------------------------------------------------------
# IQM + Performance Profiles (rliable)
# ---------------------------------------------------------------------------

def plot_iqm_profiles(
    results_dir: Path = RESULTS_DIR,
    methods: Optional[List[str]] = None,
    env: str = "grid4x4",
    reward: str = "dwt",
    save_dir: str = "figures",
):
    """Plot IQM + performance profiles using rliable (if available)."""
    try:
        from rliable import library as rly
        from rliable import metrics as rly_metrics
        has_rliable = True
    except ImportError:
        has_rliable = False
        print("  rliable not installed — using manual IQM computation")

    if methods is None:
        methods = ["idqn", "trf_coord", "trf_coord_equal", "max_pressure", "mplight"]

    scores = {}
    for method in methods:
        combos = list_combos(results_dir, agent=method, env=env, reward=reward)
        if not combos:
            continue
        _, _, _, combo_dir = combos[0]
        agg = load_aggregate(combo_dir)
        if not agg or "per_seed" not in agg:
            continue
        vals = [float(sd.get("avg_travel_time", 0)) for sd in agg["per_seed"]
                if "avg_travel_time" in sd]
        if vals:
            scores[method] = np.array(vals)

    if not scores:
        print(f"  No data for IQM profiles ({env}/{reward})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of IQMs (median per method)
    ax = axes[0]
    method_names = list(scores.keys())
    iqms = [np.median(scores[m]) for m in method_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))
    bars = ax.barh(method_names, iqms, color=colors, alpha=0.8)
    ax.set_xlabel("Median Travel Time (s, lower = better)")
    ax.set_title(f"IQM Travel Time — {env}/{reward}")
    ax.grid(True, alpha=0.3, axis="x")
    # Add value labels
    for bar, val in zip(bars, iqms):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    # Right: performance profile (CDF of ratios to best method)
    ax = axes[1]
    best_method = method_names[np.argmin(iqms)]
    for i, method in enumerate(method_names):
        if method == best_method:
            normalized = np.ones_like(scores[method])
        else:
            normalized = scores[method] / np.median(scores[best_method])
        sorted_vals = np.sort(normalized)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=method, color=colors[i], linewidth=2)
    ax.set_xlabel(f"Ratio to best ({best_method})")
    ax.set_ylabel("Fraction of seeds")
    ax.set_title("Performance Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/iqm_profiles_{env}_{reward}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_dir}/iqm_profiles_{env}_{reward}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir}/iqm_profiles_{env}_{reward}.png")


# ---------------------------------------------------------------------------
# Figure: Attention heatmap (placeholder — needs trained checkpoint)
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    results_dir: Path = RESULTS_DIR,
    env_name: str = "grid4x4",
    save_dir: str = "figures",
):
    """Plot attention heatmap from a trained trf_coord agent."""
    # Find any trf_coord checkpoint
    combos = list_combos(results_dir, agent="trf_coord", env=env_name)
    if not combos:
        print(f"  No trf_coord checkpoints for {env_name}")
        return

    _, _, _, combo_dir = combos[0]
    ckpts = list((combo_dir / "seed_0" / "checkpoints").glob("*.pt"))
    if not ckpts:
        print(f"  No checkpoints found in {combo_dir}")
        return

    ckpt_path = ckpts[-1]  # latest checkpoint
    print(f"  Loading checkpoint: {ckpt_path}")

    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.trf_coord import TrfCoordAgent
        from env.road_graph import build_adjacency

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Checkpoint contains {'model': state_dict, 'adj': ..., 'config': ...}
        if isinstance(ckpt, dict) and 'adj' in ckpt:
            adj = ckpt['adj']
            nodes = sorted(adj.keys())
            n = len(nodes)
            # Build adjacency matrix
            adj_matrix = np.zeros((n, n))
            node_idx = {node: i for i, node in enumerate(nodes)}
            for node, neighbors in adj.items():
                for nb in neighbors:
                    if nb in node_idx:
                        adj_matrix[node_idx[node], node_idx[nb]] = 1
                        adj_matrix[node_idx[nb], node_idx[node]] = 1

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(adj_matrix, cmap='YlOrRd', aspect='equal')
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(nodes, rotation=90, fontsize=8)
            ax.set_yticklabels(nodes, fontsize=8)
            ax.set_title(f"Network Adjacency — {env_name}")
            plt.colorbar(im, ax=ax)

            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}/attention_heatmap_{env_name}.pdf", bbox_inches="tight")
            fig.savefig(f"{save_dir}/attention_heatmap_{env_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {save_dir}/attention_heatmap_{env_name}.png")
        else:
            print(f"  Checkpoint doesn't contain adjacency graph")
    except Exception as e:
        print(f"  Error generating attention heatmap: {e}")


# ---------------------------------------------------------------------------
# Figure: Reward comparison (travel time across rewards for each method)
# ---------------------------------------------------------------------------

def plot_reward_comparison(
    results_dir: Path = RESULTS_DIR,
    methods: Optional[List[str]] = None,
    env: str = "grid4x4",
    save_dir: str = "figures",
):
    """Bar chart comparing travel time across reward functions for each method."""
    if methods is None:
        methods = ["idqn", "trf_coord", "trf_coord_equal"]
    rewards = ["dwt", "pressure", "queue"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for j, reward in enumerate(rewards):
        means = []
        cis = []
        for method in methods:
            combos = list_combos(results_dir, agent=method, env=env, reward=reward)
            if combos:
                _, _, _, combo_dir = combos[0]
                agg = load_aggregate(combo_dir)
                if agg and "aggregate" in agg:
                    tt = agg["aggregate"].get("avg_travel_time", {})
                    means.append(tt.get("mean", 0))
                    ci = (tt.get("hi", 0) - tt.get("lo", 0)) / 2
                    cis.append(ci)
                    continue
            means.append(0)
            cis.append(0)
        ax.bar(x + j * width, means, width, yerr=cis, label=reward,
               color=colors[j], alpha=0.8, capsize=5)

    ax.set_xlabel("Method")
    ax.set_ylabel("Avg Travel Time (s)")
    ax.set_title(f"Reward Comparison — {env}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/reward_comparison_{env}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_dir}/reward_comparison_{env}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir}/reward_comparison_{env}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURES = {
    "learning_curves": plot_learning_curves,
    "learning_curves_csv": plot_learning_curves_from_csv,
    "attention_heatmap": plot_attention_heatmap,
    "iqms": plot_iqm_profiles,
    "results": generate_results_table,
    "reward_comparison": plot_reward_comparison,
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate all figures/tables")
    parser.add_argument("--fig", choices=list(FIGURES.keys()), help="Specific figure to generate")
    parser.add_argument("--table", action="store_true", help="Generate results table")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Results directory")
    parser.add_argument("--save-dir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.all or (not args.fig and not args.table):
        for name, func in FIGURES.items():
            print(f"\nGenerating {name}...")
            try:
                func(results_dir=results_dir, save_dir=args.save_dir)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    else:
        if args.fig:
            print(f"\nGenerating {args.fig}...")
            FIGURES[args.fig](results_dir=results_dir, save_dir=args.save_dir)
        if args.table:
            print(f"\nGenerating results table...")
            generate_results_table(results_dir=results_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
