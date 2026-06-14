"""Metrics module — parses SUMO-RL's per-step CSV output.

Phase 0 acceptance gate depends on this being correct on `grid4x4`.

Primary metric: average waiting time (proxy for travel time from CSV).
Secondary: mean queue length, throughput (active vehicles), avg speed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EpisodeMetrics:
    """Per-episode traffic-signal control metrics."""

    avg_travel_time: float
    avg_waiting_time: float
    mean_queue_length: float
    throughput: int
    total_time_loss: float
    n_vehicles_completed: int = 0
    sim_time: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "avg_travel_time": self.avg_travel_time,
            "avg_waiting_time": self.avg_waiting_time,
            "mean_queue_length": self.mean_queue_length,
            "throughput": float(self.throughput),
            "total_time_loss": self.total_time_loss,
            "n_vehicles_completed": float(self.n_vehicles_completed),
            "sim_time": self.sim_time,
        }


def parse_csv(csv_path: str) -> Dict[str, float]:
    """Parse sumo_rl's per-step CSV → aggregate episode metrics.

    The CSV columns include:
      - step, system_total_stopped, system_total_waiting_time,
        system_mean_waiting_time, system_mean_speed,
        <agent>_stopped, <agent>_accumulated_waiting_time, <agent>_average_speed, ...

    Returns a dict with proxy metrics derived from per-step aggregates.
    """
    if not os.path.isfile(csv_path):
        return {
            "avg_travel_time": float("nan"),
            "avg_waiting_time": float("nan"),
            "mean_queue_length": float("nan"),
            "throughput": 0.0,
            "total_time_loss": float("nan"),
            "sim_time": 0.0,
        }

    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "avg_travel_time": float("nan"),
            "avg_waiting_time": float("nan"),
            "mean_queue_length": float("nan"),
            "throughput": 0.0,
            "total_time_loss": float("nan"),
            "sim_time": 0.0,
        }

    # Primary: mean waiting time across all steps (seconds)
    avg_waiting = float(df["system_mean_waiting_time"].mean()) if "system_mean_waiting_time" in df.columns else float("nan")

    # Queue: mean number of stopped vehicles across all steps
    mean_queue = float(df["system_total_stopped"].mean()) if "system_total_stopped" in df.columns else float("nan")

    # Throughput: approximate from active vehicle count at final step
    # (vehicles that entered the sim minus those still waiting)
    throughput = int(df["system_total_stopped"].iloc[-1]) if "system_total_stopped" in df.columns else 0

    # Total accumulated waiting time (sum over all steps)
    total_wt = float(df["system_total_waiting_time"].iloc[-1]) if "system_total_waiting_time" in df.columns else float("nan")

    # Avg speed (proxy for travel time — lower speed ≈ longer travel time)
    avg_speed = float(df["system_mean_speed"].mean()) if "system_mean_speed" in df.columns else float("nan")

    # Sim time
    sim_time = float(df["step"].iloc[-1]) if "step" in df.columns else 0.0

    # Approx travel time: we use total_waiting_time / n_stopped as a proxy
    # when tripinfo is unavailable.  Higher = worse.
    n_stopped = float(df["system_total_stopped"].mean()) if "system_total_stopped" in df.columns else 0.0
    avg_travel_time = total_wt / max(n_stopped, 1.0) if not np.isnan(total_wt) else float("nan")

    return {
        "avg_travel_time": avg_travel_time,
        "avg_waiting_time": avg_waiting,
        "mean_queue_length": mean_queue,
        "throughput": float(throughput),
        "total_time_loss": total_wt,  # proxy: total waiting = total time loss
        "sim_time": sim_time,
    }


def find_csv(run_dir: str) -> Optional[str]:
    """Find the sumo_rl CSV file in a run directory.

    sumo_rl writes to ``<out_csv_name><episode>ep.csv``.
    We look for ``csv1ep.csv`` (first episode) by convention.
    """
    # Direct lookup
    candidate = os.path.join(run_dir, "csv1ep.csv")
    if os.path.isfile(candidate):
        return candidate
    # Walk
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith("ep.csv"):
                return os.path.join(root, f)
    return None


def build_episode_metrics(run_dir: str) -> EpisodeMetrics:
    """Build a full EpisodeMetrics from a sumo_rl run directory."""
    csv = find_csv(run_dir)
    if csv is None:
        return EpisodeMetrics(
            avg_travel_time=float("nan"),
            avg_waiting_time=float("nan"),
            mean_queue_length=float("nan"),
            throughput=0,
            total_time_loss=float("nan"),
        )
    m = parse_csv(csv)
    return EpisodeMetrics(
        avg_travel_time=m["avg_travel_time"],
        avg_waiting_time=m["avg_waiting_time"],
        mean_queue_length=m["mean_queue_length"],
        throughput=int(m["throughput"]),
        total_time_loss=m["total_time_loss"],
        sim_time=m["sim_time"],
    )


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    statistic=np.mean,
) -> Dict[str, float]:
    """Stratified bootstrap 95% CI (rliable-friendly)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(0)
    n = values.size
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(values[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(np.mean(values)), "lo": float(lo), "hi": float(hi), "n": int(n)}


def aggregate_over_seeds(
    per_seed_metrics: List[EpisodeMetrics],
) -> Dict[str, Dict[str, float]]:
    """Compute mean ± 95% bootstrap CI for every metric across seeds."""
    keys = [
        "avg_travel_time",
        "avg_waiting_time",
        "mean_queue_length",
        "throughput",
        "total_time_loss",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array([getattr(m, k) for m in per_seed_metrics], dtype=float)
        out[k] = bootstrap_ci(arr)
    return out
