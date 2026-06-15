"""Max-Pressure baseline controller.

Selects the green phase that maximises the total waiting time on its
green lanes — a classic non-RL traffic signal heuristic.

Paper: "Max pressure control of a network of intersections" (Varaiya, 2013).
"""
from __future__ import annotations

import numpy as np


class MaxPressureAgent:
    """Per-intersection Max-Pressure controller.

    Interface compatible with IDQNAgent / TrfCoordAgent for train.py integration.
    """

    def __init__(self, traffic_signal, **_kwargs):
        """
        Args:
            traffic_signal: sumo_rl TrafficSignal object (has .lanes, .green_phases,
                            .num_green_phases, .sumo).
        """
        self.ts = traffic_signal
        self.n_phases = traffic_signal.num_green_phases
        self.lanes = traffic_signal.lanes  # incoming lanes
        # Pre-compute which lanes are green for each phase.
        # Phase state string covers all controlled links; each lane has 3 sub-lanes.
        self._phase_green_lanes: list[list[int]] = []
        for phase in traffic_signal.green_phases:
            state = phase.state
            green_indices = []
            for lane_idx in range(len(self.lanes)):
                start = lane_idx * 3
                sub = state[start:start + 3]
                if any(c in ("G", "g") for c in sub):
                    green_indices.append(lane_idx)
            self._phase_green_lanes.append(green_indices)

    def act(self, obs: np.ndarray, **_kwargs) -> int:
        """Choose the phase with the highest total waiting time on green lanes."""
        # obs shape: (n_lanes, 80) — per-vehicle accumulated waiting times
        best_phase = 0
        best_pressure = -1.0
        for p in range(self.n_phases):
            green_idx = self._phase_green_lanes[p]
            if not green_idx:
                continue
            # Sum all waiting times on green lanes (non-zero entries)
            pressure = float(obs[green_idx].sum())
            if pressure > best_pressure:
                best_pressure = pressure
                best_phase = p
        return best_phase

    def on_episode_end(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
