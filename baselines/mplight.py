"""MPLight baseline — pressure-based phase selection.

Simplified re-implementation of the MPLight controller:
  Wei et al., "PressLight: Learning Max Pressure Control to Coordinate
  Traffic Signals in Arterial Network", KDD 2019.

Core idea: for each green phase, compute the total pressure
(incoming waiting time − outgoing waiting time) on its green lanes,
then select the phase with the highest total pressure.

This version computes pressure purely from the observation vector
(no SUMO API calls), making it robust to connection lifecycle issues.
"""
from __future__ import annotations

import numpy as np


class MPLightAgent:
    """Per-intersection pressure-based controller.

    Interface compatible with IDQNAgent / TrfCoordAgent for train.py integration.
    """

    def __init__(self, traffic_signal, **_kwargs):
        """
        Args:
            traffic_signal: sumo_rl TrafficSignal object.
        """
        self.ts = traffic_signal
        self.n_phases = traffic_signal.num_green_phases
        self.lanes = traffic_signal.lanes

        # Pre-compute which incoming-lane indices are green for each phase.
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
        """Choose the phase with the highest total pressure.

        Pressure per phase = sum of total waiting times on green lanes.
        obs shape: (n_lanes, 80) — per-vehicle accumulated waiting times.
        """
        best_phase = 0
        best_pressure = -1.0
        for p in range(self.n_phases):
            green_idx = self._phase_green_lanes[p]
            if not green_idx:
                continue
            # Sum all waiting times on green lanes (each row = one lane)
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
