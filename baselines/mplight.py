"""MPLight baseline — pressure-based phase selection.

Re-implementation of MPLight (PressLight):
  Wei et al., "PressLight: Learning Max Pressure Control to Coordinate
  Traffic Signals in Arterial Network", KDD 2019.

For each green phase, compute:
  pressure(phase) = SUM(queue_on_green_incoming) − SUM(queue_on_green_outgoing)

where "queue" is the vehicle count (not waiting time). The phase with
the highest pressure is selected.  This creates a gradient that pushes
vehicles through the network rather than accumulating them.

Uses the TrafficSignal's built-in SUMO helpers to query per-lane vehicle
counts, avoiding raw TraCI calls that cause connection lifecycle bugs.
"""
from __future__ import annotations

import numpy as np


class MPLightAgent:
    """Per-intersection pressure-based controller (PressLight)."""

    def __init__(self, traffic_signal, **_kwargs):
        self.ts = traffic_signal
        self.n_phases = traffic_signal.num_green_phases
        self.lanes = traffic_signal.lanes
        self.out_lanes = traffic_signal.out_lanes

        # Pre-compute green incoming-lane indices per phase.
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

        # Pre-compute outgoing-lane vehicle counts per green phase.
        # Map each incoming-lane index to its connected outgoing lane IDs
        # via the SUMO controlled-links topology.
        self._phase_out_lanes: list[list[str]] = []
        links = traffic_signal.sumo.trafficlight.getControlledLinks(
            traffic_signal.id
        )
        in_to_out: dict[int, list[str]] = {}
        for idx in range(len(self.lanes)):
            if idx < len(links) and links[idx]:
                in_to_out[idx] = [lk[0][1] for lk in links[idx] if lk]
            else:
                in_to_out[idx] = []
        for p in range(self.n_phases):
            out_set: set[str] = set()
            for in_idx in self._phase_green_lanes[p]:
                out_set.update(in_to_out.get(in_idx, []))
            self._phase_out_lanes.append(list(out_set))

    def _lane_vehicle_count(self, lane_id: str) -> int:
        """Number of vehicles on a lane (incoming or outgoing)."""
        try:
            return self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
        except Exception:
            return 0

    def act(self, obs: np.ndarray, **_kwargs) -> int:
        """Select phase with highest pressure.

        pressure(p) = SUM_{green_in}(vehicles) − SUM_{green_out}(vehicles)

        obs shape: (n_lanes, 2) — [vehicle_count, waiting_time] per incoming lane.
        Column 0 is the vehicle count on each incoming lane.
        """
        best_phase = 0
        best_pressure = -1e9
        for p in range(self.n_phases):
            green_in = self._phase_green_lanes[p]
            if not green_in:
                continue
            # Incoming vehicle count on green lanes (from observation)
            in_count = float(obs[green_in, 0].sum())
            # Outgoing vehicle count on connected downstream lanes
            out_count = float(sum(
                self._lane_vehicle_count(lane)
                for lane in self._phase_out_lanes[p]
            ))
            pressure = in_count - out_count
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
