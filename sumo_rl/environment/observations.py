"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s", datefmt="%H:%M:%S")

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        self.num_bins = 5


    def __call__(self) -> np.ndarray:
        observations = []
        lane_wt = {lane: [self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in self.ts.sumo.lane.getLastStepVehicleIDs(lane)] for lane in self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)}
        all_waiting_times = [wt for wt in lane_wt.values()]
        padded_state = []
        # max_len = max(len(inner) for inner in all_waiting_times) if all_waiting_times else 0
        for inner in all_waiting_times:
            padded_inner = inner + [0] * (80 - len(inner))  # Pad with zeros
            padded_state.append(padded_inner)
        # print('Observations: ', padded_state)
        return np.array(padded_state, dtype=np.float32)

        # lane_wt = {lane: [self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in self.ts.sumo.lane.getLastStepVehicleIDs(lane)] for lane in self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)}
        # all_waiting_times = [wt for lane in lane_wt.values() for wt in lane] #Flatten the list
        # min_wt = min(all_waiting_times) if all_waiting_times else 0
        # """NOTE: Returns the accumulated waiting time [s] within the previous time interval of default length 100 s. (length is configurable per option --waiting-time-memory given to the main application)"""
        # max_wt = max(all_waiting_times) if all_waiting_times else 100 # Default max if no data
        # bin_edges = np.linspace(min_wt, max_wt, self.num_bins + 1)
        # binned_data = create_waiting_time_bins(lane_wt, bin_edges)
        # observation = np.array([binned_data[lane] for lane in binned_data])
        # logging.debug(f'Observation: {observation}')
        # return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            # low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            # high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            low=np.zeros_like([0]*self.num_bins),
            high=np.array([100]*self.num_bins)
        )