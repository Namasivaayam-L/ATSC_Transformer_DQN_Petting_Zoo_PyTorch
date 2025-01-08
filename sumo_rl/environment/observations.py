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
        for lane_id in self.ts.lanes:
            # vehicle_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
            # observations.append(vehicle_count)
            waiting_time = self.ts.sumo.lane.getWaitingTime(lane_id)
            observations.append(waiting_time)
        print('Observations: ', observations)
        return np.array(observations, dtype=np.float32)

        # # for lane in self.ts.sumo.trafficlight.getControlledLanes(self.ts.id):
        # #     veh_list.append(self.ts.sumo.lane.getLastStepVehicleIDs(lane))
        # # print(veh_list) 
        # # total_vehicles = len(veh_list)
        # # total_waiting_time = \
        # #     sum(self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in veh_list)
        # # observation.append([total_vehicles, total_waiting_time])
        # # print(observation)
        # # return np.array(observation, dtype=np.float32)
        # # """Return the default observation."""
        # # phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # # density = self.ts.get_lanes_density()
        # # queue = self.ts.get_lanes_queue()
        # # observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        # # def create_waiting_time_bins(lane_wt, bin_edges):
        # #     """Creates bins for waiting times.

        # #     Args:
        # #         lane_wt (dict): Lane waiting times.
        # #         bin_edges (list/np.array): Fixed bin edges.

        # #     Returns:
        # #         dict: Binned waiting times.
        # #         Raises ValueError if bin_edges is None or empty.
        # #     """

        # #     if bin_edges is None or not isinstance(bin_edges, (list, np.ndarray)) or len(bin_edges) == 0:
        # #         raise ValueError("bin_edges must be provided and non-empty.")
            
        # #     bin_edges = np.array(bin_edges) #Convert to numpy array for efficiency
        # #     num_bins = len(bin_edges) - 1 #Number of bins is always len(edges) - 1

        # #     binned_lane_wt = {}
        # #     for lane_name, waiting_times in lane_wt.items():
        # #         if not waiting_times:
        # #             binned_lane_wt[lane_name] = [0] * num_bins # Correct way to pad
        # #             continue
        # #         binned_wt, _ = np.histogram(waiting_times, bins=bin_edges)
        # #         binned_lane_wt[lane_name] = binned_wt.tolist()

        # #         # Sanity Check (for extra safety during development):
        # #         if len(binned_lane_wt[lane_name]) != num_bins:
        # #             raise ValueError(f"Bin length mismatch for lane {lane_name}. Expected {num_bins}, got {len(binned_lane_wt[lane_name])}. Check bin_edges.")

        # #     return binned_lane_wt
        # # lane_wt = {lane: [self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in self.ts.sumo.lane.getLastStepVehicleIDs(lane)] for lane in self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)}
        # # all_waiting_times = [wt for lane in lane_wt.values() for wt in lane] #Flatten the list
        # # min_wt = min(all_waiting_times) if all_waiting_times else 0
        # # """NOTE: Returns the accumulated waiting time [s] within the previous time interval of default length 100 s. (length is configurable per option --waiting-time-memory given to the main application)"""
        # # max_wt = max(all_waiting_times) if all_waiting_times else 100 # Default max if no data
        # # bin_edges = np.linspace(min_wt, max_wt, self.num_bins + 1)
        # # binned_data = create_waiting_time_bins(lane_wt, bin_edges)
        # # observation = np.array([binned_data[lane] for lane in binned_data])
        # # logging.debug(f'Observation: {observation}')
        # return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            # low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            # high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            low=np.zeros_like([0]*self.num_bins),
            high=np.array([100]*self.num_bins)
        )