"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

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

    def __call__(self) -> np.ndarray:
        # observations = []
        # for lane_id in self.ts.lanes:
        #     # vehicle_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
        #     # observations.append(vehicle_count)
        #     waiting_time = self.ts.sumo.lane.getWaitingTime(lane_id)
        #     observations.append(waiting_time)
        # # print('Observations: ', observations)
        # return np.array(observations, dtype=np.float32)

        # for lane in self.ts.sumo.trafficlight.getControlledLanes(self.ts.id):
        #     veh_list.append(self.ts.sumo.lane.getLastStepVehicleIDs(lane))
        # print(veh_list) 
        # total_vehicles = len(veh_list)
        # total_waiting_time = \
        #     sum(self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in veh_list)
        # observation.append([total_vehicles, total_waiting_time])
        # print(observation)
        # return np.array(observation, dtype=np.float32)
        lane_wt = {lane: [self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in self.ts.sumo.lane.getLastStepVehicleIDs(lane)] for lane in self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)}
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            # low=np.zeros_like([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            # high=np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
        )