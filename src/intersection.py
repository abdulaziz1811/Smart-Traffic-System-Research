import logging
import numpy as np
from collections import deque
from typing import Optional

log = logging.getLogger("TrafficSystem")


class LocalIntersectionAgent:
    """
    Local intersection agent that wraps the RL model and Environment.
    Supports Emergency Override (Ambulance), Intention Prediction, and reporting anomalies.
    """

    def __init__(self, intersection_id, env, rl_model):
        self.id = intersection_id
        self.env = env
        self.rl_model = rl_model

        # State mapping
        self.num_phases = env.n_phase
        self.num_lanes = env.n_app

        # Emergency Override state
        self.emergency_mode = False
        self.ambulance_lane = -1
        self.ambulance_intention: Optional[str] = None  # 'straight' or 'left'

        # Phase mapping definition: Phase -> Lanes
        self.green_map = env.green_map

        # Anomaly Detection tracking
        self.locked_queues_duration = np.zeros(self.num_lanes, dtype=int)
        self.history_queues = deque(maxlen=5)

    @property
    def queues(self):
        return self.env.queues

    @property
    def current_phase(self):
        return self.env.phase

    def detect_ambulance(self, lane, is_blinking_left_indicator=False):
        """
        Triggered when an ambulance is detected by the vision module.
        Intention is predicted based on the lane the ambulance is in and its indicators.
        """
        self.emergency_mode = True
        self.ambulance_lane = lane

        # Intention Prediction
        if lane % 2 == 1 or is_blinking_left_indicator:
            self.ambulance_intention = "left"
        else:
            self.ambulance_intention = "straight"

        log.warning(
            f"[{self.id}] EMERGENCY DETECTED! Ambulance in lane {lane}. Intention: {self.ambulance_intention}"
        )

    def clear_ambulance(self):
        """Return to normal operation after the ambulance has passed."""
        if self.emergency_mode:
            log.info(f"[{self.id}] EMERGENCY CLEARED. Returning to normal RL control.")
            self.emergency_mode = False
            self.ambulance_lane = -1
            self.ambulance_intention = None

    def get_target_phase_for_lane(self, lane):
        """Find which phase turns the specified lane green."""
        for phase, lanes in self.green_map.items():
            if lane in lanes:
                return phase
        return 0

    def get_action(self, current_obs):
        """
        Decide the next action. If emergency mode is active, override RL.
        Otherwise, query the PPO model.
        """
        if self.emergency_mode:
            target_phase = self.get_target_phase_for_lane(self.ambulance_lane)

            # If we are already in the target phase, extend it.
            if self.current_phase == target_phase:
                return 0  # Extend
            else:
                return 1  # Switch to next phase to cycle towards target

        # Normal operation via AI Model
        action, _ = self.rl_model.predict(current_obs, deterministic=True)
        return action

    def step(self, action):
        """Execute the action in the environment and check anomalies."""
        obs, reward, done, truncated, info = self.env.step(action)
        self.check_anomalies()
        return obs, reward, done, truncated, info

    def check_anomalies(self):
        """Monitor for completely stuck traffic to warn the Central Supervisor."""
        self.history_queues.append(self.queues.copy())

        if len(self.history_queues) == 5:
            # If queues haven't decreased over 5 ticks (and are high)
            stuck = np.all(
                np.array(self.history_queues) == self.history_queues[0], axis=0
            )
            high_queue = self.queues > 10
            anomalous_lanes = np.where(stuck & high_queue)[0]

            for lane in anomalous_lanes:
                self.locked_queues_duration[lane] += 1

            # Reset moving lanes
            moving_lanes = np.where(~stuck)[0]
            for lane in moving_lanes:
                self.locked_queues_duration[lane] = 0

    def get_anomalies(self):
        """Return anomalous lanes which have been locked for too long."""
        threshold = 10  # Arbitrary time steps
        return np.where(self.locked_queues_duration > threshold)[0]
