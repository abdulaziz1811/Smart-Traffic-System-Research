import logging
import numpy as np
from typing import Optional

log = logging.getLogger("TrafficSystem")


class LocalIntersectionAgent:
    """
    Local intersection agent that wraps the RL model and Environment.
    Supports Emergency Override (Ambulance), Intention Prediction, and reporting anomalies.
    """

    # Green seconds without any vehicle leaving before a lane is reported
    ANOMALY_THRESHOLD = 10
    # Ignore lanes with fewer waiting vehicles than this
    ANOMALY_MIN_QUEUE = 3.0

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

        # Anomaly Detection tracking: consecutive GREEN seconds in which a
        # lane had vehicles waiting but none of them moved
        self.locked_queues_duration = np.zeros(self.num_lanes, dtype=int)

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
        self.env.hold_green = False

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
            at_target = self.current_phase == target_phase

            # Keep the ambulance's green beyond max_green while it is needed
            self.env.hold_green = at_target

            if self.env.action_mode == "free":
                return target_phase  # jump straight to the ambulance's phase
            # Cyclic order: extend if already green, otherwise advance towards it
            return 0 if at_target else 1

        # Normal operation via AI Model
        action, _ = self.rl_model.predict(current_obs, deterministic=True)
        return action

    def step(self, action):
        """Execute the action in the environment and check anomalies."""
        prev_queues = self.queues.copy()
        obs, reward, done, truncated, info = self.env.step(action)
        self.check_anomalies(prev_queues)
        return obs, reward, done, truncated, info

    def check_anomalies(self, prev_queues):
        """
        Monitor for lanes that do not move although they have GREEN.

        A red lane that does not move is normal, so only lanes that were
        served in the last step are judged. With vehicles leaving at
        ~0.6 veh/s a green lane almost always shrinks; if it keeps getting
        green without shrinking, something (accident, broken-down vehicle,
        faulty signal head) is blocking it.
        """
        for lane in self.env.last_active_lanes:
            waiting = prev_queues[lane] >= self.ANOMALY_MIN_QUEUE
            moved = self.queues[lane] < prev_queues[lane]
            if waiting and not moved:
                self.locked_queues_duration[lane] += 1
            elif moved:
                self.locked_queues_duration[lane] = 0

    def get_anomalies(self):
        """Return anomalous lanes which have been locked for too long."""
        return np.where(self.locked_queues_duration > self.ANOMALY_THRESHOLD)[0]
