"""
Traffic Signal RL Environment V6
=================================
V5 fixed the modelling issues that made the V4 comparison invalid:

1. Clearance (yellow + all-red) lost time: every switch is followed by
   `yellow_time` seconds in which no lane is served, and the first
   `startup_lost_time` seconds of a new green discharge nobody. (In V4
   switching was free, and a fixed timer with 6s green matched the trained
   agents.) service_rate 0.5 veh/s and 2 s start-up loss match the
   saturation flow measured in SUMO (1800 veh/h per lane).
2. Real action modes:
   - "cyclic": Discrete(2) -> 0 = extend, 1 = advance to next phase in cycle.
   - "free":   Discrete(n_phase) -> index of the phase that should be green
               (same as current phase = extend).
3. Common random numbers: every step draws the same amount of randomness,
   so all policies see identical arrivals for a given seed.

V6 adds what the agent needs to learn a good policy:

4. Waiting time per lane in the observation (`red_wait`: seconds a lane has
   had vehicles without green) and a clearance indicator.
5. Decision points: `is_decision_point()` tells whether the agent's action
   can change anything this second (not during clearance or min green).
   The training wrapper in src/agents.py only queries the agent there.
6. `reward_type`: "shaped" (original multi-term reward) or "queue"
   (negative total queue = total delay per second, the evaluation metric).
7. `max_wait` in info: worst lane waiting time, to check fairness.

Observation Layout (31-dim):
   [0:8]   Queue lengths per lane (raw count)
   [8:12]  Current phase one-hot encoding
   [12]    Normalized green timer (0.0 to 1.0)
   [13]    Next phase (in cycle) queue density (normalized)
   [14:22] Queue trend per lane (avg change per step, can be negative)
   [22:30] Waiting time per lane (red_wait / 60 s)
   [30]    Clearance progress (1.0 = yellow just started, 0.0 = green)

Phase Mapping:
   Phase 0: North/South Straight  (lanes 0, 2)
   Phase 1: North/South Left      (lanes 1, 3)
   Phase 2: East/West Straight    (lanes 4, 6)
   Phase 3: East/West Left        (lanes 5, 7)
"""

import logging
from typing import Optional
from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces, Env
except ImportError:
    import gym
    from gym import spaces, Env

log = logging.getLogger("traffic")

ACTION_MODES = ("cyclic", "free")
REWARD_TYPES = ("shaped", "queue")
WAIT_SCALE = 60.0  # seconds, normalization of red_wait in the observation

# Standard 4-phase cycle: phase -> green lanes
GREEN_MAP = {
    0: [0, 2],   # North/South Straight
    1: [1, 3],   # North/South Left
    2: [4, 6],   # East/West Straight
    3: [5, 7],   # East/West Left
}


def build_observation(queues, phase, timer, max_green, n_phase, green_map, queue_history,
                      red_wait, clearance_left=0, yellow_time=0):
    """
    Build the 31-dim observation vector.

    Shared by the simulator and the real video pipeline (track_video.py) so
    the agent sees identically-computed features in both places.
    """
    queues = np.asarray(queues, dtype=np.float32)

    # Current phase as one-hot vector
    phase_oh = np.zeros(n_phase, dtype=np.float32)
    phase_oh[phase] = 1.0

    # Normalized green timer (0.0 = just switched, 1.0 = about to force-switch)
    norm_timer = timer / max(max_green, 1)

    # Next phase queue density (how busy are the lanes that go next)
    next_lanes = green_map[(phase + 1) % n_phase]
    next_density = sum(queues[l] for l in next_lanes) / 50.0

    # Queue trend: average change per step over the history window
    # Positive = lane is getting more congested, negative = lane is clearing
    if len(queue_history) >= 2:
        trend = (queues - queue_history[0]) / (len(queue_history) - 1)
    else:
        trend = np.zeros_like(queues)

    # Waiting time per lane and clearance progress
    waiting = np.asarray(red_wait, dtype=np.float32) / WAIT_SCALE
    clearance = clearance_left / max(yellow_time, 1)

    obs = np.concatenate([
        queues,                 # [0:8]   per-lane queue lengths
        phase_oh,               # [8:12]  current phase
        [norm_timer],           # [12]    green timer progress
        [next_density],         # [13]    next phase demand
        trend,                  # [14:22] queue growth direction
        waiting,                # [22:30] how long each lane has waited
        [clearance],            # [30]    yellow / all-red in progress
    ])
    return obs.astype(np.float32)


def infer_action_mode(model, n_phase=4):
    """Return 'free' if a trained model picks phases directly, else 'cyclic'."""
    n = getattr(model.action_space, "n", 2)
    return "free" if int(n) == n_phase and n_phase > 2 else "cyclic"


class TrafficSignalEnv(Env): # type: ignore
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: dict, action_mode: Optional[str] = None):
        super().__init__()
        rc = cfg["rl"]

        # --- Core dimensions ---
        self.n_app = rc["num_approaches"]
        self.n_phase = rc["num_phases"]

        # --- Timing constraints ---
        self.max_steps = rc["max_steps"]
        self.min_green = rc["min_green"]
        self.max_green = rc["max_green"]
        self.yellow_time = int(rc.get("yellow_time", 0))
        self.startup_lost_time = int(rc.get("startup_lost_time", 0))

        # --- Traffic flow parameters ---
        self.arr_low = rc.get("arrival_rate_low", 0.02)
        self.arr_high = rc.get("arrival_rate_high", 0.15)
        self.service = rc["service_rate"]
        self.switch_pen = rc.get("switch_penalty", -5.0)
        self.reward_type = rc.get("reward_type", "shaped")
        if self.reward_type not in REWARD_TYPES:
            raise ValueError(f"reward_type must be one of {REWARD_TYPES}, got {self.reward_type!r}")

        # --- Trend history depth ---
        self.trend_window = max(int(rc.get("trend_window", 4)), 2)

        # --- Action mode ---
        self.action_mode = action_mode or rc.get("action_mode", "cyclic")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {self.action_mode!r}")

        # --- Phase-to-lane mapping (standard 4-phase cycle) ---
        self.green_map = GREEN_MAP

        # --- Observation space (31-dim) ---
        # queues(8) + phase_one_hot(4) + timer(1) + next_density(1) + trend(8)
        # + waiting time(8) + clearance(1)
        obs_dim = self.n_app + self.n_phase + 1 + 1 + self.n_app + self.n_app + 1
        self.observation_space = spaces.Box(
            low=-500.0, high=500.0, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space ---
        # cyclic: 0 = extend, 1 = next phase | free: target phase index
        n_actions = 2 if self.action_mode == "cyclic" else self.n_phase
        self.action_space = spaces.Discrete(n_actions)

        # --- External controls (used by LocalIntersectionAgent / demo) ---
        self.hold_green = False        # emergency pre-emption: ignore max_green
        self.blocked_lanes = set()     # simulated incident: lane gets no service

        # --- Internal state ---
        self.queues = np.zeros(self.n_app, dtype=np.float32)
        self.waits = np.zeros(self.n_app, dtype=np.float32)
        self.red_wait = np.zeros(self.n_app, dtype=np.float32)
        self.max_wait = 0.0
        self.phase = 0
        self.timer = 0
        self.clearance_left = 0
        self.step_n = 0
        self.arrivals = np.zeros(self.n_app, dtype=np.float32)
        self.switches = 0
        self.total_served = 0.0
        self.last_active_lanes = []
        self.yellow_lanes = []
        self.queue_history = deque(maxlen=self.trend_window)

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        # Randomize per-lane arrival rates for this episode
        # (always drawn so the random stream is identical with or without overrides)
        self.arrivals = self.np_random.uniform(
            self.arr_low, self.arr_high, size=self.n_app
        ).astype(np.float32)
        if options.get("arrivals") is not None:
            self.arrivals = np.asarray(options["arrivals"], dtype=np.float32)

        self.queues = np.zeros(self.n_app, dtype=np.float32)
        self.waits = np.zeros(self.n_app, dtype=np.float32)
        self.red_wait = np.zeros(self.n_app, dtype=np.float32)
        self.max_wait = 0.0
        self.phase = 0
        self.timer = 0
        self.clearance_left = 0
        self.step_n = 0
        self.switches = 0
        self.total_served = 0.0
        self.last_active_lanes = []
        self.yellow_lanes = []
        self._startup = False
        self.hold_green = False
        self.blocked_lanes = set()

        # Pre-fill trend history with zeros so it is valid from step 1
        self.queue_history.clear()
        for _ in range(self.trend_window):
            self.queue_history.append(np.zeros(self.n_app, dtype=np.float32))

        return self._obs(), {}

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    def set_arrival_range(self, low, high):
        """Change the traffic level for the next episodes (curriculum learning)."""
        self.arr_low, self.arr_high = float(low), float(high)

    def is_decision_point(self) -> bool:
        """True if the next action can change the signal (not in clearance / min green)."""
        return self.clearance_left == 0 and self.timer >= self.min_green

    def noop_action(self):
        """Action that keeps the current phase in either action mode."""
        return self.phase if self.action_mode == "free" else 0

    def _target_phase(self, action) -> Optional[int]:
        """Decode an action into the phase to switch to (None = extend)."""
        a = int(np.asarray(action).reshape(-1)[0])
        if self.action_mode == "free":
            return None if a == self.phase else a % self.n_phase
        return (self.phase + 1) % self.n_phase if a == 1 else None

    def _switch_to(self, phase):
        self.yellow_lanes = list(self.green_map[self.phase])  # shown yellow during clearance
        self.phase = phase
        self.timer = 0
        self.switches += 1
        self.clearance_left = self.yellow_time

    def step(self, action):
        self.step_n += 1
        old_phase = self.phase
        in_clearance = self.clearance_left > 0
        switch_requested = False

        if in_clearance:
            # Yellow / all-red: no lane is served and actions are ignored
            self.clearance_left -= 1
        else:
            # --- Enforce minimum green time, then execute agent action ---
            target = self._target_phase(action) if self.timer >= self.min_green else None
            if target is not None:
                self._switch_to(target)
                switch_requested = True
            else:
                self.timer += 1
                # --- Enforce maximum green time (unless pre-empted) ---
                if self.timer >= self.max_green and not self.hold_green:
                    self._switch_to((self.phase + 1) % self.n_phase)

        did_switch = (self.phase != old_phase)

        # The step in which a switch is decided is the last green second of
        # the old phase; the clearance interval follows it.
        if in_clearance:
            active_lanes = []
        else:
            active_lanes = self.green_map[old_phase if did_switch else self.phase]
        self.last_active_lanes = list(active_lanes)
        # Start-up lost time: the first seconds of a new green discharge nobody
        self._startup = (not in_clearance) and (not did_switch) and self.timer <= self.startup_lost_time

        # --- Traffic simulation (arrivals + service) ---
        cars_in_green, served = self._simulate_traffic(active_lanes)
        self.total_served += served

        # --- Waiting time: seconds each lane has had vehicles without being served ---
        served_lanes = [l for l in active_lanes if l not in self.blocked_lanes]
        has_cars = self.queues >= 1.0
        self.red_wait = np.where(has_cars, self.red_wait + 1.0, 0.0).astype(np.float32)
        self.red_wait[served_lanes] = 0.0
        self.max_wait = max(self.max_wait, float(self.red_wait.max()))

        # --- Record queue snapshot for trend computation ---
        self.queue_history.append(self.queues.copy())

        # --- Compute reward ---
        extended = (not in_clearance) and (not switch_requested)
        reward = self._compute_reward(extended, switch_requested, cars_in_green, served)

        terminated = False
        truncated = self.step_n >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    #  Traffic model (overridden by the SUMO environment)
    # ------------------------------------------------------------------

    def _simulate_traffic(self, active_lanes):
        """
        Advance traffic by one second with `active_lanes` green.
        Returns (vehicles on the green lanes before service, vehicles served).
        """
        # Random draws: fixed amount every step (common random numbers)
        new_cars = self.np_random.poisson(self.arrivals)
        flows = self.service * self.np_random.uniform(0.8, 1.2, size=self.n_app)
        self.queues += new_cars
        cars_in_green = sum(self.queues[l] for l in active_lanes)

        served = 0.0
        for lane in active_lanes:
            if lane in self.blocked_lanes or self._startup:
                continue
            s = min(self.queues[lane], flows[lane])
            self.queues[lane] -= s
            served += s
            if self.queues[lane] > 0:
                self.waits[lane] *= 0.95
            else:
                self.waits[lane] = 0.0

        self.queues = np.maximum(self.queues, 0.0)
        self.waits += self.queues
        return cars_in_green, served

    # ------------------------------------------------------------------
    #  Reward: proportional shaping
    # ------------------------------------------------------------------

    def _compute_reward(self, extended, switch_requested, cars_in_green, served):
        """
        Multi-component reward with proportional penalty scaling.

        Components:
          1. queue_cost        -- quadratic penalty on total queue length
          2. wait_cost         -- linear penalty on accumulated wait times
          3. wasted_green      -- proportional to OTHER lanes demand
          4. premature_switch  -- proportional to active lane occupancy
          5. service_reward    -- per-vehicle bonus for clearing queues

        During clearance no action-based penalty is applied; the cost of the
        lost time shows up naturally through the queue and wait costs.

        reward_type "queue": negative total queue (vehicles waiting this second,
        i.e. total delay), scaled by 1/10 -- exactly the evaluation metric.
        """
        if self.reward_type == "queue":
            return -float(np.sum(self.queues)) / 10.0

        # 1. Base cost: penalize long queues (quadratic = punish extremes more)
        queue_cost = -np.sum(self.queues ** 2) / 100.0

        # 2. Wait cost: penalize accumulated waiting (encourages throughput)
        wait_cost = -np.sum(self.waits) / 500.0

        # 3. Wasted green: extending on empty lane while others wait
        #    Penalty scales with how many vehicles are stuck on OTHER lanes.
        #    Empty lane + empty intersection = small penalty (-1.0)
        #    Empty lane + 20 cars waiting elsewhere = large penalty (-3.0)
        wasted_green = 0.0
        if extended and cars_in_green < 1.0:
            waiting_others = max(np.sum(self.queues) - cars_in_green, 0.0)
            wasted_green = -1.0 * (1.0 + waiting_others / 10.0)

        # 4. Premature switch: switching while active lane still has vehicles
        #    Penalty scales with how many vehicles remain (interruption cost).
        #    2 cars remaining = small penalty (-0.4)
        #    15 cars remaining = large penalty (-3.0)
        premature_switch = 0.0
        if switch_requested and cars_in_green > 2.0:
            premature_switch = -1.0 * (cars_in_green / 5.0)

        # 5. Service reward: bonus per vehicle cleared
        service_reward = served * 3.0

        return (queue_cost + wait_cost + wasted_green
                + premature_switch + service_reward)

    # ------------------------------------------------------------------
    #  Observation
    # ------------------------------------------------------------------

    def _obs(self):
        return build_observation(
            self.queues, self.phase, self.timer, self.max_green,
            self.n_phase, self.green_map, self.queue_history,
            self.red_wait, self.clearance_left, self.yellow_time,
        )

    # ------------------------------------------------------------------
    #  Info dict
    # ------------------------------------------------------------------

    def _info(self):
        return {
            "switches": self.switches,
            "served": self.total_served,
            "avg_queue": float(np.mean(self.queues)),
            "in_clearance": self.clearance_left > 0,
            "max_wait": self.max_wait,
        }
