"""
Traffic Signal RL Environment V4 (Enhanced Observation + Proportional Rewards)
===============================================================================
Builds on V3 cyclic logic with two key improvements:

1. Richer Observation (22-dim instead of 14-dim):
   - Added queue trend vector (8-dim) showing whether each lane is
     growing or shrinking. Gives the agent temporal context beyond
     a single snapshot, enabling proactive rather than reactive control.

2. Proportional Reward Shaping:
   - Wasted green penalty now scales with demand on OTHER lanes
     (opportunity cost: wasting green is worse when others are waiting).
   - Premature switch penalty scales with remaining vehicles in active
     lane (interruption cost: switching mid-service is worse with more cars).

Observation Layout (22-dim):
   [0:8]   Queue lengths per lane (raw count)
   [8:12]  Current phase one-hot encoding
   [12]    Normalized green timer (0.0 to 1.0)
   [13]    Next phase queue density (normalized)
   [14:22] Queue trend per lane (current - previous, can be negative)

Action Space: Discrete(2)
   0 = Extend current green phase
   1 = Advance to next phase in fixed cycle (0 -> 1 -> 2 -> 3 -> 0)

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
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

log = logging.getLogger("traffic")


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: dict):
        super().__init__()
        rc = cfg["rl"]

        # --- Core dimensions ---
        self.n_app = rc["num_approaches"]
        self.n_phase = rc["num_phases"]

        # --- Timing constraints ---
        self.max_steps = rc["max_steps"]
        self.min_green = rc["min_green"]
        self.max_green = rc["max_green"]

        # --- Traffic flow parameters ---
        self.arr_low = rc.get("arrival_rate_low", 0.02)
        self.arr_high = rc.get("arrival_rate_high", 0.15)
        self.service = rc["service_rate"]
        self.switch_pen = rc.get("switch_penalty", -5.0)

        # --- Trend history depth ---
        self.trend_window = rc.get("trend_window", 4)

        # --- Phase-to-lane mapping (standard 4-phase cycle) ---
        self.green_map = {
            0: [0, 2],   # North/South Straight
            1: [1, 3],   # North/South Left
            2: [4, 6],   # East/West Straight
            3: [5, 7],   # East/West Left
        }

        # --- Observation space (22-dim) ---
        # queues(8) + phase_one_hot(4) + timer(1) + next_density(1) + trend(8)
        obs_dim = self.n_app + self.n_phase + 1 + 1 + self.n_app
        self.observation_space = spaces.Box(
            low=-500.0, high=500.0, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space: binary (extend or switch) ---
        self.action_space = spaces.Discrete(2)

        # --- Internal state ---
        self.queues = np.zeros(self.n_app, dtype=np.float32)
        self.waits = np.zeros(self.n_app, dtype=np.float32)
        self.phase = 0
        self.timer = 0
        self.step_n = 0
        self.arrivals = np.zeros(self.n_app, dtype=np.float32)
        self.switches = 0
        self.total_served = 0.0
        self.queue_history = deque(maxlen=self.trend_window)

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomize per-lane arrival rates for this episode
        self.arrivals = self.np_random.uniform(
            self.arr_low, self.arr_high, size=self.n_app
        ).astype(np.float32)

        self.queues = np.zeros(self.n_app, dtype=np.float32)
        self.waits = np.zeros(self.n_app, dtype=np.float32)
        self.phase = 0
        self.timer = 0
        self.step_n = 0
        self.switches = 0
        self.total_served = 0.0

        # Pre-fill trend history with zeros so it is valid from step 1
        self.queue_history.clear()
        for _ in range(self.trend_window):
            self.queue_history.append(np.zeros(self.n_app, dtype=np.float32))

        return self._obs(), {}

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    def step(self, action):
        self.step_n += 1
        old_phase = self.phase

        # --- Enforce minimum green time ---
        if self.timer < self.min_green:
            action = 0

        # --- Execute agent action ---
        if action == 1:
            # Advance to next phase in cycle (0 -> 1 -> 2 -> 3 -> 0)
            self.phase = (self.phase + 1) % self.n_phase
            self.timer = 0
            self.switches += 1
        else:
            self.timer += 1

        # --- Enforce maximum green time ---
        if self.timer >= self.max_green:
            self.phase = (self.phase + 1) % self.n_phase
            self.timer = 0
            self.switches += 1

        did_switch = (self.phase != old_phase)

        # --- Traffic simulation: arrivals ---
        new_cars = self.np_random.poisson(self.arrivals)
        self.queues += new_cars

        # --- Traffic simulation: service (green lanes clear vehicles) ---
        active_lanes = self.green_map[old_phase if did_switch else self.phase]
        cars_in_green = sum(self.queues[l] for l in active_lanes)

        served = 0.0
        for lane in active_lanes:
            flow = self.service * self.np_random.uniform(0.8, 1.2)
            s = min(self.queues[lane], flow)
            self.queues[lane] -= s
            served += s
            if self.queues[lane] > 0:
                self.waits[lane] *= 0.95
            else:
                self.waits[lane] = 0.0

        self.queues = np.maximum(self.queues, 0.0)
        self.waits += self.queues
        self.total_served += served

        # --- Record queue snapshot for trend computation ---
        self.queue_history.append(self.queues.copy())

        # --- Compute reward ---
        reward = self._compute_reward(action, cars_in_green, served)

        terminated = False
        truncated = self.step_n >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    #  Reward: proportional shaping
    # ------------------------------------------------------------------

    def _compute_reward(self, action, cars_in_green, served):
        """
        Multi-component reward with proportional penalty scaling.

        Components:
          1. queue_cost        -- quadratic penalty on total queue length
          2. wait_cost         -- linear penalty on accumulated wait times
          3. wasted_green      -- proportional to OTHER lanes demand
          4. premature_switch  -- proportional to active lane occupancy
          5. service_reward    -- per-vehicle bonus for clearing queues
        """
        # 1. Base cost: penalize long queues (quadratic = punish extremes more)
        queue_cost = -np.sum(self.queues ** 2) / 100.0

        # 2. Wait cost: penalize accumulated waiting (encourages throughput)
        wait_cost = -np.sum(self.waits) / 500.0

        # 3. Wasted green: extending on empty lane while others wait
        #    Penalty scales with how many vehicles are stuck on OTHER lanes.
        #    Empty lane + empty intersection = small penalty (-1.0)
        #    Empty lane + 20 cars waiting elsewhere = large penalty (-3.0)
        wasted_green = 0.0
        if action == 0 and cars_in_green < 1.0:
            waiting_others = max(np.sum(self.queues) - cars_in_green, 0.0)
            wasted_green = -1.0 * (1.0 + waiting_others / 10.0)

        # 4. Premature switch: switching while active lane still has vehicles
        #    Penalty scales with how many vehicles remain (interruption cost).
        #    2 cars remaining = small penalty (-0.4)
        #    15 cars remaining = large penalty (-3.0)
        premature_switch = 0.0
        if action == 1 and cars_in_green > 2.0:
            premature_switch = -1.0 * (cars_in_green / 5.0)

        # 5. Service reward: bonus per vehicle cleared
        service_reward = served * 3.0

        return (queue_cost + wait_cost + wasted_green
                + premature_switch + service_reward)

    # ------------------------------------------------------------------
    #  Observation
    # ------------------------------------------------------------------

    def _obs(self):
        # Current phase as one-hot vector
        phase_oh = np.zeros(self.n_phase, dtype=np.float32)
        phase_oh[self.phase] = 1.0

        # Normalized green timer (0.0 = just switched, 1.0 = about to force-switch)
        norm_timer = self.timer / max(self.max_green, 1)

        # Next phase queue density (how busy are the lanes that go next)
        next_p = (self.phase + 1) % self.n_phase
        next_lanes = self.green_map[next_p]
        next_density = sum(self.queues[l] for l in next_lanes) / 50.0

        # Queue trend: difference between current and recent past
        # Positive = lane is getting more congested
        # Negative = lane is clearing
        if len(self.queue_history) >= 2:
            trend = self.queues - self.queue_history[-2]
        else:
            trend = np.zeros(self.n_app, dtype=np.float32)

        obs = np.concatenate([
            self.queues,            # [0:8]   per-lane queue lengths
            phase_oh,               # [8:12]  current phase
            [norm_timer],           # [12]    green timer progress
            [next_density],         # [13]    next phase demand
            trend,                  # [14:22] queue growth direction
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    #  Info dict
    # ------------------------------------------------------------------

    def _info(self):
        return {
            "switches": self.switches,
            "served": self.total_served,
            "avg_queue": float(np.mean(self.queues)),
        }