"""
Traffic Signal RL Environment
===============================
Gymnasium-compatible environment for adaptive signal control
using vehicle counts from the detection model.

State:  queue_lengths + phase_one_hot + timer + waiting_times
Action: 0=keep | 1=next_phase | 2=demand_switch
Reward: −queues − switch_penalty + service_bonus
"""

import logging
from typing import Dict
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
        self.n_app = rc["num_approaches"]
        self.n_phase = rc["num_phases"]
        self.max_steps = rc["max_steps"]
        self.min_green = rc["min_green"]
        self.max_green = rc["max_green"]
        self.arrivals = np.array(rc["arrival_rates"], dtype=np.float32)
        self.service = rc["service_rate"]
        self.switch_pen = rc.get("switch_penalty", -2.0)

        # Which approaches get green per phase
        self.green_map = {0: [0, 2], 1: [0, 2], 2: [1, 3], 3: [1, 3]}

        obs_dim = self.n_app + self.n_phase + 1 + self.n_app
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 200, (obs_dim,), np.float32)
        self._reset_state()

    def _reset_state(self):
        self.queues = np.zeros(self.n_app, np.float32)
        self.waits = np.zeros(self.n_app, np.float32)
        self.phase = self.timer = self.step_n = 0
        self.total_served = self.switches = 0

    def _obs(self):
        oh = np.zeros(self.n_phase, np.float32); oh[self.phase] = 1.
        return np.concatenate([self.queues, oh, [self.timer], self.waits])

    def _info(self):
        return dict(queues=self.queues.copy(), phase=self.phase,
                    timer=self.timer, served=self.total_served, switches=self.switches)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._reset_state()
        self.queues = self.np_random.poisson(3, self.n_app).astype(np.float32)
        return self._obs(), self._info()

    def step(self, action):
        self.step_n += 1; old = self.phase
        if action == 1 and self.timer >= self.min_green:
            self.phase = (self.phase + 1) % self.n_phase
        elif action == 2 and self.timer >= self.min_green:
            demands = [sum(self.queues[a] for a in self.green_map[p]) for p in range(self.n_phase)]
            self.phase = int(np.argmax(demands))
        if self.phase != old: self.timer = 0; self.switches += 1
        else: self.timer += 1
        if self.timer >= self.max_green:
            self.phase = (self.phase + 1) % self.n_phase; self.timer = 0; self.switches += 1

        self.queues += self.np_random.poisson(self.arrivals)
        served = 0.
        for a in self.green_map[self.phase]:
            s = min(self.queues[a], self.service); self.queues[a] -= s; served += s
        self.queues = np.maximum(self.queues, 0)
        self.total_served += served; self.waits += self.queues

        reward = -self.queues.sum() + 0.5 * served
        if self.phase != old: reward += self.switch_pen
        return self._obs(), float(reward), False, self.step_n >= self.max_steps, self._info()

    def set_detection_counts(self, counts: Dict[int, int]):
        """Override queues with real detection counts from the model."""
        for a, c in counts.items():
            if 0 <= a < self.n_app: self.queues[a] = float(c)

    def render(self):
        names = ["N-S Str", "N-S Left", "E-W Str", "E-W Left"]
        print(f"\nStep {self.step_n} | Phase: {names[self.phase]} (t={self.timer})")
        for i in range(self.n_app):
            print(f"  App {i}: {'█' * int(self.queues[i])} ({self.queues[i]:.0f})")
