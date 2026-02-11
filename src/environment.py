"""
Traffic Signal RL Environment (Realistic Cycle Logic)
=====================================================
Philosophy:
The agent acts as an adaptive timer. It must respect the cycle order
(0 -> 1 -> 2 -> 3 -> 0) but decides exactly how long each phase lasts
based on the queue length.

Key Logic:
- No random phase jumping (ensures synchronization).
- Penalties for holding green on empty roads (Wasted Green).
- Rewards for clearing the queue fully before switching.
"""

import logging
from typing import Dict, Optional
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
        
        # Traffic parameters
        self.arr_low = rc.get("arrival_rate_low", 0.02)
        self.arr_high = rc.get("arrival_rate_high", 0.15)
        self.service = rc["service_rate"]
        self.switch_pen = rc.get("switch_penalty", -5.0)

        # Phase Mapping (Standard 4-Phase Cycle)
        # 0: North/South Straight
        # 1: North/South Left
        # 2: East/West Straight
        # 3: East/West Left
        self.green_map = {
            0: [0, 2],
            1: [1, 3],
            2: [4, 6],
            3: [5, 7]
        }

        # Observation Space
        # [Queues(8), Phase_OH(4), Normalized_Timer(1), Next_Phase_Queue_Density(1)]
        obs_dim = self.n_app + self.n_phase + 1 + 1
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action Space: BINARY
        # 0: Extend Green (Keep current phase)
        # 1: Switch to Next Phase (Standard Cycle)
        self.action_space = spaces.Discrete(2)

        # Internal State
        self.queues = np.zeros(self.n_app, dtype=np.float32)
        self.waits = np.zeros(self.n_app, dtype=np.float32)
        self.phase = 0
        self.timer = 0
        self.step_n = 0
        self.arrivals = np.zeros(self.n_app, dtype=np.float32)
        self.switches = 0
        self.total_served = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Randomize traffic density
        self.arrivals = self.np_random.uniform(
            self.arr_low, self.arr_high, size=self.n_app
        ).astype(np.float32)
        
        self.queues.fill(0)
        self.waits.fill(0)
        self.phase = 0
        self.timer = 0
        self.step_n = 0
        self.switches = 0
        self.total_served = 0
        
        return self._obs(), {}

    def step(self, action):
        self.step_n += 1
        old_phase = self.phase
        
        # --- Logic: Cyclic Control ---
        # Action 0: Extend Green Time
        # Action 1: Switch to Next Phase (Sequential)
        
        # Enforce Minimum Green Time
        if self.timer < self.min_green:
            action = 0 # Force Keep
        
        if action == 1:
            # Move to next phase in cycle (0->1->2->3->0)
            self.phase = (self.phase + 1) % self.n_phase
            self.timer = 0
            self.switches += 1
        else:
            self.timer += 1
            
        # Enforce Maximum Green Time
        if self.timer >= self.max_green:
            self.phase = (self.phase + 1) % self.n_phase
            self.timer = 0
            self.switches += 1

        did_switch = (self.phase != old_phase)

        # --- Simulation ---
        new_cars = self.np_random.poisson(self.arrivals)
        self.queues += new_cars
        
        served = 0.0
        active_lanes = self.green_map[old_phase if did_switch else self.phase]
        
        # Calculate how many cars are WAITING in the active lanes
        cars_in_green_lane = sum(self.queues[l] for l in active_lanes)
        
        for lane in active_lanes:
            flow = self.service * self.np_random.uniform(0.8, 1.2)
            s = min(self.queues[lane], flow)
            self.queues[lane] -= s
            served += s
            if self.queues[lane] > 0: self.waits[lane] *= 0.95
            else: self.waits[lane] = 0

        self.queues = np.maximum(self.queues, 0)
        self.waits += self.queues
        self.total_served += served

        # --- Smart Reward Function ---
        
        # 1. Base Penalties (Queues & Waits)
        queue_cost = -np.sum(self.queues ** 2) / 100.0
        wait_cost = -np.sum(self.waits) / 500.0
        
        # 2. Logic Reward: Wasted Green Time Penalty
        # If agent says "Keep" (Action 0) but lanes are empty -> HEAVY Penalty
        wasted_green_penalty = 0.0
        if action == 0 and cars_in_green_lane < 1.0:
            wasted_green_penalty = -5.0
            
        # 3. Logic Reward: Premature Switch Penalty
        # If agent says "Switch" (Action 1) but lanes are full -> Penalty
        premature_switch_penalty = 0.0
        if action == 1 and cars_in_green_lane > 2.0: 
            premature_switch_penalty = -5.0
            
        # 4. Service Reward
        service_reward = served * 3.0

        reward = queue_cost + wait_cost + wasted_green_penalty + premature_switch_penalty + service_reward

        terminated = False
        truncated = self.step_n >= self.max_steps
        
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _obs(self):
        phase_oh = np.zeros(self.n_phase)
        phase_oh[self.phase] = 1
        
        normalized_timer = self.timer / self.max_green
        
        # Add context: How busy is the NEXT phase?
        next_p = (self.phase + 1) % self.n_phase
        next_lanes = self.green_map[next_p]
        next_density = sum(self.queues[l] for l in next_lanes) / 50.0 # Normalize
        
        obs = np.concatenate([
            self.queues,
            phase_oh,
            [normalized_timer],
            [next_density] # Critical for deciding to switch
        ])
        return obs.astype(np.float32)

    def _info(self):
        return {
            "switches": self.switches,
            "served": self.total_served,
            "avg_queue": np.mean(self.queues)
        }