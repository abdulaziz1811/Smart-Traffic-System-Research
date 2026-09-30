"""
The real-time controller in track_video.py must feed the agent exactly the
same observations and apply the same timing rules as the simulator.

Run with:  python -m pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.environment import TrafficSignalEnv, GREEN_MAP
from track_video import SignalController

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RecordingAgent:
    """Obs-only rule policy (gap-out, jumps to busiest phase in free mode)."""

    def __init__(self, mode):
        self.mode, self.seen = mode, []

    def action_mode(self, n_phase=4):
        return self.mode

    def predict(self, obs, deterministic=True):
        self.seen.append(np.array(obs))
        q, phase = obs[:8], int(np.argmax(obs[8:12]))
        demand = [q[a] + q[b] for a, b in GREEN_MAP.values()]
        if demand[phase] >= 0.5:
            return (phase if self.mode == "free" else 0), None
        if self.mode == "free":
            return max((p for p in range(4) if p != phase), key=lambda p: demand[p]), None
        return 1, None


@pytest.mark.parametrize("mode", ["cyclic", "free"])
def test_controller_matches_simulator(mode):
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    env = TrafficSignalEnv(cfg, action_mode=mode)
    obs, _ = env.reset(seed=3)
    sim_agent, ctl_agent = RecordingAgent(mode), RecordingAgent(mode)
    ctl = SignalController(cfg, ctl_agent)

    for t in range(1500):
        n_before = len(ctl_agent.seen)
        ctl.step(env.queues.copy())
        if len(ctl_agent.seen) > n_before:   # controller queried the agent
            assert np.allclose(ctl_agent.seen[-1], obs, atol=1e-5), f"obs differ at t={t}"
        action, _ = sim_agent.predict(obs)
        obs, *_ = env.step(action)
        assert (ctl.phase, ctl.timer, ctl.clearance_left) == (env.phase, env.timer, env.clearance_left)
    assert env.switches > 20


def test_emergency_preemption_reaches_and_holds_lane():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    ctl = SignalController(cfg, RecordingAgent("cyclic"))
    ctl.preempt(lane=5, seconds=120)          # E left -> phase 3
    for _ in range(100):
        ctl.step(np.full(8, 3.0))
    assert ctl.phase == 3 and ctl.timer > cfg["rl"]["max_green"]
