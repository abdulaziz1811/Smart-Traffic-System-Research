"""
SUMO closed-loop environment (skipped if SUMO / traci is not installed).

Run with:  python -m pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("traci")
try:
    from src.sumo_env import SumoTrafficSignalEnv, sumo_home
    sumo_home()
except RuntimeError as e:  # SUMO binaries missing
    pytest.skip(str(e), allow_module_level=True)

from src.config import load_config
from src.baselines import actuated_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def env():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    e = SumoTrafficSignalEnv(cfg, action_mode="cyclic")
    yield e
    e.close()


def test_every_model_lane_has_a_signal_link(env):
    env.reset(seed=1)
    assert sorted(l for l in env._link_lane if l is not None) == list(range(8))
    assert env.observation_space.shape == (31,)


def test_closed_loop_episode_serves_traffic_and_shows_yellow(env):
    obs, _ = env.reset(seed=1)
    states = set()
    for _ in range(600):
        obs, r, term, trunc, info = env.step(actuated_policy(obs, env))
        states.add(env._conn.trafficlight.getRedYellowGreenState("C"))
    assert info["served"] > 50 and info["switches"] > 5
    assert any("y" in s for s in states), "yellow never shown"


def test_same_green_lanes_as_fluid_model(env):
    env.reset(seed=1)
    for lanes in env.green_map.values():
        state = env._signal_state(lanes)
        green = {env._link_lane[i] for i, ch in enumerate(state) if ch == "G"}
        assert green == set(lanes)
