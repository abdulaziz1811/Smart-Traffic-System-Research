"""
Sanity tests for the traffic signal environment.

Run with:  python -m pytest tests/
"""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.environment import TrafficSignalEnv
from src.baselines import fixed_time_policy, actuated_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cfg():
    return load_config(os.path.join(ROOT, "configs", "config.yaml"))


def make_env(cfg, **rl_overrides):
    c = copy.deepcopy(cfg)
    c["rl"].update(rl_overrides)
    return TrafficSignalEnv(c)


def test_observation_shape(cfg):
    env = make_env(cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (31,)
    obs, *_ = env.step(0)
    assert obs.shape == (31,)


def test_min_green_is_enforced(cfg):
    env = make_env(cfg, yellow_time=0)
    env.reset(seed=0)
    for _ in range(env.min_green):
        env.step(1)
        assert env.phase == 0
    env.step(1)
    assert env.phase == 1


def test_max_green_forces_switch_unless_held(cfg):
    env = make_env(cfg, yellow_time=0)
    env.reset(seed=0)
    for _ in range(env.max_green):
        env.step(0)
    assert env.phase == 1

    env.reset(seed=0)
    env.hold_green = True
    for _ in range(env.max_green + 10):
        env.step(0)
    assert env.phase == 0


def test_clearance_serves_no_lane(cfg):
    env = make_env(cfg, yellow_time=3, arrival_rate_low=0.3, arrival_rate_high=0.3)
    env.reset(seed=0)
    for _ in range(env.min_green):
        env.step(0)
    env.step(1)  # switch: last green second of phase 0
    assert env.phase == 1
    for _ in range(3):
        served_before = env.total_served
        env.step(1)  # actions are ignored during clearance
        assert env.last_active_lanes == []
        assert env.total_served == served_before
        assert env.phase == 1
    env.step(0)
    assert env.last_active_lanes == env.green_map[1]


def test_free_mode_can_jump_to_any_phase(cfg):
    env = make_env(cfg, yellow_time=0, action_mode="free")
    assert env.action_space.n == 4
    env.reset(seed=0)
    for _ in range(env.min_green):
        env.step(0)   # 0 == current phase -> extend
    env.step(3)
    assert env.phase == 3


def test_blocked_lane_is_not_served(cfg):
    env = make_env(cfg, yellow_time=0, arrival_rate_low=0.5, arrival_rate_high=0.5)
    env.reset(seed=0)
    env.blocked_lanes = {0}
    for _ in range(20):
        before = env.queues[0]
        env.step(0)
        assert env.queues[0] >= before


def test_same_seed_gives_same_arrivals_for_any_policy(cfg):
    """Common random numbers: the policy must not change the traffic demand."""
    def arrivals(policy):
        env = make_env(cfg)
        obs, _ = env.reset(seed=123)
        total = []
        for t in range(500):
            before = env.queues.copy()
            served_before = env.total_served
            obs, *_ = env.step(policy(obs, env, t))
            # arrivals = new queue + served - old queue
            total.append(float(np.sum(env.queues) + (env.total_served - served_before) - np.sum(before)))
        return np.array(total)

    a = arrivals(lambda obs, env, t: 0)
    b = arrivals(lambda obs, env, t: 1)
    c = arrivals(lambda obs, env, t: fixed_time_policy(obs, env, green=10))
    # float32 queues -> compare with a tolerance far below one vehicle
    assert np.allclose(a, b, atol=1e-3) and np.allclose(a, c, atol=1e-3)


def test_fixed_time_policy_green_length(cfg):
    env = make_env(cfg, yellow_time=0)
    obs, _ = env.reset(seed=0)
    for _ in range(3600):
        obs, _, _, trunc, info = env.step(fixed_time_policy(obs, env, green=30))
        if trunc:
            break
    assert info["switches"] == 3600 // 30


def test_actuated_beats_long_fixed_timer(cfg):
    def avg_queue(policy, seed):
        env = make_env(cfg)
        obs, _ = env.reset(seed=seed)
        q = []
        for _ in range(3600):
            obs, _, _, trunc, info = env.step(policy(obs, env))
            q.append(info["avg_queue"])
        return np.mean(q)

    fixed = np.mean([avg_queue(lambda o, e: fixed_time_policy(o, e, 30), s) for s in range(3)])
    act = np.mean([avg_queue(actuated_policy, s) for s in range(3)])
    assert act < fixed
