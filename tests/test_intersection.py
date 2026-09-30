"""
Tests for the local agent / supervisor / reporter logic used by demo_system.py.

Run with:  python -m pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.environment import TrafficSignalEnv
from src.baselines import actuated_policy
from src.intersection import LocalIntersectionAgent
from src.supervisor import CentralSupervisor
from src.vlm_reporter import VLMReporter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ActuatedModel:
    """Stand-in for a PPO model (same predict() signature)."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        return actuated_policy(obs, self.env), None


@pytest.fixture
def cfg():
    return load_config(os.path.join(ROOT, "configs", "config.yaml"))


def make_agent(cfg, name="Main_Intersection_1", seed=0):
    env = TrafficSignalEnv(cfg)
    obs, _ = env.reset(seed=seed)
    return LocalIntersectionAgent(name, env, ActuatedModel(env)), obs


def run(agent, obs, steps, reporter=None, supervisor=None):
    reports = []
    for t in range(steps):
        obs, *_ = agent.step(agent.get_action(obs))
        if reporter and t % 10 == 0:
            reports += reporter.check_and_report(supervisor)
    return obs, reports


def test_no_false_anomalies_in_normal_traffic(cfg):
    for seed in range(3):
        agent, obs = make_agent(cfg, seed=seed)
        sup = CentralSupervisor()
        sup.register_intersection(agent)
        _, reports = run(agent, obs, 3600, VLMReporter(), sup)
        assert reports == []


def test_blocked_green_lane_is_reported(cfg):
    agent, obs = make_agent(cfg)
    sup = CentralSupervisor()
    sup.register_intersection(agent)
    agent.env.arrivals[:] = 0.1
    agent.env.blocked_lanes.add(4)
    _, reports = run(agent, obs, 600, VLMReporter(), sup)
    assert reports, "incident was never reported"
    assert all(r["lanes"] == [4] for r in reports)
    assert "E straight" in reports[0]["summary"]


def test_emergency_holds_green_beyond_max_green(cfg):
    agent, obs = make_agent(cfg)
    agent.detect_ambulance(lane=4)          # E straight -> phase 2
    obs, _ = run(agent, obs, 200)
    assert agent.current_phase == 2
    agent.clear_ambulance()
    assert agent.env.hold_green is False


def test_green_wave_deactivates_after_route_completed(cfg):
    a1, _ = make_agent(cfg, "Main_Intersection_1")
    a2, _ = make_agent(cfg, "Main_Intersection_2")
    sup = CentralSupervisor()
    sup.register_intersection(a1)
    sup.register_intersection(a2)

    a1.detect_ambulance(lane=0)
    sup.handle_ambulance_intention(a1.id, a1.ambulance_intention)
    assert sup.green_wave_active and a2.emergency_mode

    sup.handle_ambulance_cleared(a1.id)
    assert sup.green_wave_active          # still on its way to intersection 2
    sup.handle_ambulance_cleared(a2.id)
    assert not sup.green_wave_active
    assert sup.ambulance_path_plan == []
