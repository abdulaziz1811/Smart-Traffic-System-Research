"""
Tests for the RL agent utilities (training wrapper, imitation, save/load).

Run with:  python -m pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO

from src.config import load_config
from src.environment import TrafficSignalEnv
from src.agents import (DecisionStepWrapper, make_training_env, behavior_cloning,
                        save_agent, load_agent, agent_is_compatible)
from src.baselines import actuated_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cfg():
    return load_config(os.path.join(ROOT, "configs", "config.yaml"))


def test_decision_wrapper_only_stops_at_decision_points(cfg):
    env = DecisionStepWrapper(TrafficSignalEnv(cfg, action_mode="cyclic"))
    env.reset(seed=0)
    base = env.unwrapped
    for t in range(300):
        _, _, term, trunc, _ = env.step(t % 2)
        assert base.is_decision_point() or term or trunc


def test_imitation_then_save_and_load_round_trip(cfg, tmp_path):
    venv = make_training_env(cfg, "cyclic", n_envs=1, seed=0)
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, device="cpu", seed=0)
    acc = behavior_cloning(model, venv, cfg, "cyclic", actuated_policy, n_steps=3000, epochs=5)
    assert acc > 0.85

    path = str(tmp_path / "cyclic_agent")
    save_agent(model, path)
    assert os.path.exists(path + "_vecnormalize.pkl")

    agent = load_agent(path)
    assert agent.action_mode() == "cyclic" and agent_is_compatible(agent, cfg)
    raw = venv.get_original_obs()[0]
    expected = model.predict(venv.normalize_obs(raw), deterministic=True)[0]
    assert int(agent.predict(raw)[0]) == int(expected)
    venv.close()
