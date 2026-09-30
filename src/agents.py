"""
RL Agent Utilities: Training Environments · Saving · Loading
=============================================================
One place for everything that must be identical between training and
every script that uses a trained agent (evaluation, video, demo):

  DecisionStepWrapper  -- the agent only acts when its action can matter
                          (skips clearance and min-green seconds)
  make_training_env()  -- parallel envs + VecNormalize (obs & reward)
  save_agent()         -- model .zip + <name>_vecnormalize.pkl
  load_agent()         -- TrainedAgent with .predict(obs) that applies the
                          saved observation normalization

A TrainedAgent is queried every second with the raw 31-dim observation;
at non-decision seconds the environment ignores its action anyway.
"""

import os
import pickle
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from src.environment import TrafficSignalEnv, infer_action_mode

# Pickled SB3 schedules break across Python versions; they are not needed
# for inference, so replace them when loading.
_CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.2,
}


# =====================================================================
#  Training wrapper: act only at decision points
# =====================================================================

class DecisionStepWrapper(gym.Wrapper):
    """
    Semi-Markov wrapper: one agent step = from one decision point to the
    next. Seconds in which the action cannot change anything (yellow /
    all-red, minimum green) are simulated with the no-op action and their
    rewards are summed, so every action the agent learns from matters.
    """

    def __init__(self, env):
        super().__init__(env) # type: ignore
        self._base = env.unwrapped

    def _advance(self, obs, total_reward, terminated, truncated, info):
        while not (terminated or truncated) and not self._base.is_decision_point():
            obs, r, terminated, truncated, info = self.env.step(self._base.noop_action())
            total_reward += r
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, _, _, info = self._advance(obs, 0.0, False, False, info)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        return self._advance(obs, r, terminated, truncated, info)


# =====================================================================
#  Training environment factory
# =====================================================================

def make_training_env(cfg, action_mode, n_envs=4, seed=42, decision_steps=True,
                      normalize=True, gamma=0.99):
    """Vectorized, monitored, optionally normalized training environment."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

    def make():
        env = TrafficSignalEnv(cfg, action_mode=action_mode)
        return DecisionStepWrapper(env) if decision_steps else env

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = make_vec_env(make, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)
    return venv


def _vecnormalize_path(model_path):
    return model_path + "_vecnormalize.pkl"


def save_agent(model, path):
    """Save the PPO model and (if used) its VecNormalize statistics."""
    model.save(path)
    venv = model.get_env()
    from stable_baselines3.common.vec_env import VecNormalize
    if isinstance(venv, VecNormalize):
        venv.save(_vecnormalize_path(path))


# =====================================================================
#  Loading a trained agent for inference
# =====================================================================

class TrainedAgent:
    """PPO policy + frozen observation normalization, used like a model."""

    def __init__(self, model, obs_rms=None, clip_obs=10.0, epsilon=1e-8, path=""):
        self.model = model
        self.obs_rms = obs_rms
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.path = path
        self.action_space = model.action_space
        self.observation_space = model.observation_space

    @property
    def obs_dim(self):
        return int(self.observation_space.shape[0])

    def action_mode(self, n_phase=4):
        return infer_action_mode(self.model, n_phase)

    def normalize(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_rms is None:
            return obs
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def predict(self, obs, deterministic=True):
        return self.model.predict(self.normalize(obs), deterministic=deterministic)


def load_agent(path) -> TrainedAgent:
    """
    Load `<path>.zip` (and `<path>_vecnormalize.pkl` if present).
    Raises FileNotFoundError if the model does not exist.
    """
    from stable_baselines3 import PPO

    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(f"RL agent not found: {path}.zip (run train_rl_agent.py)")
    model = PPO.load(path, device="cpu", custom_objects=_CUSTOM_OBJECTS)

    obs_rms, clip_obs, eps = None, 10.0, 1e-8
    vn_path = _vecnormalize_path(path)
    if os.path.exists(vn_path):
        with open(vn_path, "rb") as f:
            vn = pickle.load(f)
        if getattr(vn, "norm_obs", False):
            obs_rms, clip_obs, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    return TrainedAgent(model, obs_rms, clip_obs, eps, path)


def agent_is_compatible(agent, cfg, action_mode: Optional[str] = None) -> bool:
    """True if the agent was trained on the current observation layout."""
    mode = action_mode or agent.action_mode(cfg["rl"]["num_phases"])
    env_dim = TrafficSignalEnv(cfg, action_mode=mode).observation_space.shape[0] # type: ignore
    return agent.obs_dim == env_dim
