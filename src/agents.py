"""
RL Agent Utilities: Training Environments · Saving · Loading
=============================================================
One place for everything that must be identical between training and
every script that uses a trained agent (evaluation, video, demo):

  DecisionStepWrapper  -- the agent only acts when its action can matter
                          (skips clearance and min-green seconds). Note:
                          PPO discounts per step, so a step spanning
                          several seconds biases against switching; off
                          by default in train_rl_agent.py.
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

def make_training_env(cfg, action_mode, n_envs=4, seed=42, decision_steps=False,
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


def behavior_cloning(model, venv, cfg, action_mode, expert, n_steps=50_000, epochs=10,
                     batch_size=256, lr=1e-3, dagger_rounds=0):
    """
    Pre-train the PPO policy to imitate a rule-based expert (e.g. actuated
    control) before RL fine-tuning, so PPO starts from a competent policy
    instead of random switching.

    Round 0: the expert drives `venv` (this also warms up the VecNormalize
    statistics) and the policy is fitted to (normalized obs, expert action)
    pairs with a cross-entropy loss.
    DAgger rounds: the learner drives, every state it visits is labelled with
    the expert's action, and the policy is refitted on all data. This covers
    the states plain cloning never sees (after the learner's own mistakes).

    Returns the final accuracy on the collected data.
    """
    import torch as th

    spec = TrafficSignalEnv(cfg, action_mode=action_mode)  # constants for the expert policy
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    raw_obs, actions = [], []

    def collect(learner_drives):
        obs = venv.reset()
        for _ in range(max(n_steps // venv.num_envs, 1)):
            raw = venv.get_original_obs() if hasattr(venv, "get_original_obs") else obs
            labels = np.array([expert(o, spec) for o in raw])
            raw_obs.append(raw.copy())
            actions.append(labels)
            act = model.predict(obs, deterministic=True)[0] if learner_drives else labels
            obs, _, _, _ = venv.step(act)

    def fit():
        data = np.concatenate(raw_obs)
        obs_n = venv.normalize_obs(data) if hasattr(venv, "normalize_obs") else data
        x = th.as_tensor(obs_n, dtype=th.float32, device=policy.device)
        y = th.as_tensor(np.concatenate(actions), dtype=th.long, device=policy.device)
        policy.set_training_mode(True)
        for _ in range(epochs):
            perm = th.randperm(len(x))
            for i in range(0, len(x), batch_size):
                idx = perm[i:i + batch_size]
                loss = -policy.get_distribution(x[idx]).log_prob(y[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        policy.set_training_mode(False)
        with th.no_grad():
            pred = policy.get_distribution(x).distribution.probs.argmax(-1)
        return float((pred == y).float().mean())

    collect(learner_drives=False)
    acc = fit()
    for _ in range(dagger_rounds):
        collect(learner_drives=True)
        acc = fit()
    return acc


def evaluate_policy_queue(model, venv, cfg, action_mode, seeds, steps=3600):
    """Average queue of the (deterministic) policy on the given seeds, fluid model."""
    queues = []
    for seed in seeds:
        env = TrafficSignalEnv(cfg, action_mode=action_mode)
        obs, _ = env.reset(seed=int(seed))
        for _ in range(steps):
            obs_n = venv.normalize_obs(obs) if hasattr(venv, "normalize_obs") else obs
            action = model.predict(obs_n, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            queues.append(info["avg_queue"])
            if term or trunc:
                break
    return float(np.mean(queues))


def make_best_model_callback(cfg, action_mode, path, eval_freq=50_000, seeds=(1000, 1001, 1002, 1003, 1004),
                             verbose=1):
    """
    Keep the best policy seen during training (early stopping by model
    selection). Evaluated on validation seeds that are disjoint from the
    test seeds (42-51) used by compare_agents.py / validate_full.py, at the
    start (the imitation warm start) and every `eval_freq` steps; the best
    one is saved to `path` with its normalization statistics.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class BestModelCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose)
            self.best = float("inf")
            self.history = []
            self._last = 0

        def _evaluate(self):
            score = evaluate_policy_queue(self.model, self.training_env, cfg, action_mode, seeds)
            self.history.append((self.num_timesteps, score))
            if score < self.best:
                self.best = score
                save_agent(self.model, path)
            if self.verbose:
                print(f"  [Validation] step {self.num_timesteps:>9,}: avg queue {score:.3f} "
                      f"(best {self.best:.3f})", flush=True)

        def _on_training_start(self):
            self._evaluate()

        def _on_step(self):
            if self.num_timesteps - self._last >= eval_freq:
                self._last = self.num_timesteps
                self._evaluate()
            return True

    return BestModelCallback()


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
