"""
Rule-Based Baseline Controllers
================================
Shared by train_rl_agent.py, compare_agents.py, validate_full.py and the
tests, so every script compares the RL agent against the SAME baselines.

Each policy has the signature  policy(obs, env) -> action  and returns an
action that is valid for `env.action_mode` ("cyclic" or "free").

  fixed_time_policy     -- switch after a fixed green duration (traditional)
  actuated_policy       -- gap-out: switch as soon as the green lanes are empty
  longest_queue_policy  -- gap-out, then jump to the busiest phase (free order)
"""

import numpy as np


def _phase_and_timer(obs, env):
    phase = int(np.argmax(obs[env.n_app: env.n_app + env.n_phase]))
    timer = int(round(float(obs[env.n_app + env.n_phase]) * env.max_green))
    return phase, timer


def _action(env, phase, target):
    """Translate 'go to phase `target`' into an action for the env's mode."""
    if env.action_mode == "free":
        return int(target)
    return 0 if target == phase else 1


def fixed_time_policy(obs, env, green=30):
    """Traditional fixed-time plan: `green` seconds per phase, fixed order."""
    phase, timer = _phase_and_timer(obs, env)
    # The switching step itself is the last green second of the phase
    if timer >= green - 1:
        return _action(env, phase, (phase + 1) % env.n_phase)
    return _action(env, phase, phase)


def actuated_policy(obs, env, gap=0.5):
    """
    Vehicle-actuated control (gap-out): keep green while the green lanes
    still hold vehicles, switch to the next phase once they are (almost)
    empty. min_green / max_green are enforced by the environment.
    """
    phase, _ = _phase_and_timer(obs, env)
    queues = obs[:env.n_app]
    active_queue = sum(queues[l] for l in env.green_map[phase])
    if active_queue < gap:
        return _action(env, phase, (phase + 1) % env.n_phase)
    return _action(env, phase, phase)


def longest_queue_policy(obs, env, gap=0.5):
    """
    Longest-queue-first with gap-out (single-intersection max-pressure).
    Keeps the current green while its lanes still hold vehicles (every
    switch costs yellow time), then gives green to the busiest phase.
    In "cyclic" mode the order is fixed, so it behaves like actuated_policy.
    """
    phase, _ = _phase_and_timer(obs, env)
    queues = obs[:env.n_app]
    demand = [sum(queues[l] for l in env.green_map[p]) for p in range(env.n_phase)]
    if demand[phase] >= gap:
        return _action(env, phase, phase)
    if env.action_mode == "free":
        others = [p for p in range(env.n_phase) if p != phase]
        return max(others, key=lambda p: demand[p])
    return _action(env, phase, (phase + 1) % env.n_phase)
