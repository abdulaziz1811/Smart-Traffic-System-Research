#!/usr/bin/env python3
"""
RL Agent Live Test
==================
Loads a trained PPO agent and runs it through a full episode
with a real-time dashboard showing queue states, decisions,
and performance metrics.

Compatible with V4 environment (22-dim observation).

Usage:
    python test_rl_agent.py
    python test_rl_agent.py --model models/rl_agents/free_agent
    python test_rl_agent.py --speed 0.05
"""

import os
import sys
import time
import argparse

import numpy as np
from stable_baselines3 import PPO
from src.config import bootstrap
from src.environment import TrafficSignalEnv


# -- Observation parsing helpers (V4: 22-dim) --------------------------

PHASE_NAMES = [
    "Phase 0: N/S Straight",
    "Phase 1: N/S Left",
    "Phase 2: E/W Straight",
    "Phase 3: E/W Left",
]


def parse_obs(obs):
    """
    Parse the 22-dim observation vector into named components.

    Layout:
        [0:8]   queue lengths per lane
        [8:12]  current phase (one-hot)
        [12]    normalized green timer
        [13]    next phase queue density
        [14:22] queue trend per lane
    """
    return {
        "queues": obs[0:8],
        "phase_idx": int(np.argmax(obs[8:12])),
        "timer": float(obs[12]),
        "next_density": float(obs[13]),
        "trend": obs[14:22],
    }


# -- Dashboard rendering -----------------------------------------------

def render_dashboard(step, max_steps, parsed, action, reward, total_reward):
    """Print a formatted dashboard to the terminal."""
    q = parsed["queues"]
    t = parsed["trend"]
    phase = parsed["phase_idx"]
    timer = parsed["timer"]
    next_d = parsed["next_density"]

    # Clear screen
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 60)
    print("  SMART TRAFFIC SIGNAL -- LIVE TEST")
    print("=" * 60)
    print()
    print(f"  Step: {step:4d} / {max_steps}   |   Timer: {timer:.2f}")
    print(f"  Phase: {PHASE_NAMES[phase]}")
    print()

    # Queue table with trend arrows
    print("  Lane             Queue   Trend")
    print("  " + "-" * 38)
    lane_names = [
        "North Straight", "North Left",
        "South Straight", "South Left",
        "East  Straight", "East  Left",
        "West  Straight", "West  Left",
    ]
    for i, name in enumerate(lane_names):
        trend_val = t[i]
        if trend_val > 0.5:
            arrow = "(++)"
        elif trend_val > 0:
            arrow = "(+)"
        elif trend_val < -0.5:
            arrow = "(--)"
        elif trend_val < 0:
            arrow = "(-)"
        else:
            arrow = "(=)"
        print(f"  {name:<18s}  {int(q[i]):3d}    {arrow}")

    print()
    print("  " + "-" * 38)

    action_str = "EXTEND green" if action == 0 else "SWITCH phase"
    print(f"  Decision:        {action_str}")
    print(f"  Next Density:    {next_d:.3f}")
    print(f"  Step Reward:     {reward:+.2f}")
    print(f"  Total Reward:    {total_reward:+.1f}")
    print()
    print("=" * 60)


# -- Main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Test a trained RL agent interactively")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="models/rl_agents/free_agent",
                    help="Path to trained model (without .zip extension)")
    ap.add_argument("--speed", type=float, default=0.1,
                    help="Delay between steps in seconds (lower = faster)")
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)

    # -- Load environment --
    env = TrafficSignalEnv(cfg)
    max_steps = cfg["rl"]["max_steps"]

    # -- Load model --
    model_path = args.model
    if not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}.zip")
        print("       Run train_rl_agent.py first to produce trained agents.")
        sys.exit(1)

    log.info("Loading agent from: %s", model_path)
    model = PPO.load(model_path)

    # -- Verify observation space compatibility --
    env_obs_dim = env.observation_space.shape[0]
    model_obs_dim = model.observation_space.shape[0]
    if env_obs_dim != model_obs_dim:
        print(f"WARNING: Observation dimension mismatch!")
        print(f"  Environment expects: {env_obs_dim}")
        print(f"  Model trained on:    {model_obs_dim}")
        print(f"  You may need to retrain the agent with the new environment.")
        sys.exit(1)

    # -- Run episode --
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0

    print()
    print("=" * 60)
    print("  STARTING SMART TRAFFIC LOGIC TEST")
    print("=" * 60)
    print()

    try:
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            parsed = parse_obs(obs)
            render_dashboard(step, max_steps, parsed, int(action), reward, total_reward)

            time.sleep(args.speed)

        # -- Summary --
        print()
        print("=" * 60)
        print("  TEST COMPLETE")
        print("=" * 60)
        print(f"  Steps completed:   {step}")
        print(f"  Total reward:      {total_reward:+.1f}")
        print(f"  Vehicles served:   {info['served']:.0f}")
        print(f"  Phase switches:    {info['switches']}")
        print(f"  Final avg queue:   {info['avg_queue']:.2f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print()
        print("  Simulation stopped by user.")
        print(f"  Steps completed: {step}  |  Total reward: {total_reward:+.1f}")


if __name__ == "__main__":
    main()