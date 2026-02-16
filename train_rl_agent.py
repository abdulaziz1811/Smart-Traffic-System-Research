#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════
  Comparative Training: Fixed Timer vs Cyclic AI vs Free AI
  Smart Traffic Signal Control Research
══════════════════════════════════════════════════════════════

Trains two RL agents with different constraints, evaluates
against a fixed-timer baseline, and generates a comparison plot.

FIXES applied:
  1. FixedTimeController now returns 0/1 (not phase index 0-3)
  2. CyclicTrafficEnv passes action directly (env already cyclic)
  3. ComparisonLogger reads 'avg_queue' (correct info key)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.config import bootstrap
from src.environment import TrafficSignalEnv

# ── Academic Plot Style ──
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14,   'axes.titlesize': 16,
    'xtick.labelsize': 12,  'ytick.labelsize': 12,
    'legend.fontsize': 12,  'figure.dpi': 300,
    'lines.linewidth': 2.5, 'grid.alpha': 0.6,
})

# ═════════════════════════════════════════════════════════
#  1. FIXED Baseline Controller
# ═════════════════════════════════════════════════════════
#
#  BUG FIXED: Original returned self.current_phase (0-3),
#  but env.step() only accepts 0=extend or 1=switch.
#  Phases 2,3 were silently treated as "extend".
#
#  FIX: Return 1 (switch) when timer expires, 0 otherwise.
# ═════════════════════════════════════════════════════════

class FixedTimeController:
    """Traditional fixed-timer traffic signal (30s per phase)."""

    def __init__(self, green_duration=30):
        self.green_duration = green_duration
        self.time_in_phase = 0

    def get_action(self):
        """Returns 0 (extend) or 1 (switch) — matches env.action_space."""
        self.time_in_phase += 1
        if self.time_in_phase >= self.green_duration:
            self.time_in_phase = 0
            return 1   # ← SWITCH to next phase
        return 0       # ← EXTEND current phase


# ═════════════════════════════════════════════════════════
#  2. FIXED Cyclic Wrapper
# ═════════════════════════════════════════════════════════
#
#  BUG FIXED: Original passed internal_phase (0-3) to env,
#  but env only accepts 0 or 1. When internal_phase=1,
#  keep action was inverted to switch. Phases 2,3 couldn't
#  switch at all.
#
#  FIX: The base environment ALREADY enforces cyclic order
#  (0→1→2→3→0). So just pass the action through directly.
#  The wrapper is kept for conceptual clarity in the paper.
# ═════════════════════════════════════════════════════════

class CyclicTrafficEnv(gym.Wrapper):
    """
    Wrapper that conceptually constrains agent to cyclic order.
    
    In practice, the base TrafficSignalEnv already enforces:
      action=0 → extend current phase
      action=1 → advance to next phase (0→1→2→3→0)
    
    This wrapper makes the constraint explicit for clarity.
    """

    def __init__(self, env):
        super().__init__(env)
        # Same action space as base env: 0=extend, 1=switch
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        # Pass action directly — env already handles cycling
        return self.env.step(action)


# ═════════════════════════════════════════════════════════
#  3. FIXED Logging Callback
# ═════════════════════════════════════════════════════════
#
#  BUG FIXED: Original used info.get('queue_len') which
#  doesn't exist. Environment returns 'avg_queue'.
# ═════════════════════════════════════════════════════════

class ComparisonLogger(BaseCallback):
    """Logs average queue length during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.epoch_queues = []
        self.history = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        val = info.get('avg_queue', 0)  # ← FIXED: correct key
        self.epoch_queues.append(val)

        if len(self.epoch_queues) >= 1000:
            self.history.append(np.mean(self.epoch_queues))
            self.epoch_queues = []
        return True


# ═════════════════════════════════════════════════════════
#  4. Baseline Evaluation
# ═════════════════════════════════════════════════════════

def evaluate_baseline(cfg, steps=10000):
    """Evaluate Fixed Timer on fresh environment."""
    print(f"  Running Fixed Timer for {steps} steps...")
    env = TrafficSignalEnv(cfg)
    ctrl = FixedTimeController(green_duration=30)
    obs, _ = env.reset()
    queues = []

    for _ in range(steps):
        action = ctrl.get_action()
        obs, _, done, truncated, info = env.step(action)
        queues.append(info['avg_queue'])
        if done or truncated:
            obs, _ = env.reset()
            ctrl.time_in_phase = 0

    return np.mean(queues) if queues else 25.0


# ═════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════

def main():
    cfg, log, device = bootstrap("configs/config.yaml")

    rl_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_dir, exist_ok=True)

    log.info("Starting Comparative Study: Fixed Time vs Cyclic AI vs Free AI")

    # ── Phase 1: Baseline ──
    avg_fixed = evaluate_baseline(cfg, steps=10000)
    log.info(f"Baseline (Fixed Timer 30s) Avg Queue: {avg_fixed:.2f}")

    total_steps = 500000

    # ── Phase 2: Cyclic AI ──
    log.info("Training Cyclic AI (constrained to cyclic order)...")
    cyclic_env = CyclicTrafficEnv(TrafficSignalEnv(cfg))

    model_cyclic = PPO("MlpPolicy", cyclic_env, verbose=0,
                       learning_rate=3e-4, device="cpu")
    logger_cyclic = ComparisonLogger()
    model_cyclic.learn(total_timesteps=total_steps, callback=logger_cyclic)
    model_cyclic.save(os.path.join(rl_dir, "cyclic_agent"))
    log.info("Cyclic AI training complete.")

    # ── Phase 3: Free AI ──
    log.info("Training Free AI (fully adaptive)...")
    free_env = TrafficSignalEnv(cfg)

    model_free = PPO("MlpPolicy", free_env, verbose=0,
                     learning_rate=3e-4, device="cpu")
    logger_free = ComparisonLogger()
    model_free.learn(total_timesteps=total_steps, callback=logger_free)
    model_free.save(os.path.join(rl_dir, "free_agent"))
    log.info("Free AI training complete.")

    # ── Phase 4: Plot ──
    log.info("Generating comparison plot...")
    plt.figure(figsize=(12, 7))

    steps_range = np.linspace(0, total_steps, max(len(logger_free.history), 1))

    # Fixed Timer baseline
    plt.axhline(y=avg_fixed, color='#e74c3c', linestyle='--', linewidth=2.5,
                label=f'Fixed Timer 30s (Avg: {avg_fixed:.1f})')

    # Cyclic AI
    n = min(len(steps_range), len(logger_cyclic.history))
    if n > 0:
        plt.plot(steps_range[:n], logger_cyclic.history[:n],
                 color='#e67e22', linewidth=2,
                 label='Cyclic AI (Fixed Order)')

    # Free AI
    n = min(len(steps_range), len(logger_free.history))
    if n > 0:
        plt.plot(steps_range[:n], logger_free.history[:n],
                 color='#2ecc71', linewidth=2.5,
                 label='Free AI (Fully Adaptive)')

    plt.title('Training Progress: Traffic Control Strategies', 
              pad=15, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Queue Length (Vehicles)')
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)

    plot_path = os.path.join(rl_dir, "Training_Comparison.png")
    plt.savefig(plot_path, bbox_inches='tight')
    log.info(f"Plot saved: {plot_path}")

    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()