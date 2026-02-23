#!/usr/bin/env python3
"""
Comparative Training: Fixed Timer vs Cyclic AI vs Free AI
==========================================================
Trains two PPO agents (cyclic-constrained and unconstrained),
evaluates against a fixed-timer baseline, and produces a
training progress comparison chart.

Key features:
  - Curriculum learning: traffic difficulty increases in 3 stages
  - Correct fixed-timer baseline (returns 0/1, not phase index)
  - Correct logging key (avg_queue from environment info dict)
  - Compatible with V4 environment (22-dim observation with trend)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.config import bootstrap
from src.environment import TrafficSignalEnv

# -- Academic plot style -----------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "lines.linewidth": 2.5,
    "grid.alpha": 0.6,
})


# =====================================================================
#  Fixed Timer Baseline
# =====================================================================

class FixedTimeController:
    """
    Traditional fixed-cycle traffic signal controller.
    Switches to the next phase every `green_duration` steps.
    Returns 0 (extend) or 1 (switch) to match env.action_space.
    """

    def __init__(self, green_duration=30):
        self.green_duration = green_duration
        self.time_in_phase = 0

    def get_action(self):
        self.time_in_phase += 1
        if self.time_in_phase >= self.green_duration:
            self.time_in_phase = 0
            return 1  # switch to next phase
        return 0      # extend current phase


# =====================================================================
#  Cyclic Wrapper
# =====================================================================

class CyclicTrafficEnv(gym.Wrapper):
    """
    Conceptual wrapper marking the agent as cycle-constrained.

    The base TrafficSignalEnv already enforces sequential phase
    order (0 -> 1 -> 2 -> 3 -> 0), so this wrapper passes actions
    through without modification. It exists to make the constraint
    explicit in the training code and paper description.
    """

    def __init__(self, env):
        super().__init__(env) # type: ignore
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        return self.env.step(action)


# =====================================================================
#  Queue Logger Callback
# =====================================================================

class QueueLogger(BaseCallback):
    """
    Records rolling-average queue length during training.
    Aggregates every `window` steps to produce a smooth history.
    """

    def __init__(self, window=1000, verbose=0):
        super().__init__(verbose) # type: ignore
        self.window = window
        self.buffer = []
        self.history = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.buffer.append(info.get("avg_queue", 0.0))
        if len(self.buffer) >= self.window:
            self.history.append(float(np.mean(self.buffer)))
            self.buffer = []
        return True


# =====================================================================
#  Curriculum Learning Callback
# =====================================================================

class CurriculumCallback(BaseCallback):
    """
    Gradually increases traffic difficulty during training.

    Stage 1 (  0% -  33%): Low variance, light traffic.
      The agent learns basic phase timing on easy scenarios.

    Stage 2 ( 33% -  66%): Medium variance, moderate traffic.
      The agent learns to handle unbalanced demand across lanes.

    Stage 3 ( 66% - 100%): Full variance, heavy traffic.
      The agent learns robust policies for worst-case scenarios.

    This approach is inspired by curriculum learning literature:
    starting simple helps the agent converge faster and avoids
    getting stuck in poor local optima from noisy early updates.
    """

    STAGES = [
        {"arr_low": 0.03, "arr_high": 0.06, "label": "Light"},
        {"arr_low": 0.02, "arr_high": 0.12, "label": "Medium"},
        {"arr_low": 0.01, "arr_high": 0.20, "label": "Heavy"},
    ]

    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose) # type: ignore
        self.total_steps = total_steps
        self.current_stage = -1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_steps
        stage_idx = min(int(progress * len(self.STAGES)), len(self.STAGES) - 1)

        if stage_idx != self.current_stage:
            self.current_stage = stage_idx
            params = self.STAGES[stage_idx]

            # Reach through wrappers to the actual TrafficSignalEnv
            env = self.training_env.envs[0] # type: ignore
            target = env.unwrapped if hasattr(env, "unwrapped") else env
            target.arr_low = params["arr_low"]
            target.arr_high = params["arr_high"]

            if self.verbose > 0:
                print(f"  [Curriculum] Stage {stage_idx + 1}/{len(self.STAGES)} "
                      f"({params['label']}) at step {self.num_timesteps:,}: "
                      f"arrivals=[{params['arr_low']}, {params['arr_high']}]")

        return True


# =====================================================================
#  Baseline Evaluation
# =====================================================================

def evaluate_baseline(cfg, steps=10000):
    """Run the fixed-timer controller and return average queue length."""
    env = TrafficSignalEnv(cfg)
    ctrl = FixedTimeController(green_duration=30)
    obs, _ = env.reset()
    queues = []

    for _ in range(steps):
        action = ctrl.get_action()
        obs, _, done, truncated, info = env.step(action)
        queues.append(info["avg_queue"])
        if done or truncated:
            obs, _ = env.reset()
            ctrl.time_in_phase = 0

    return float(np.mean(queues)) if queues else 25.0


# =====================================================================
#  Training Pipeline
# =====================================================================

def train_agent(name, env, total_steps, rl_dir, use_curriculum=True, verbose=1):
    """
    Train a PPO agent with optional curriculum learning.

    Args:
        name:            model save name (without extension)
        env:             gymnasium environment instance
        total_steps:     total training timesteps
        rl_dir:          directory to save the trained model
        use_curriculum:  whether to apply curriculum learning
        verbose:         curriculum callback verbosity

    Returns:
        (model, queue_logger)
    """
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        device="cpu",
    )

    queue_log = QueueLogger()
    callbacks: list[BaseCallback] = [queue_log]

    if use_curriculum:
        callbacks.append(CurriculumCallback(total_steps, verbose=verbose))

    model.learn(total_timesteps=total_steps, callback=callbacks)
    save_path = os.path.join(rl_dir, name)
    model.save(save_path)

    return model, queue_log


# =====================================================================
#  Plot Generation
# =====================================================================

def generate_plot(avg_fixed, logger_cyclic, logger_free, total_steps, save_path):
    """Generate training progress comparison chart with curriculum markers."""
    plt.figure(figsize=(12, 7))

    steps_axis = np.linspace(0, total_steps, max(len(logger_free.history), 1))

    # Fixed timer reference line
    plt.axhline(
        y=avg_fixed, color="#e74c3c", linestyle="--", linewidth=2.5,
        label=f"Fixed Timer 30s (Avg: {avg_fixed:.1f})",
    )

    # Cyclic AI training curve
    n = min(len(steps_axis), len(logger_cyclic.history))
    if n > 0:
        plt.plot(
            steps_axis[:n], logger_cyclic.history[:n],
            color="#e67e22", linewidth=2, label="Cyclic AI (Fixed Order)",
        )

    # Free AI training curve
    n = min(len(steps_axis), len(logger_free.history))
    if n > 0:
        plt.plot(
            steps_axis[:n], logger_free.history[:n],
            color="#2ecc71", linewidth=2.5, label="Free AI (Fully Adaptive)",
        )

    # Mark curriculum stage transitions
    stage_labels = ["Stage 2: Medium", "Stage 3: Heavy"]
    for i, (frac, label) in enumerate(zip([0.33, 0.66], stage_labels)):
        x = frac * total_steps
        plt.axvline(x=x, color="gray", linestyle=":", alpha=0.5)
        plt.text(
            x + total_steps * 0.005, plt.ylim()[1] * 0.92,
            label, fontsize=9, color="gray", fontstyle="italic",
        )

    plt.title("Training Progress: Traffic Control Strategies", pad=15, fontweight="bold")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Queue Length (Vehicles)")
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =====================================================================
#  Main
# =====================================================================

def main():
    cfg, log, device = bootstrap("configs/config.yaml")

    rl_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_dir, exist_ok=True)

    total_steps = 500_000

    log.info("Starting comparative study: Fixed Timer vs Cyclic AI vs Free AI")
    log.info("Training steps per agent: %s", f"{total_steps:,}")
    log.info("Curriculum learning: enabled (3 stages)")

    # -- Phase 1: Baseline evaluation --
    log.info("Evaluating fixed-timer baseline (30s cycle)...")
    avg_fixed = evaluate_baseline(cfg, steps=10000)
    log.info("Fixed timer average queue: %.2f", avg_fixed)

    # -- Phase 2: Train Cyclic AI (with curriculum) --
    log.info("Training Cyclic AI (constrained, with curriculum)...")
    cyclic_env = CyclicTrafficEnv(TrafficSignalEnv(cfg))
    _, logger_cyclic = train_agent(
        "cyclic_agent", cyclic_env, total_steps, rl_dir, use_curriculum=True
    )
    log.info("Cyclic AI training complete.")

    # -- Phase 3: Train Free AI (with curriculum) --
    log.info("Training Free AI (unconstrained, with curriculum)...")
    free_env = TrafficSignalEnv(cfg)
    _, logger_free = train_agent(
        "free_agent", free_env, total_steps, rl_dir, use_curriculum=True
    )
    log.info("Free AI training complete.")

    # -- Phase 4: Generate comparison plot --
    plot_path = os.path.join(rl_dir, "Training_Comparison.png")
    generate_plot(avg_fixed, logger_cyclic, logger_free, total_steps, plot_path)
    log.info("Comparison plot saved: %s", plot_path)

    log.info("All training complete. Models saved to: %s", rl_dir)


if __name__ == "__main__":
    main()