#!/usr/bin/env python3
"""
Comparative Training: Fixed Timer vs Cyclic AI vs Free AI
==========================================================
Trains two PPO agents (cyclic-constrained and unconstrained),
evaluates against a fixed-timer baseline, and produces a
training progress comparison chart.

Key features:
  - Curriculum learning: traffic difficulty increases in 3 stages
  - Cyclic agent = Discrete(2) extend/next, Free agent = Discrete(4) pick
    any phase (previously both agents used the identical cyclic env)
  - Baselines (fixed timer + actuated) are evaluated on the SAME traffic
    level as each curriculum stage, so the training curve is compared
    against a like-for-like reference
  - Seeded PPO for reproducibility
  - Compatible with V5 environment (22-dim observation, yellow lost time)

Usage:
  python train_rl_agent.py
  python train_rl_agent.py --steps 1000000 --seed 42
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.config import bootstrap
from src.environment import TrafficSignalEnv
from src.baselines import fixed_time_policy, actuated_policy

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

def evaluate_baseline(cfg, policy, arr_low, arr_high, seeds=range(5), steps=3600):
    """Run a rule-based controller at the given traffic level; return average queue."""
    queues = []
    for seed in seeds:
        env = TrafficSignalEnv(cfg, action_mode="cyclic")
        env.arr_low, env.arr_high = arr_low, arr_high
        obs, _ = env.reset(seed=int(seed))
        for _ in range(steps):
            obs, _, done, truncated, info = env.step(policy(obs, env))
            queues.append(info["avg_queue"])
            if done or truncated:
                break
    return float(np.mean(queues))


def evaluate_baselines_per_stage(cfg):
    """Baseline average queue for every curriculum stage (like-for-like reference)."""
    ref = {"Fixed Timer 30s": [], "Actuated": []}
    for st in CurriculumCallback.STAGES:
        ref["Fixed Timer 30s"].append(evaluate_baseline(
            cfg, lambda o, e: fixed_time_policy(o, e, green=30), st["arr_low"], st["arr_high"]))
        ref["Actuated"].append(evaluate_baseline(
            cfg, actuated_policy, st["arr_low"], st["arr_high"]))
    return ref


# =====================================================================
#  Training Pipeline
# =====================================================================

def train_agent(name, env, total_steps, rl_dir, seed=42, use_curriculum=True, verbose=1):
    """
    Train a PPO agent with optional curriculum learning.

    Args:
        name:            model save name (without extension)
        env:             gymnasium environment instance
        total_steps:     total training timesteps
        rl_dir:          directory to save the trained model
        seed:            random seed (PPO weights, sampling and env)
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
        seed=seed,
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

def generate_plot(baselines, logger_cyclic, logger_free, total_steps, save_path):
    """Generate training progress comparison chart with per-stage baselines."""
    plt.figure(figsize=(12, 7))

    n_stages = len(CurriculumCallback.STAGES)
    bounds = [i * total_steps / n_stages for i in range(n_stages + 1)]
    ref_style = {"Fixed Timer 30s": ("#e74c3c", "--"), "Actuated": ("#8e44ad", "-.")}

    # Baselines evaluated on the same traffic level as each curriculum stage
    for name, values in baselines.items():
        color, ls = ref_style.get(name, ("#555", "--"))
        for i, v in enumerate(values):
            plt.hlines(v, bounds[i], bounds[i + 1], colors=color, linestyles=ls,
                       linewidth=2.5, label=name if i == 0 else None)

    # Agent training curves
    for logger, color, label, lw in [
        (logger_cyclic, "#e67e22", "Cyclic AI (extend / next phase)", 2),
        (logger_free, "#2ecc71", "Free AI (pick any phase)", 2.5),
    ]:
        if logger.history:
            x = np.linspace(0, total_steps, len(logger.history))
            plt.plot(x, logger.history, color=color, linewidth=lw, label=label)

    # Mark curriculum stage transitions
    for i, st in enumerate(CurriculumCallback.STAGES[1:], start=1):
        x = bounds[i]
        plt.axvline(x=x, color="gray", linestyle=":", alpha=0.5)
        plt.text(
            x + total_steps * 0.005, plt.ylim()[1] * 0.92,
            f"Stage {i + 1}: {st['label']}", fontsize=9, color="gray", fontstyle="italic",
        )

    plt.title("Training Progress: Traffic Control Strategies", pad=15, fontweight="bold")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Queue Length (Vehicles)")
    plt.legend(loc="upper left", frameon=True, framealpha=0.9, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Train cyclic and free PPO agents")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--steps", type=int, default=500_000, help="training steps per agent")
    ap.add_argument("--seed", type=int, default=None, help="defaults to training.seed in config")
    ap.add_argument("--output", default=os.path.join("models", "rl_agents"))
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]

    rl_dir = args.output
    os.makedirs(rl_dir, exist_ok=True)

    total_steps = args.steps

    log.info("Starting comparative study: Fixed Timer vs Cyclic AI vs Free AI")
    log.info("Training steps per agent: %s | seed: %d | yellow_time: %s s",
             f"{total_steps:,}", seed, cfg["rl"].get("yellow_time", 0))
    log.info("Curriculum learning: enabled (3 stages)")

    # -- Phase 1: Baseline evaluation (per curriculum stage) --
    log.info("Evaluating baselines for every curriculum stage...")
    baselines = evaluate_baselines_per_stage(cfg)
    for name, values in baselines.items():
        log.info("  %-16s avg queue per stage: %s", name, ", ".join(f"{v:.2f}" for v in values))

    # -- Phase 2: Train Cyclic AI (with curriculum) --
    log.info("Training Cyclic AI (Discrete(2): extend / next phase)...")
    cyclic_env = TrafficSignalEnv(cfg, action_mode="cyclic")
    _, logger_cyclic = train_agent(
        "cyclic_agent", cyclic_env, total_steps, rl_dir, seed=seed, use_curriculum=True
    )
    log.info("Cyclic AI training complete.")

    # -- Phase 3: Train Free AI (with curriculum) --
    log.info("Training Free AI (Discrete(4): pick any phase)...")
    free_env = TrafficSignalEnv(cfg, action_mode="free")
    _, logger_free = train_agent(
        "free_agent", free_env, total_steps, rl_dir, seed=seed, use_curriculum=True
    )
    log.info("Free AI training complete.")

    # -- Phase 4: Generate comparison plot --
    plot_path = os.path.join(rl_dir, "Training_Comparison.png")
    generate_plot(baselines, logger_cyclic, logger_free, total_steps, plot_path)
    log.info("Comparison plot saved: %s", plot_path)

    log.info("All training complete. Models saved to: %s", rl_dir)
    log.info("Next step: python compare_agents.py --agents-dir %s", rl_dir)


if __name__ == "__main__":
    main()