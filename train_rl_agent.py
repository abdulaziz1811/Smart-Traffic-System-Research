#!/usr/bin/env python3
"""
Comparative Training: Fixed Timer vs Cyclic AI vs Free AI
==========================================================
Trains two PPO agents (cyclic-constrained and unconstrained),
evaluates against a fixed-timer baseline, and produces a
training progress comparison chart.

Key features:
  - Cyclic agent = Discrete(2) extend/next, Free agent = Discrete(4) pick
    any phase
  - Decision steps: the agent only acts when its action can matter
    (DecisionStepWrapper skips yellow and min-green seconds)
  - VecNormalize (observations + rewards) and 4 parallel environments;
    the statistics are saved next to each model (<name>_vecnormalize.pkl)
  - reward_type from config ("queue" = total delay, the evaluation metric)
  - Optional curriculum learning (3 traffic levels); baselines are evaluated
    on the SAME traffic level as each stage for a like-for-like plot
  - Optional imitation warm start (--bc-steps): the policy first imitates
    the actuated (cyclic) / longest-queue (free) controller, then PPO
    fine-tunes it with a lower learning rate
  - Seeded for reproducibility; V6 environment (31-dim observation)

Usage:
  python train_rl_agent.py
  python train_rl_agent.py --steps 2000000 --modes free --curriculum
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.config import bootstrap
from src.environment import TrafficSignalEnv
from src.baselines import fixed_time_policy, actuated_policy, longest_queue_policy
from src.agents import make_training_env, save_agent, behavior_cloning

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
        {"arr_low": 0.02, "arr_high": 0.06, "label": "Light"},
        {"arr_low": 0.02, "arr_high": 0.12, "label": "Medium"},
        {"arr_low": 0.01, "arr_high": 0.17, "label": "Heavy"},
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

            # Applies to every (sub-process) environment from its next episode
            self.training_env.env_method("set_arrival_range", params["arr_low"], params["arr_high"])

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

def train_agent(name, cfg, action_mode, total_steps, rl_dir, seed=42, use_curriculum=False,
                n_envs=4, gamma=0.99, bc_steps=0, dagger_rounds=2, finetune_lr=1e-4, clip_range=0.2,
                decision_steps=False, verbose=1):
    """
    Train a PPO agent (decision steps + VecNormalize + parallel envs).

    Args:
        name:            model save name (without extension)
        cfg:             configuration dict
        action_mode:     "cyclic" or "free"
        total_steps:     total training timesteps (agent decisions)
        rl_dir:          directory to save the trained model
        seed:            random seed (PPO weights, sampling and env)
        use_curriculum:  whether to apply curriculum learning
        n_envs:          parallel environments
        gamma:           discount factor per decision
        bc_steps:        expert decisions for the imitation warm start (0 = off)
        dagger_rounds:   DAgger rounds after the first imitation fit
        finetune_lr:     PPO learning rate after the warm start
        clip_range:      PPO clip range (smaller = stay closer to the warm start)
        decision_steps:  query the agent only at decision points (see module doc)
        verbose:         curriculum callback verbosity

    Returns:
        (model, queue_logger)
    """
    env = make_training_env(cfg, action_mode, n_envs=n_envs, seed=seed, gamma=gamma,
                            decision_steps=decision_steps)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=finetune_lr if bc_steps else 3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        ent_coef=0.0 if bc_steps else 0.01,
        clip_range=clip_range,
        policy_kwargs=dict(net_arch=[128, 128]),
        device="cpu",
        seed=seed,
    )

    if bc_steps:
        expert = actuated_policy if action_mode == "cyclic" else longest_queue_policy
        acc = behavior_cloning(model, env, cfg, action_mode, expert, n_steps=bc_steps,
                               dagger_rounds=dagger_rounds)
        print(f"  [Imitation] {action_mode}: policy matches the expert on {acc:.1%} of decisions")

    queue_log = QueueLogger()
    callbacks: list[BaseCallback] = [queue_log]

    if use_curriculum:
        callbacks.append(CurriculumCallback(total_steps, verbose=verbose))

    model.learn(total_timesteps=total_steps, callback=callbacks)
    save_agent(model, os.path.join(rl_dir, name))
    env.close()

    return model, queue_log


# =====================================================================
#  Plot Generation
# =====================================================================

def generate_plot(baselines, logger_cyclic, logger_free, total_steps, save_path, show_stages=True):
    """Generate training progress comparison chart with like-for-like baselines."""
    plt.figure(figsize=(12, 7))

    n_stages = len(CurriculumCallback.STAGES)
    bounds = [i * total_steps / n_stages for i in range(n_stages + 1)]
    ref_style = {"Fixed Timer 30s": ("#e74c3c", "--"), "Actuated": ("#8e44ad", "-.")}

    # Baselines evaluated on the same traffic level the agent was trained on
    medium = [s["label"] for s in CurriculumCallback.STAGES].index("Medium")
    for name, values in baselines.items():
        color, ls = ref_style.get(name, ("#555", "--"))
        if show_stages:
            for i, v in enumerate(values):
                plt.hlines(v, bounds[i], bounds[i + 1], colors=color, linestyles=ls,
                           linewidth=2.5, label=name if i == 0 else None)
        else:
            plt.axhline(values[medium], color=color, linestyle=ls, linewidth=2.5, label=name)

    # Agent training curves
    for logger, color, label, lw in [
        (logger_cyclic, "#e67e22", "Cyclic AI (extend / next phase)", 2),
        (logger_free, "#2ecc71", "Free AI (pick any phase)", 2.5),
    ]:
        if logger.history:
            x = np.linspace(0, total_steps, len(logger.history))
            plt.plot(x, logger.history, color=color, linewidth=lw, label=label)

    # Mark curriculum stage transitions
    for i, st in enumerate(CurriculumCallback.STAGES[1:] if show_stages else [], start=1):
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
    ap.add_argument("--steps", type=int, default=1_000_000, help="PPO fine-tuning steps per agent")
    ap.add_argument("--seed", type=int, default=None, help="defaults to training.seed in config")
    ap.add_argument("--modes", default="cyclic,free", help="comma-separated: cyclic,free")
    ap.add_argument("--curriculum", action="store_true", help="3-stage traffic curriculum")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--bc-steps", type=int, default=50_000,
                    help="imitation warm start from the rule-based expert (0 = PPO from scratch)")
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--decision-steps", action="store_true",
                    help="query the agent only at decision points (biased discounting, see doc)")
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--output", default=os.path.join("models", "rl_agents"))
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rl_dir = args.output
    os.makedirs(rl_dir, exist_ok=True)

    total_steps = args.steps

    log.info("Training PPO agents: %s", ", ".join(modes))
    log.info("Steps per agent: %s | seed: %d | reward: %s | yellow: %ss | curriculum: %s | "
             "imitation: %s decisions + %d DAgger rounds",
             f"{total_steps:,}", seed, cfg["rl"].get("reward_type", "shaped"),
             cfg["rl"].get("yellow_time", 0), "on" if args.curriculum else "off",
             f"{args.bc_steps:,}", args.dagger_rounds if args.bc_steps else 0)

    # -- Baseline evaluation (per curriculum stage, for the plot) --
    log.info("Evaluating baselines for every curriculum stage...")
    baselines = evaluate_baselines_per_stage(cfg)
    for name, values in baselines.items():
        log.info("  %-16s avg queue per stage: %s", name, ", ".join(f"{v:.2f}" for v in values))

    loggers = {}
    for mode in modes:
        label = "Cyclic AI (Discrete(2): extend / next phase)" if mode == "cyclic" \
            else "Free AI (Discrete(4): pick any phase)"
        log.info("Training %s...", label)
        _, loggers[mode] = train_agent(
            f"{mode}_agent", cfg, mode, total_steps, rl_dir, seed=seed,
            use_curriculum=args.curriculum, n_envs=args.n_envs, gamma=args.gamma,
            bc_steps=args.bc_steps, dagger_rounds=args.dagger_rounds, finetune_lr=args.finetune_lr,
            clip_range=args.clip_range, decision_steps=args.decision_steps,
        )
        log.info("%s training complete.", mode)

    # -- Comparison plot --
    plot_path = os.path.join(rl_dir, "Training_Comparison.png")
    generate_plot(baselines, loggers.get("cyclic", QueueLogger()), loggers.get("free", QueueLogger()),
                  total_steps, plot_path, show_stages=args.curriculum)
    log.info("Comparison plot saved: %s", plot_path)

    log.info("All training complete. Models saved to: %s", rl_dir)
    log.info("Next step: python compare_agents.py --agents-dir %s", rl_dir)


if __name__ == "__main__":
    main()