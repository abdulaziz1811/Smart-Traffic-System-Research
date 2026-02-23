#!/usr/bin/env python3
"""
3-Agent Comparison Benchmark
==============================
Compares 4 traffic control strategies on identical seeds:
  1. Fixed Timer    -- 30s green per phase (traditional baseline)
  2. Cyclic Agent   -- PPO with cyclic constraint
  3. Free Agent     -- PPO unconstrained
  4. Final PPO      -- PPO with larger network [256,256]

Outputs:
  outputs/comparison/fig1_bar_comparison.png     Bar chart (3 metrics)
  outputs/comparison/fig2_queue_timeline.png     Queue over time
  outputs/comparison/fig3_combined_paper.png     Combined figure for paper
  outputs/comparison/fig4_box_whisker.png        Seed variance
  outputs/comparison/results.json                Raw numeric results

Usage:
  python compare_agents.py
  python compare_agents.py --seeds 20 --steps 3600
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import bootstrap
from src.environment import TrafficSignalEnv

# -- Academic plot style -----------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Fixed Timer":    "#e74c3c",
    "Cyclic Agent":   "#e67e22",
    "Free Agent":     "#3498db",
    "Final PPO":      "#27ae60",
}


# =====================================================================
#  Fixed Timer Baseline (correct: returns 0 or 1, not phase index)
# =====================================================================

def fixed_timer_action(step, cycle_length=30):
    """
    Traditional fixed-cycle controller.
    Returns 1 (switch) every cycle_length steps, 0 (extend) otherwise.
    """
    if step > 0 and step % cycle_length == 0:
        return 1
    return 0


# =====================================================================
#  Episode Runner
# =====================================================================

def run_episode(cfg, get_action, seed, max_steps=3600):
    """
    Run a single episode and collect performance metrics.

    Args:
        cfg:        configuration dict (creates a fresh env each call)
        get_action: callable(obs, step) -> int (0 or 1)
        seed:       random seed for reproducibility
        max_steps:  maximum episode length

    Returns:
        dict with total_reward, avg_queue, max_queue, total_served,
        switches, and full queue_history list.
    """
    env = TrafficSignalEnv(cfg)
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    queue_history = []
    info = {"served": 0.0, "switches": 0, "avg_queue": 0.0}

    for step in range(max_steps):
        action = get_action(obs, step)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        queue_history.append(info["avg_queue"])

        if terminated or truncated:
            break

    return {
        "total_reward":  float(total_reward),
        "avg_queue":     float(np.mean(queue_history)),
        "max_queue":     float(np.max(queue_history)),
        "total_served":  float(info["served"]),
        "switches":      float(info["switches"]),
        "queue_history": queue_history,
    }


# =====================================================================
#  Multi-Seed Evaluation
# =====================================================================

def evaluate(cfg, get_action, name, seeds, max_steps):
    """Run agent across N seeds and aggregate statistics."""
    all_runs = []
    t0 = time.time()

    for seed in seeds:
        r = run_episode(cfg, get_action, seed, max_steps)
        all_runs.append(r)

    def stat(key):
        vals = [r[key] for r in all_runs]
        return float(np.mean(vals)), float(np.std(vals))

    aq_m, aq_s = stat("avg_queue")
    mq_m, mq_s = stat("max_queue")
    sv_m, sv_s = stat("total_served")
    rw_m, rw_s = stat("total_reward")
    sw_m, sw_s = stat("switches")
    elapsed = time.time() - t0

    print(f"  [OK] {name:<16s} | AvgQ: {aq_m:5.2f} +/- {aq_s:<5.2f} | "
          f"Served: {sv_m:7.0f} | Reward: {rw_m:8.0f} | {elapsed:.1f}s")

    return {
        "name": name,
        "avg_queue_mean": aq_m,  "avg_queue_std": aq_s,
        "max_queue_mean": mq_m,  "max_queue_std": mq_s,
        "served_mean": sv_m,     "served_std": sv_s,
        "reward_mean": rw_m,     "reward_std": rw_s,
        "switches_mean": sw_m,   "switches_std": sw_s,
        "n_seeds": len(seeds),
        "runs": all_runs,
    }


# =====================================================================
#  Plot 1: Bar Chart (3 metrics side by side)
# =====================================================================

def plot_bars(summaries, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [s["name"] for s in summaries]
    colors = [COLORS.get(n, "#555") for n in names]
    x = np.arange(len(names))

    metrics = [
        ("avg_queue", "Avg Queue Length (vehicles)", "Average Queue Length"),
        ("served",    "Vehicles Served",             "Total Throughput"),
        ("reward",    "Cumulative Reward",           "Total Reward"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        m = [s[f"{key}_mean"] for s in summaries]
        e = [s[f"{key}_std"]  for s in summaries]
        ax.bar(x, m, yerr=e, capsize=5, color=colors,
               alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        for i, (v, s) in enumerate(zip(m, e)):
            ax.text(i, v + s + abs(v) * 0.02 + 0.3, f"{v:.1f}",
                    ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_bar_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {path}")


# =====================================================================
#  Plot 2: Queue Timeline (first seed, smoothed)
# =====================================================================

def plot_timeline(summaries, out_dir, window=50):
    fig, ax = plt.subplots(figsize=(13, 5))

    for s in summaries:
        name = s["name"]
        h = s["runs"][0]["queue_history"]
        if len(h) > window:
            sm = np.convolve(h, np.ones(window) / window, mode="valid")
        else:
            sm = h
        lw = 3.0 if "Final" in name else 1.8
        alpha = 1.0 if "Final" in name else 0.65
        ax.plot(sm, label=name, color=COLORS.get(name, "#555"),
                linewidth=lw, alpha=alpha)

    ax.set_title("Queue Length Over Time (seed=42)", fontweight="bold")
    ax.set_xlabel("Simulation Step (seconds)")
    ax.set_ylabel("Avg Queue Length (vehicles)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, shadow=True)
    ax.set_xlim(0)
    ax.set_ylim(0)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_queue_timeline.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {path}")


# =====================================================================
#  Plot 3: Combined Figure (paper-ready)
# =====================================================================

def plot_combined(summaries, out_dir, window=50):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.3)

    names = [s["name"] for s in summaries]
    colors = [COLORS.get(n, "#555") for n in names]
    x = np.arange(len(names))

    # (a) Avg Queue
    ax = fig.add_subplot(gs[0, 0])
    m = [s["avg_queue_mean"] for s in summaries]
    e = [s["avg_queue_std"]  for s in summaries]
    ax.bar(x, m, yerr=e, capsize=5, color=colors,
           alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.set_title("(a) Average Queue Length", fontweight="bold")
    ax.set_ylabel("Vehicles")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    for i, (v, s) in enumerate(zip(m, e)):
        ax.text(i, v + s + 0.2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # (b) Throughput
    ax = fig.add_subplot(gs[0, 1])
    m = [s["served_mean"] for s in summaries]
    e = [s["served_std"]  for s in summaries]
    ax.bar(x, m, yerr=e, capsize=5, color=colors,
           alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.set_title("(b) Total Throughput", fontweight="bold")
    ax.set_ylabel("Vehicles Served")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    for i, (v, s) in enumerate(zip(m, e)):
        ax.text(i, v + s + 3, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

    # (c) Timeline
    ax = fig.add_subplot(gs[1, :])
    for s in summaries:
        name = s["name"]
        h = s["runs"][0]["queue_history"]
        if len(h) > window:
            sm = np.convolve(h, np.ones(window) / window, mode="valid")
        else:
            sm = h
        lw = 3.0 if "Final" in name else 1.5
        alpha = 1.0 if "Final" in name else 0.6
        ax.plot(sm, label=name, color=COLORS.get(name, "#555"),
                linewidth=lw, alpha=alpha)
    ax.set_title("(c) Queue Length Over Time", fontweight="bold")
    ax.set_xlabel("Simulation Step (seconds)")
    ax.set_ylabel("Avg Queue Length")
    ax.legend(frameon=True, framealpha=0.9, shadow=True)
    ax.set_xlim(0)
    ax.set_ylim(0)

    fig.suptitle("Comparative Evaluation of Traffic Signal Control Strategies",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out_dir, "fig3_combined_paper.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {path}")


# =====================================================================
#  Plot 4: Box-Whisker (seed variance)
# =====================================================================

def plot_boxes(summaries, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [
        (axes[0], "avg_queue",    "Avg Queue Length"), # type: ignore
        (axes[1], "total_served", "Total Throughput"), # type: ignore
    ]:
        data, tick_labels, clrs = [], [], []
        for s in summaries:
            data.append([r[key] for r in s["runs"]])
            tick_labels.append(s["name"])
            clrs.append(COLORS.get(s["name"], "#555"))

        bp = ax.boxplot(
            data, tick_labels=tick_labels, patch_artist=True, widths=0.6,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_box_whisker.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {path}")


# =====================================================================
#  Results Table
# =====================================================================

def print_results(summaries):
    fixed = next((s for s in summaries if "Fixed" in s["name"]), None)

    n = summaries[0]["n_seeds"]
    print(f"\n{'=' * 72}")
    print(f"  RESULTS ({n} seeds)")
    print(f"{'=' * 72}")
    print(f"  {'Strategy':<16s} | {'Avg Queue':>13s} | {'Max Queue':>13s} | "
          f"{'Served':>11s} | {'Switches':>10s}")
    print(f"  {'-' * 16}-+-{'-' * 13}-+-{'-' * 13}-+-{'-' * 11}-+-{'-' * 10}")

    for s in summaries:
        print(f"  {s['name']:<16s} | "
              f"{s['avg_queue_mean']:6.2f} +/- {s['avg_queue_std']:<5.2f} | "
              f"{s['max_queue_mean']:6.1f} +/- {s['max_queue_std']:<5.1f} | "
              f"{s['served_mean']:6.0f}+/-{s['served_std']:<4.0f} | "
              f"{s['switches_mean']:5.0f}+/-{s['switches_std']:<4.0f}")

    if fixed:
        print(f"\n  Improvement vs Fixed Timer:")
        for s in summaries:
            if "Fixed" in s["name"]:
                continue
            q = ((fixed["avg_queue_mean"] - s["avg_queue_mean"])
                 / max(fixed["avg_queue_mean"], 0.01) * 100)
            v = ((s["served_mean"] - fixed["served_mean"])
                 / max(fixed["served_mean"], 1) * 100)
            print(f"    {s['name']:<16s} -> Queue: {q:+.1f}%  |  Served: {v:+.1f}%")
    print()


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Compare 3 RL agents + fixed baseline")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seeds",  type=int, default=10)
    ap.add_argument("--steps",  type=int, default=3600)
    ap.add_argument("--output", default="outputs/comparison")
    args = ap.parse_args()

    cfg, log, _ = bootstrap(args.config)
    os.makedirs(args.output, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))

    print(f"\n{'=' * 60}")
    print(f"  3-Agent Comparison Benchmark")
    print(f"  Seeds: {args.seeds}  |  Steps/episode: {args.steps}")
    print(f"{'=' * 60}\n")

    summaries = []

    # -- 1. Fixed Timer baseline --
    print("[1/4] Fixed Timer (30s cycle)...")
    summaries.append(evaluate(
        cfg,
        lambda obs, step: fixed_timer_action(step, cycle_length=30),
        "Fixed Timer", seeds, args.steps,
    ))

    # -- 2-4. Load RL agents --
    from stable_baselines3 import PPO

    agents = [
        ("models/rl_agents/cyclic_agent",    "Cyclic Agent"),
        ("models/rl_agents/free_agent",      "Free Agent"),
        ("models/rl_agents/final_ppo_agent", "Final PPO"),
    ]

    for i, (path, name) in enumerate(agents, start=2):
        zip_path = path + ".zip"
        if not os.path.exists(zip_path):
            print(f"  [SKIP] {zip_path} not found")
            continue

        print(f"[{i}/4] {name}...")
        model = PPO.load(path)

        # Check observation space compatibility
        env_dim = TrafficSignalEnv(cfg).observation_space.shape[0] # type: ignore
        model_dim = model.observation_space.shape[0] # type: ignore
        if env_dim != model_dim:
            print(f"  [SKIP] {name}: obs mismatch (env={env_dim}, model={model_dim}). "
                  f"Retrain with new environment.")
            continue

        summaries.append(evaluate(
            cfg,
            lambda obs, step, m=model: int(m.predict(obs, deterministic=True)[0]),
            name, seeds, args.steps,
        ))

    # -- Results --
    print_results(summaries)

    # -- Plots --
    print("Generating figures...")
    plot_bars(summaries, args.output)
    plot_timeline(summaries, args.output)
    plot_combined(summaries, args.output)
    plot_boxes(summaries, args.output)

    # -- Save JSON (convert all values to native Python types) --
    save = []
    for s in summaries:
        d = {k: v for k, v in s.items() if k != "runs"}
        d["per_seed_avg_queue"] = [float(r["avg_queue"]) for r in s["runs"]] # type: ignore
        d["per_seed_served"]    = [float(r["total_served"]) for r in s["runs"]] # type: ignore
        save.append(d)

    jpath = os.path.join(args.output, "results.json")
    with open(jpath, "w") as f:
        json.dump(save, f, indent=2)

    print(f"\n  Results: {jpath}")
    print(f"  Figures: {args.output}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()