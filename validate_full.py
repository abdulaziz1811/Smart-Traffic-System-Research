#!/usr/bin/env python3
"""
Comprehensive RL Validation for Research Paper
===============================================
Runs: 3 baselines + AI agent × 4 scenarios × N seeds
Outputs: Tables, CSVs, and plots ready for the paper.

Usage:
    python scripts/validate_full.py --seeds 10
    python scripts/validate_full.py --seeds 30 --output paper_results/
"""

import argparse, os, json, time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.config import bootstrap
from src.environment import TrafficSignalEnv

# ══════════════════════════════════════════════════════════
#  BASELINES
# ══════════════════════════════════════════════════════════

def fixed_timer_policy(obs, env, cycle_length=30):
    """Fixed-time: switch every `cycle_length` steps regardless of traffic."""
    timer_normalized = obs[env.n_app + env.n_phase]  # normalized timer
    timer_actual = timer_normalized * env.max_green
    return 1 if timer_actual >= cycle_length else 0

def actuated_policy(obs, env):
    """Actuated: switch when active lanes are nearly empty (greedy)."""
    queues = obs[:env.n_app]
    phase = np.argmax(obs[env.n_app : env.n_app + env.n_phase])
    active_lanes = env.green_map[phase]
    active_queue = sum(queues[l] for l in active_lanes)
    timer_normalized = obs[env.n_app + env.n_phase]
    timer_actual = timer_normalized * env.max_green
    # Switch if lanes nearly empty AND min green met
    if active_queue < 0.5 and timer_actual >= env.min_green:
        return 1
    return 0

def random_policy(obs, env, rng):
    """Random baseline: coin flip each step."""
    return rng.integers(2)

# ══════════════════════════════════════════════════════════
#  SCENARIO CONFIGS (override arrival rates)
# ══════════════════════════════════════════════════════════

SCENARIOS = {
    "low_traffic": {
        "desc": "Night/Dawn — light traffic",
        "arrival_low": 0.01,
        "arrival_high": 0.03,
    },
    "medium_traffic": {
        "desc": "Normal daytime traffic",
        "arrival_low": 0.05,
        "arrival_high": 0.10,
    },
    "rush_hour": {
        "desc": "Peak hour — heavy traffic",
        "arrival_low": 0.12,
        "arrival_high": 0.20,
    },
    "asymmetric": {
        "desc": "Main road (N/S heavy) × side road (E/W light)",
        "arrival_low": 0.02,
        "arrival_high": 0.18,
        "asymmetric": True,  # N/S gets high, E/W gets low
    },
}

# ══════════════════════════════════════════════════════════
#  SINGLE EPISODE RUNNER
# ══════════════════════════════════════════════════════════

def run_episode(env, policy_fn, seed, max_steps=3600):
    """Run one full episode and collect metrics."""
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0.0
    total_served = 0.0
    queue_history = []
    wait_history = []
    switches = 0
    
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        queue_history.append(info["avg_queue"])
        total_served = info["served"]
        switches = info["switches"]
        
        # Track cumulative wait from obs
        waits = obs[env.n_app + env.n_phase + 1 + 1:]  # after queues+phase+timer+density
        # Note: waits are not in obs for V3, use info or queues
        
        if terminated or truncated:
            break
    
    return {
        "total_reward": total_reward,
        "avg_queue": np.mean(queue_history),
        "max_queue": np.max(queue_history),
        "std_queue": np.std(queue_history),
        "total_served": total_served,
        "switches": switches,
        "queue_history": queue_history,
    }

# ══════════════════════════════════════════════════════════
#  MAIN VALIDATION
# ══════════════════════════════════════════════════════════

def run_validation(cfg, n_seeds=10, output_dir="outputs/validation"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Try loading trained agent
    ai_model = None
    try:
        from stable_baselines3 import PPO
        for path in ["models/rl_agents/final_ppo_agent", 
                      "models/rl_agents/final_ppo_agent_v2",
                      "models/rl_agents/final_ppo_agent_v3"]:
            if os.path.exists(path + ".zip"):
                ai_model = PPO.load(path)
                print(f"✅ Loaded AI agent from: {path}")
                break
        if ai_model is None:
            print("⚠️  No trained RL agent found! Running baselines only.")
    except ImportError:
        print("⚠️  stable-baselines3 not installed. Running baselines only.")
    
    seeds = list(range(42, 42 + n_seeds))
    
    # Define policies
    policies = {
        "Fixed_30s": lambda obs, env=None, rng=None: fixed_timer_policy(obs, env, 30),
        "Actuated": lambda obs, env=None, rng=None: actuated_policy(obs, env),
        "Random": lambda obs, env=None, rng=None: random_policy(obs, env, rng),
    }
    if ai_model:
        policies["AI_Agent"] = lambda obs, env=None, rng=None: int(ai_model.predict(obs, deterministic=True)[0]) # type: ignore
    
    # ── Run all combinations ──
    all_results = {}  # {scenario: {policy: [metrics_per_seed]}}
    
    for sc_name, sc_cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"📊 Scenario: {sc_name} — {sc_cfg['desc']}")
        print(f"{'='*60}")
        
        all_results[sc_name] = {}
        
        for pol_name, pol_fn in policies.items():
            print(f"  🔄 Running {pol_name} ({n_seeds} seeds)...", end=" ", flush=True)
            t0 = time.time()
            
            seed_results = []
            for seed in seeds:
                # Create fresh env with scenario-specific arrivals
                env = TrafficSignalEnv(cfg)
                env.arr_low = sc_cfg["arrival_low"] # type: ignore
                env.arr_high = sc_cfg["arrival_high"] # type: ignore
                
                rng = np.random.default_rng(seed)
                
                # Wrap policy to pass env
                def wrapped_policy(obs, _env=env, _rng=rng, _fn=pol_fn):
                    return _fn(obs, env=_env, rng=_rng)
                
                metrics = run_episode(env, wrapped_policy, seed)
                seed_results.append(metrics)
            
            all_results[sc_name][pol_name] = seed_results # type: ignore
            
            avg_q = np.mean([r["avg_queue"] for r in seed_results])
            print(f"done ({time.time()-t0:.1f}s) — avg_queue={avg_q:.2f}")
    
    # ── Generate Tables ──
    print("\n\n" + "="*80)
    print("📋 RESULTS TABLES (Copy to paper)")
    print("="*80)
    
    metrics_to_show = [
        ("avg_queue", "Avg Queue ↓", False),
        ("max_queue", "Max Queue ↓", False),
        ("total_served", "Served ↑", True),
        ("total_reward", "Reward ↑", True),
        ("switches", "Switches", False),
    ]
    
    # Table per scenario
    all_tables = {}
    for sc_name in SCENARIOS:
        print(f"\n── {sc_name}: {SCENARIOS[sc_name]['desc']} ──")
        header = f"{'Metric':<20s}"
        for pol in policies:
            header += f" | {pol:>18s}"
        print(header)
        print("-" * len(header))
        
        table_data = {}
        for metric_key, metric_label, higher_better in metrics_to_show:
            row = f"{metric_label:<20s}"
            values = {}
            for pol_name in policies:
                vals = [r[metric_key] for r in all_results[sc_name][pol_name]] # type: ignore
                mean, std = np.mean(vals), np.std(vals)
                values[pol_name] = (mean, std)
                row += f" | {mean:>9.2f} ± {std:>5.2f}"
            print(row)
            table_data[metric_key] = values
        
        all_tables[sc_name] = table_data
    
    # ── Improvement Summary ──
    if ai_model:
        print(f"\n\n── IMPROVEMENT vs Fixed Timer ──")
        for sc_name in SCENARIOS:
            fixed_q = np.mean([r["avg_queue"] for r in all_results[sc_name]["Fixed_30s"]]) # type: ignore
            ai_q = np.mean([r["avg_queue"] for r in all_results[sc_name]["AI_Agent"]]) # type: ignore
            improvement = (fixed_q - ai_q) / fixed_q * 100
            print(f"  {sc_name:<20s}: {improvement:>+.1f}% queue reduction")
    
    # ── Save Raw Data ──
    save_data: dict = {}
    for sc in all_results:
        save_data[sc] = {}
        for pol in all_results[sc]: # type: ignore
            save_data[sc][pol] = [] # type: ignore
            for r in all_results[sc][pol]: # type: ignore
                save_data[sc][pol].append({ # type: ignore
                    k: v for k, v in r.items() if k != "queue_history"
                })
    
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n💾 Raw data saved to {output_dir}/validation_results.json")
    
    # ── Generate Plots ──
    generate_plots(all_results, policies, output_dir)
    
    return all_results

# ══════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════

def generate_plots(all_results, policies, output_dir):
    colors = {
        "Fixed_30s": "#e74c3c",
        "Actuated": "#f39c12", 
        "Random": "#95a5a6",
        "AI_Agent": "#2ecc71",
    }
    
    # ── Plot 1: Bar chart comparison (avg_queue across scenarios) ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle("Average Queue Length Comparison", fontsize=14, fontweight="bold")
    
    for idx, (sc_name, sc_cfg) in enumerate(SCENARIOS.items()):
        ax = axes[idx] # type: ignore
        names, means, stds = [], [], []
        for pol_name in policies:
            vals = [r["avg_queue"] for r in all_results[sc_name][pol_name]]
            names.append(pol_name.replace("_", "\n"))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        bars = ax.bar(names, means, yerr=stds, capsize=4,
                      color=[colors.get(p, "#333") for p in policies])
        ax.set_title(sc_name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Avg Queue Length" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"📊 Saved: {path}")
    plt.close()
    
    # ── Plot 2: Queue over time (rush hour, seed=42) ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for pol_name in policies:
        history = all_results["rush_hour"][pol_name][0]["queue_history"]
        alpha = 1.0 if pol_name == "AI_Agent" else 0.5
        lw = 2.5 if pol_name == "AI_Agent" else 1.2
        # Smooth with rolling average
        window = 50
        if len(history) > window:
            smoothed = np.convolve(history, np.ones(window)/window, mode="valid")
        else:
            smoothed = history
        ax.plot(smoothed, label=pol_name.replace("_", " "), 
                color=colors.get(pol_name, "#333"), alpha=alpha, linewidth=lw)
    
    ax.set_title("Rush Hour: Queue Length Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Simulation Step (seconds)")
    ax.set_ylabel("Avg Queue Length (vehicles)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, "fig_queue_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"📊 Saved: {path}")
    plt.close()
    
    # ── Plot 3: Improvement heatmap ──
    if "AI_Agent" in policies:
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_for_heatmap = ["avg_queue", "max_queue", "total_served"]
        baselines_for_heatmap = ["Fixed_30s", "Actuated", "Random"]
        
        improvements = np.zeros((len(baselines_for_heatmap), len(SCENARIOS)))
        
        for i, baseline in enumerate(baselines_for_heatmap):
            for j, sc_name in enumerate(SCENARIOS):
                bl_val = np.mean([r["avg_queue"] for r in all_results[sc_name][baseline]])
                ai_val = np.mean([r["avg_queue"] for r in all_results[sc_name]["AI_Agent"]])
                improvements[i, j] = (bl_val - ai_val) / max(bl_val, 0.01) * 100
        
        im = ax.imshow(improvements, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=100)
        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=9)
        ax.set_yticks(range(len(baselines_for_heatmap)))
        ax.set_yticklabels([b.replace("_", " ") for b in baselines_for_heatmap])
        
        for i in range(len(baselines_for_heatmap)):
            for j in range(len(SCENARIOS)):
                ax.text(j, i, f"{improvements[i,j]:.0f}%", ha="center", va="center", fontsize=11)
        
        ax.set_title("Queue Reduction vs Baselines (%)", fontweight="bold")
        plt.colorbar(im, ax=ax, label="Improvement %")
        
        path = os.path.join(output_dir, "fig_improvement_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved: {path}")
        plt.close()

# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Full RL Validation for Paper")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    ap.add_argument("--output", default="outputs/validation")
    args = ap.parse_args()
    
    cfg, log, device = bootstrap(args.config)
    
    print(f"🔬 Starting Comprehensive Validation")
    print(f"   Seeds: {args.seeds}")
    print(f"   Scenarios: {len(SCENARIOS)}")
    print(f"   Output: {args.output}")
    print()
    
    results = run_validation(cfg, n_seeds=args.seeds, output_dir=args.output)
    
    print(f"\n✅ Validation complete! Results in {args.output}/")

if __name__ == "__main__":
    main()