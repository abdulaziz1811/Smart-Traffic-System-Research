import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.config import bootstrap
from src.environment import TrafficSignalEnv

# --- Plotting Configuration (Academic Style) ---
# Sets up professional plotting aesthetics suitable for research papers.
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'lines.linewidth': 2.5,
    'grid.alpha': 0.6
})
sns.set_theme(style="whitegrid", palette="deep")

# ---------------------------------------------------------
# 1. Baseline Controller: Fixed Time Strategy
# ---------------------------------------------------------
class FixedTimeController:
    """
    Simulates a traditional traffic signal controller with fixed timing and cycling.
    Standard sequence: 0 -> 1 -> 2 -> 3.
    """
    def __init__(self, green_duration=30, num_phases=4):
        self.green_duration = green_duration
        self.num_phases = num_phases
        self.current_phase = 0
        self.time_in_phase = 0

    def get_action(self):
        """
        Determines the next phase based on a fixed timer.
        Returns the index of the phase to be active.
        """
        self.time_in_phase += 1
        # Switch phase if duration exceeded
        if self.time_in_phase >= self.green_duration:
            self.time_in_phase = 0
            self.current_phase = (self.current_phase + 1) % self.num_phases
            return self.current_phase 
        
        # Keep current phase
        return self.current_phase 

# ---------------------------------------------------------
# 2. Environment Wrapper: Cyclic AI (Constrained)
# ---------------------------------------------------------
class CyclicTrafficEnv(gym.Wrapper):
    """
    Gym Wrapper that constrains the RL agent to follow a fixed phase order.
    The agent can only decide WHEN to switch, not WHICH phase to select next.
    
    Action Space:
        0: Keep current phase (Extend green time).
        1: Switch to next phase (Force cyclic order).
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(2) 
        self.internal_phase = 0 
        self.num_phases = 4 

    def reset(self, **kwargs):
        self.internal_phase = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # Translate binary action to multi-discrete phase index
        if action == 1:
            self.internal_phase = (self.internal_phase + 1) % self.num_phases
            
        real_action = self.internal_phase
        return self.env.step(real_action)

# ---------------------------------------------------------
# Logging Callback
# ---------------------------------------------------------
class ComparisonLogger(BaseCallback):
    """
    Custom callback to log queue lengths during training for comparative analysis.
    """
    def __init__(self, verbose=0):
        super(ComparisonLogger, self).__init__(verbose)
        self.epoch_queues = []
        self.history = []
        
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        # Safely extract metrics, handling potential missing keys
        val = info.get('queue_len', info.get('system_total_stopped', 0))
        self.epoch_queues.append(val)
        
        # Aggregate data every 1000 steps for smoothing
        if len(self.epoch_queues) >= 1000:
            self.history.append(np.mean(self.epoch_queues))
            self.epoch_queues = []
        return True

# ---------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------
def evaluate_baseline(env, steps=10000):
    """
    Evaluates the Fixed Time Controller to establish a baseline performance metric.
    """
    print(f"Running Baseline Evaluation for {steps} steps...")
    ctrl = FixedTimeController(green_duration=30)
    obs, _ = env.reset()
    queues = []
    
    for _ in range(steps):
        action = ctrl.get_action()
        obs, _, done, truncated, info = env.step(action)
        
        val = info.get('queue_len', info.get('system_total_stopped', 0))
        queues.append(val)
        
        if done or truncated:
            obs, _ = env.reset()
            
    if len(queues) == 0: return 25.0 # Fallback default
    return np.mean(queues)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # Load configuration
    cfg, log, device = bootstrap("configs/config.yaml")
    
    # Setup output directories
    rl_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_dir, exist_ok=True)
    
    log.info("Starting Comparative Study: Fixed Time vs. Cyclic AI vs. Free AI")

    # -----------------------------------------------------
    # Phase 1: Baseline Evaluation (Fixed Time)
    # -----------------------------------------------------
    base_env = TrafficSignalEnv(cfg)
    avg_fixed = evaluate_baseline(base_env, steps=10000)
    log.info(f"Baseline (Fixed Time) Average Queue Length: {avg_fixed:.2f}")

    # Training parameters
    total_steps = 500000 
    
    # -----------------------------------------------------
    # Phase 2: Train System 2 (Cyclic AI)
    # -----------------------------------------------------
    log.info("Training System 2: Cyclic AI (Smart Timing, Fixed Order)...")
    cyclic_env = CyclicTrafficEnv(TrafficSignalEnv(cfg)) 
    
    model_cyclic = PPO("MlpPolicy", cyclic_env, verbose=0, learning_rate=3e-4, device="cpu")
    logger_cyclic = ComparisonLogger()
    
    model_cyclic.learn(total_timesteps=total_steps, callback=logger_cyclic)
    model_cyclic.save(os.path.join(rl_dir, "cyclic_agent"))
    log.info("System 2 Training Completed.")

    # -----------------------------------------------------
    # Phase 3: Train System 3 (Free AI)
    # -----------------------------------------------------
    log.info("Training System 3: Free AI (Fully Adaptive)...")
    free_env = TrafficSignalEnv(cfg) # Original unconstrained environment
    
    model_free = PPO("MlpPolicy", free_env, verbose=0, learning_rate=3e-4, device="cpu")
    logger_free = ComparisonLogger()
    
    model_free.learn(total_timesteps=total_steps, callback=logger_free)
    model_free.save(os.path.join(rl_dir, "free_agent"))
    log.info("System 3 Training Completed.")

    # -----------------------------------------------------
    # Phase 4: Generate Comparative Graph
    # -----------------------------------------------------
    log.info("Generating Comparative Performance Graph...")
    plt.figure(figsize=(12, 7))
    
    # Define X-axis based on logged history
    # Note: We use the length of the Free AI history as reference
    steps_range = np.linspace(0, total_steps, len(logger_free.history))
    
    # 1. Plot Baseline (Horizontal Line)
    plt.axhline(y=avg_fixed, color='#e74c3c', linestyle='--', linewidth=2.5, 
                label=f'Fixed Time Standard (Avg: {avg_fixed:.1f})')
    
    # 2. Plot Cyclic AI (Orange)
    # Adjust length in case of minor mismatches in logging steps
    len_cyc = min(len(steps_range), len(logger_cyclic.history))
    plt.plot(steps_range[:len_cyc], logger_cyclic.history[:len_cyc], 
             color='#e67e22', linewidth=2, label='Cyclic AI (Fixed Order)')
    
    # 3. Plot Free AI (Green)
    len_free = min(len(steps_range), len(logger_free.history))
    plt.plot(steps_range[:len_free], logger_free.history[:len_free], 
             color='#2ecc71', linewidth=2.5, label='Free AI (Fully Adaptive)')

    # Graph Styling
    plt.title('Performance Comparison of Traffic Control Strategies', pad=15, fontweight='bold')
    plt.xlabel('Training Steps (Experience)')
    plt.ylabel('Average Queue Length (Vehicles)')
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save Figure
    plot_path = os.path.join(rl_dir, "Traffic_Comparison_Study.png")
    plt.savefig(plot_path, bbox_inches='tight')
    log.info(f"Graph saved locally at: {plot_path}")

    # Show plot if running in an environment that supports it
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()