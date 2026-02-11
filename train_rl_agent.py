import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from src.config import bootstrap
from src.environment import TrafficSignalEnv

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    cfg, log, device = bootstrap("configs/config.yaml")
    set_random_seed(42)

    rl_models_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_models_dir, exist_ok=True)
    
    # Archive old model
    if os.path.exists(os.path.join(rl_models_dir, "final_ppo_agent.zip")):
        os.rename(
            os.path.join(rl_models_dir, "final_ppo_agent.zip"),
            os.path.join(rl_models_dir, "final_ppo_agent_OLD_CYCLE.zip")
        )

    log.info("Creating Realistic Cyclic Environment...")
    env = TrafficSignalEnv(cfg)

    # Larger network for better logical reasoning
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(0.0003),
        gamma=0.99,
        n_steps=2048,
        batch_size=128, # Increased for stability
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg["paths"]["log_dir"],
        device="cpu"
    )

    # Train for enough steps to learn the cycles
    total_steps = 600000 
    log.info(f"Starting Logic-Based Training for {total_steps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=rl_models_dir, 
        name_prefix="ppo_logic"
    )

    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

    final_path = os.path.join(rl_models_dir, "final_ppo_agent")
    model.save(final_path)
    log.info(f"Training Finished! Logic-Agent saved to {final_path}")

if __name__ == "__main__":
    main()