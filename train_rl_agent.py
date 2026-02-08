import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from src.config import bootstrap
from src.environment import TrafficSignalEnv

def main():
    # 1. إعداد المشروع
    cfg, log, device = bootstrap("configs/config.yaml")
    rl_cfg = cfg["rl"]  # قراءة قسم RL الجديد

    # 2. إنشاء مجلد لحفظ نماذج الذكاء
    rl_models_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_models_dir, exist_ok=True)

    # 3. تهيئة البيئة
    log.info("Creating Traffic Signal Environment...")
    env = TrafficSignalEnv(cfg)
    check_env(env)
    log.info("Environment is valid! ✅")

    # 4. إعداد موديل الذكاء (PPO Agent) باستخدام القيم من Config
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=rl_cfg["learning_rate"],
        gamma=rl_cfg["gamma"],
        n_steps=rl_cfg.get("n_steps", 2048),
        batch_size=rl_cfg.get("batch_size", 64),
        tensorboard_log=cfg["paths"]["log_dir"]
    )

    # 5. بدء التدريب
    total_steps = rl_cfg["total_timesteps"]
    log.info(f"Starting RL Training for {total_steps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=rl_models_dir, 
        name_prefix="ppo_traffic"
    )

    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

    # 6. حفظ النموذج النهائي
    final_path = os.path.join(rl_models_dir, "final_ppo_agent")
    model.save(final_path)
    log.info(f"Training Finished! Model saved to {final_path}")

if __name__ == "__main__":
    main()