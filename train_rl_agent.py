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
    
    # 2. إنشاء مجلد لحفظ نماذج الذكاء
    rl_models_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_models_dir, exist_ok=True)

    # 3. تهيئة البيئة (Environment)
    log.info("Creating Traffic Signal Environment...")
    env = TrafficSignalEnv(cfg)
    
    # فحص البيئة للتأكد من أنها متوافقة مع المعايير
    check_env(env)
    log.info("Environment is valid! ✅")

    # 4. إعداد موديل الذكاء (PPO Agent)
    # PPO هو أحد أقوى خوارزميات التعلم التعزيزي وأكثرها استقراراً
    model = PPO(
        "MlpPolicy",  # شبكة عصبية عادية (ليست صور) لأن المدخلات أرقام (عدد السيارات)
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        tensorboard_log=cfg["paths"]["log_dir"]
    )

    # 5. بدء التدريب
    log.info("Starting RL Training... (This takes time)")
    # سنحفظ الموديل كل 10,000 خطوة
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=rl_models_dir, 
        name_prefix="ppo_traffic"
    )

    total_steps = 100000  # يمكنك زيادة الرقم لاحقاً لذكاء أكبر
    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

    # 6. حفظ النموذج النهائي
    final_path = os.path.join(rl_models_dir, "final_ppo_agent")
    model.save(final_path)
    log.info(f"Training Finished! Model saved to {final_path}")

if __name__ == "__main__":
    main()