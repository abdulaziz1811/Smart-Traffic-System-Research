import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from src.config import bootstrap
from src.environment import TrafficSignalEnv

# دالة لتغيير سرعة التعلم تدريجياً (Smart Scheduler)
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    # 1. إعداد المشروع
    cfg, log, device = bootstrap("configs/config.yaml")
    
    # تثبيت العشوائية لنتائج قابلة للتكرار
    set_random_seed(42)

    # 2. تجهيز المجلدات
    rl_models_dir = os.path.join("models", "rl_agents")
    os.makedirs(rl_models_dir, exist_ok=True)
    
    # **مهم:** احفظ الموديل القديم باسم آخر عشان ما يضيع!
    if os.path.exists(os.path.join(rl_models_dir, "final_ppo_agent.zip")):
        os.rename(
            os.path.join(rl_models_dir, "final_ppo_agent.zip"),
            os.path.join(rl_models_dir, "final_ppo_agent_OLD_1560.zip")
        )

    # 3. تهيئة البيئة
    log.info("Creating Traffic Signal Environment (Pro Mode)...")
    env = TrafficSignalEnv(cfg)

    # 4. إعدادات الشبكة العصبية "العميقة" (Deep Network)
    # [256, 256] تعني طبقتين كل واحدة فيها 256 عصب صناعي (دماغ أكبر بـ 4 أضعاف)
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # 5. إعداد الموديل بإعدادات احترافية
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(0.0003),
        gamma=0.99,
        n_steps=4096,
        batch_size=128,
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg["paths"]["log_dir"],
        device="cpu"
    )

    # 6. بدء التدريب الطويل (500,000 خطوة)
    # سيستغرق وقتاً أطول (20-40 دقيقة) لكن النتيجة تستاهل!
    total_steps = 500000 
    log.info(f"🚀 Starting PRO Training for {total_steps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=rl_models_dir, 
        name_prefix="ppo_pro"
    )

    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

    # 7. حفظ النموذج النهائي
    final_path = os.path.join(rl_models_dir, "final_ppo_agent")
    model.save(final_path)
    log.info(f"🏆 Training Finished! Super-Model saved to {final_path}")

if __name__ == "__main__":
    main()