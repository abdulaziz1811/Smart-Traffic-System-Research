import yaml
from stable_baselines3 import PPO
from src.environment import TrafficSignalEnv
import os

# 1. تحميل الإعدادات
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. تجهيز البيئة
env = TrafficSignalEnv(config)

# 3. تحميل الموديل الذي دربته للتو
# تأكد أن الاسم مطابق للملف الموجود عندك
model_path = "models/rl_agents/final_ppo_agent_v2"

print(f"🔄 Loading Improved V2 model from: {model_path}")
model = PPO.load(model_path, env=env)

# 4. إكمال التدريب
# سنقوم بتدريبه لـ 200,000 خطوة إضافية
# ملاحظة: الموديل سيبدأ بذكاء عالي ولن يبدأ عشوائياً
additional_steps = 200000

print("🚀 Starting Fine-Tuning (Continuing Training)...")
model.learn(total_timesteps=additional_steps, progress_bar=True)

# 5. حفظ الموديل الجديد باسم مختلف (عشان ما تخسر القديم)
new_model_path = "models/rl_agents/final_ppo_agent_v3"
model.save(new_model_path)
print(f"✅ Super-Improved model saved to: {new_model_path}")