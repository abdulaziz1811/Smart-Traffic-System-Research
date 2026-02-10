import yaml
import numpy as np
from stable_baselines3 import PPO
from src.environment import TrafficSignalEnv
import torch

# دالة لتحديث معدل التعلم (Learning Rate Schedule)
# نبدأ برقم صغير ونقلله أكثر كلما زادت الصعوبة
def get_learning_rate(progress_remaining):
    return 0.0001 * progress_remaining  # يبدأ بـ 1e-4 وينتهي بـ 0

def run_training_stage(stage_name, model_path, save_path, traffic_config, steps=100000):
    print(f"\n🔥🔥 STARTING STAGE: {stage_name} 🔥🔥")
    
    # 1. تحميل الإعدادات الأساسية
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. تطبيق تعديلات المرحلة الحالية (Override)
    config['rl'].update(traffic_config)
    print(f"⚙️ Config: Low={config['rl']['arrival_rate_low']}, High={config['rl']['arrival_rate_high']}")
    
    # 3. تجهيز البيئة
    env = TrafficSignalEnv(config)
    
    # 4. تحميل الموديل السابق
    print(f"🔄 Loading model: {model_path}")
    # نستخدم learning_rate مخصص لهذه المرحلة ليكون التدريب دقيقاً
    custom_lr = 0.00005  # معدل بطيء جداً للحفاظ على المعلومات (Fine-Tuning)
    
    model = PPO.load(model_path, env=env, learning_rate=custom_lr)
    
    # 5. بدء التدريب
    print(f"🚀 Training for {steps} steps...")
    model.learn(total_timesteps=steps, progress_bar=True)
    
    # 6. الحفظ
    model.save(save_path)
    print(f"✅ Stage {stage_name} Completed! Saved to: {save_path}")
    return save_path

# ==========================================
# 🏁 تنفيذ الخطة (The Roadmap)
# ==========================================

# نبدأ من الموديل الذكي v2
current_model = "models/rl_agents/final_ppo_agent_v2"

# --- المرحلة 1: رفع اللياقة (High Intensity) ---
# زحمة ثابتة وعالية نوعاً ما (0.16)
stage_1_cfg = {
    'arrival_rate_low': 0.10,
    'arrival_rate_high': 0.16
}
current_model = run_training_stage("1_HighIntensity", current_model, "models/rl_agents/agent_stage_1", stage_1_cfg, steps=150000)

# --- المرحلة 2: التباين العالي (Chaos Mode) ---
# نوسع المجال جداً ليتعلم التأقلم مع التغيرات المفاجئة
stage_2_cfg = {
    'arrival_rate_low': 0.05,  # هدوء
    'arrival_rate_high': 0.19  # ذروة مفاجئة
}
current_model = run_training_stage("2_ChaosMode", current_model, "models/rl_agents/agent_stage_2", stage_2_cfg, steps=150000)

# --- المرحلة 3: الوحش (Survival Mode) ---
# أقصى ضغط ممكن
stage_3_cfg = {
    'arrival_rate_low': 0.12,
    'arrival_rate_high': 0.20  # مستوى خطير جداً
}
run_training_stage("3_Survival", current_model, "models/rl_agents/agent_final_beast", stage_3_cfg, steps=200000)

print("\n🏆🏆 ALL STAGES COMPLETED! You have a beast agent now. 🏆🏆")