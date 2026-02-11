import time
import os
import numpy as np
from stable_baselines3 import PPO
from src.config import bootstrap
from src.environment import TrafficSignalEnv

def main():
    # 1. إعدادات
    cfg, log, device = bootstrap("configs/config.yaml")
    
    # 2. تحميل بيئة المحاكاة
    env = TrafficSignalEnv(cfg)
    
    # 3. تحميل موديل الذكاء
    model_path = "models/rl_agents/final_ppo_agent"
    if not os.path.exists(model_path + ".zip"):
        log.error(f"Model not found at {model_path}! Did you run training?")
        return

    log.info(f"Loading Logic-Agent from: {model_path}")
    model = PPO.load(model_path)

    # 4. تشغيل المحاكاة
    obs, _ = env.reset()
    
    # متغيرات التحكم في الحلقة
    done = False
    truncated = False  # مهم جداً: متغير تتبع انتهاء الوقت
    
    total_reward = 0
    step = 0

    # أسماء المراحل لتوضيح العرض
    phase_names = [
        "Phase 0: N/S Straight", 
        "Phase 1: N/S Left", 
        "Phase 2: E/W Straight", 
        "Phase 3: E/W Left"
    ]

    print("\n" + "="*60)
    print("🚦 STARTING SMART TRAFFIC LOGIC TEST 🚦")
    print("="*60 + "\n")

    try:
        # التعديل هنا: التوقف إذا انتهت اللعبة أو انتهى الوقت
        while not (done or truncated):
            # الذكاء يقرر
            action, _ = model.predict(obs, deterministic=True)
            
            # تنفيذ القرار
            # لاحظ استقبال المتغير truncated
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1

            # --- استخراج البيانات ---
            queues = obs[:8]
            current_phase_idx = np.argmax(obs[8:12])
            timer = obs[12]
            next_density = obs[13]

            # --- عرض لوحة القيادة ---
            # مسح الشاشة (اختياري)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"⏱️  Step: {step} / {cfg['rl']['max_steps']} | Timer: {timer:.2f}")
            print(f"🚦 Current: {phase_names[current_phase_idx]}")
            print("-" * 40)
            print(f"   North: [Str: {int(queues[0]):02d} | Left: {int(queues[1]):02d}]")
            print(f"   South: [Str: {int(queues[2]):02d} | Left: {int(queues[3]):02d}]")
            print(f"   East : [Str: {int(queues[4]):02d} | Left: {int(queues[5]):02d}]")
            print(f"   West : [Str: {int(queues[6]):02d} | Left: {int(queues[7]):02d}]")
            print("-" * 40)
            
            action_str = "🟢 EXTEND Green" if action == 0 else "🔴 CYCLE Phase"
            print(f"🧠 Logic: {action_str}")
            print(f"👀 Next Phase Density: {next_density:.2f}")
            print(f"💰 Step Reward: {reward:.2f}")
            print("=" * 60)

            time.sleep(0.1) # تسريع العرض قليلاً

        print("\n🏁 Test Finished: Max steps reached or Episode ended.")
        print(f"📊 Total Reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n🛑 Simulation Stopped.")

if __name__ == "__main__":
    main()