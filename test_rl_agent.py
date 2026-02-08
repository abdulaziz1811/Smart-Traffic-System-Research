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
    
    # 3. تحميل موديل الذكاء اللي دربته تو
    model_path = "models/rl_agents/final_ppo_agent"
    if not os.path.exists(model_path + ".zip"):
        log.error(f"Model not found at {model_path}! Did you run training?")
        return

    log.info(f"Loading trained agent from: {model_path}")
    model = PPO.load(model_path)

    # 4. تشغيل المحاكاة
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("\n" + "="*50)
    print("🚦 STARTING SMART TRAFFIC CONTROL SIMULATION 🚦")
    print("="*50 + "\n")

    try:
        while not done:
            # الذكاء يقرر: هل يغير الإشارة (Action)؟
            action, _ = model.predict(obs, deterministic=True)
            
            # تنفيذ القرار في البيئة
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # --- عرض حي لما يحدث (Visualization) ---
            # مسح الشاشة لتحديث الأرقام
            # os.system('cls' if os.name == 'nt' else 'clear') 
            
            # قراءة الطوابير من الحالة (أول 4 أرقام هي طوابير المسارات)
            # ملاحظة: هذا يعتمد على ترتيب obs في environment.py
            queues = obs[:4]  
            phase = np.argmax(obs[4:8]) # الإشارة الخضراء الحالية
            
            print(f"Step: {step} | Phase: {['NS Green', 'NS Left', 'EW Green', 'EW Left'][phase]}")
            print(f"🚗 Queues: N={int(queues[0])} | S={int(queues[1])} | E={int(queues[2])} | W={int(queues[3])}")
            print(f"🤖 Action: {['Keep', 'Next', 'Switch'][int(action)]} | Reward: {reward:.1f}")
            print("-" * 30)
            
            time.sleep(0.1)  # تأخير بسيط عشان تلحق تقرأ الأرقام

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")

    print(f"\n✅ Simulation Finished. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()