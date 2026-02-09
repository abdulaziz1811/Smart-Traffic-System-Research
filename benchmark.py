import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from src.environment import TrafficSignalEnv
from src.config import bootstrap

def run_benchmark():
    # 1. إعداد الكونفق والبيئة
    cfg, log, device = bootstrap("configs/config.yaml")
    
    # نستخدم نفس البيئة بالضبط للمقارنة العادلة
    env = TrafficSignalEnv(cfg)
    
    # ── سيناريو 1: النظام التقليدي (Fixed Timer) ──
    print("⏳ Running Fixed-Time Simulation...")
    obs, _ = env.reset(seed=42) # تثبيت الـ Seed للمقارنة العادلة
    fixed_rewards = []
    fixed_queues = []
    
    # محاكاة لمدة 3600 خطوة (ساعة)
    for step in range(3600):
        # منطق بسيط: غيّر الإشارة كل 30 ثانية
        if step % 30 == 0:
            action = 1 # Next Phase
        else:
            action = 0 # Keep
            
        obs, reward, done, _, info = env.step(action)
        fixed_rewards.append(reward)
        fixed_queues.append(info['avg_queue'])
        
        if done: break
        
    print(f"✅ Fixed-Time Total Reward: {sum(fixed_rewards):.2f}")

    # ── سيناريو 2: نظامك الذكي (AI Agent) ──
    print("🧠 Running AI Agent Simulation...")
    model = PPO.load("models/rl_agents/final_ppo_agent")
    
    obs, _ = env.reset(seed=42) # نفس الـ Seed بالضبط لنفس الزحمة
    ai_rewards = []
    ai_queues = []
    
    for step in range(3600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        ai_rewards.append(reward)
        ai_queues.append(info['avg_queue'])
        
        if done: break

    print(f"✅ AI Agent Total Reward: {sum(ai_rewards):.2f}")

    # ── الرسم البياني (The Proof) ──
    plt.figure(figsize=(12, 6))
    
    # رسم متوسط طول الطابور
    plt.plot(fixed_queues, label='Fixed Timer (Traditional)', color='red', alpha=0.6)
    plt.plot(ai_queues, label='Smart AI Agent (Ours)', color='green', linewidth=2)
    
    plt.title('Performance Comparison: AI vs Traditional Signal')
    plt.xlabel('Simulation Steps (Seconds)')
    plt.ylabel('Average Queue Length (Vehicles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # حفظ الصورة
    plt.savefig('benchmark_results.png')
    print("📊 Graph saved to 'benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()