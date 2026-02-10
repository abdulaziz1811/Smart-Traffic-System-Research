import yaml
import numpy as np
from stable_baselines3 import PPO
from src.environment import TrafficSignalEnv
import os

def test_model(name, path, env):
    if not os.path.exists(path + ".zip"):
        print(f"⚠️ Model {name} not found, skipping.")
        return None
        
    print(f"\n🤖 Testing: {name} ...")
    model = PPO.load(path)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    queues = []
    
    # اختبار لمدة 2000 خطوة (فترة كافية للحكم)
    for i in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        queues.append(info['avg_queue'])
        if truncated: break
            
    avg_q = np.mean(queues)
    print(f"   -> Score: Avg Queue Length = {avg_q:.2f} cars (Lower is better)")
    return avg_q

# --- الإعدادات ---
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# سنجربهم في "زحمة واقعية صعبة" (0.16) وليس المستحيلة
config['rl']['arrival_rate_high'] = 0.16
env = TrafficSignalEnv(config)

candidates = {
    "V2 (Previous Best)": "models/rl_agents/final_ppo_agent_v2",
    "Stage 1 (High Intensity)": "models/rl_agents/agent_stage_1",
    "Stage 2 (Chaos Mode)": "models/rl_agents/agent_stage_2",
    "Stage 3 (The Beast)": "models/rl_agents/agent_final_beast"
}

results = {}
print("🏆 --- THE GRAND FINALE BENCHMARK --- 🏆")

for name, path in candidates.items():
    score = test_model(name, path, env)
    if score is not None:
        results[name] = score

print("\n📊 --- FINAL STANDINGS ---")
sorted_results = sorted(results.items(), key=lambda x: x[1])

for rank, (name, score) in enumerate(sorted_results, 1):
    print(f"#{rank}: {name} -> Avg Queue: {score:.2f}")

winner = sorted_results[0][0]
print(f"\n🥇 The Winner to use in your project is: {winner}")