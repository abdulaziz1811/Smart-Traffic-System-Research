"""
Traffic Signal RL Environment (Enhanced V2)
===========================================
Improvements:
1. Dynamic Arrival Rates: Random traffic density per episode (Robustness).
2. Quadratic Reward: Penalizes long queues heavily (Fairness).
"""

import logging
from typing import Dict, Optional
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

log = logging.getLogger("traffic")


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: dict):
        super().__init__()
        rc = cfg["rl"]
        self.n_app = rc["num_approaches"]
        self.n_phase = rc["num_phases"]
        self.max_steps = rc["max_steps"]
        self.min_green = rc["min_green"]
        self.max_green = rc["max_green"]
        
        # إعدادات الزحمة الديناميكية
        self.arr_low = rc.get("arrival_rate_low", 0.02)
        self.arr_high = rc.get("arrival_rate_high", 0.15)
        self.service = rc["service_rate"]
        self.switch_pen = rc.get("switch_penalty", -2.0)

        # خريطة الإشارات (أي مسارات تأخذ الأخضر في كل مرحلة)
        # Phase 0: NS Straight (North-South)
        # Phase 1: NS Left
        # Phase 2: EW Straight (East-West)
        # Phase 3: EW Left
        self.green_map = {0: [0, 2], 1: [0, 2], 2: [1, 3], 3: [1, 3]}

        # تعريف مساحة الملاحظات (State Space)
        # [Queues(4) + Phase_OneHot(4) + Timer(1) + Waits(4)] = 13 inputs
        obs_dim = self.n_app + self.n_phase + 1 + self.n_app
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(obs_dim,), dtype=np.float32
        )
        
        # تعريف مساحة الأكشن (Actions): 0=Keep, 1=Next, 2=Switch Logic
        self.action_space = spaces.Discrete(3)

        # الحالة الداخلية
        self.queues = np.zeros(self.n_app, dtype=np.float32)
        self.waits = np.zeros(self.n_app, dtype=np.float32)
        self.phase = 0
        self.timer = 0
        self.step_n = 0
        self.arrivals = np.zeros(self.n_app, dtype=np.float32)
        self.switches = 0
        self.total_served = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # 🌟 التحسين الأول: زحمة عشوائية في كل بداية (Dynamic Traffic)
        # هذا يجعل الايجنت جاهزاً لأي سيناريو (صباح، ليل، ذروة)
        self.arrivals = self.np_random.uniform(
            self.arr_low, self.arr_high, size=self.n_app
        ).astype(np.float32)
        
        # إعادة تصفير العدادات
        self.queues.fill(0)
        self.waits.fill(0)
        self.phase = 0
        self.timer = 0
        self.step_n = 0
        self.switches = 0
        self.total_served = 0
        
        return self._obs(), {}

    def step(self, action):
        self.step_n += 1
        old_phase = self.phase
        
        # تنفيذ القرار (Logic Control)
        # Action 0: Keep (لا تفعل شيئاً)
        # Action 1: Next Phase (انتقل للمرحلة التالية بالترتيب)
        # Action 2: Smart Switch (انتقل للمسار الأكثر ازدحاماً فوراً)
        
        if action == 1 and self.timer >= self.min_green:
            self.phase = (self.phase + 1) % self.n_phase
        elif action == 2 and self.timer >= self.min_green:
            # منطق ذكي: البحث عن المرحلة التي تخدم أكبر عدد من المنتظرين
            demands = []
            for p in range(self.n_phase):
                # مجموع السيارات في المسارات التي ستفتح لها الإشارة p
                lane_sum = sum(self.queues[a] for a in self.green_map[p])
                demands.append(lane_sum)
            self.phase = int(np.argmax(demands))
            
        # تحديث عداد الإشارة
        if self.phase != old_phase:
            self.timer = 0
            self.switches += 1
        else:
            self.timer += 1
            
        # القفل الإجباري (Max Green Violation)
        if self.timer >= self.max_green:
            self.phase = (self.phase + 1) % self.n_phase
            self.timer = 0
            self.switches += 1

        # محاكاة البيئة (Simulation Step)
        # 1. وصول سيارات جديدة
        new_cars = self.np_random.poisson(self.arrivals)
        self.queues += new_cars
        
        # 2. تصريف السيارات (Service)
        served = 0.0
        active_lanes = self.green_map[self.phase]
        for lane in active_lanes:
            # يمكن تمرير عدد معين فقط (service rate) أو الموجود في الطابور أيهما أقل
            # نضيف عشوائية بسيطة لسرعة التصريف لمحاكاة الواقع
            flow_rate = self.service * self.np_random.uniform(0.8, 1.2)
            s = min(self.queues[lane], flow_rate)
            self.queues[lane] -= s
            served += s
            
            # تقليل وقت الانتظار للمسارات النشطة (تقريبياً)
            if self.queues[lane] > 0:
                 self.waits[lane] *= 0.9 # تخفيض تدريجي للانتظار
            else:
                 self.waits[lane] = 0

        self.queues = np.maximum(self.queues, 0)
        
        # 3. تحديث أوقات الانتظار للباقين
        # كل سيارة باقية تزيد "ضغط الانتظار"
        self.waits += self.queues 
        self.total_served += served

        # ── 🔥 التحسين الثاني: دالة المكافأة التربيعية (Quadratic Reward) 🔥 ──
        # العقاب بأس 2 يجعل النظام يكره الطوابير الطويلة جداً
        # مثال: طابورين (10, 10) => العقوبة 100+100=200
        # بينما (1, 19) => العقوبة 1+361=362 (عقوبة أكبر لنفس عدد السيارات!)
        # هذا يجبر الايجنت على "موازنة" التقاطع
        
        queue_cost = -np.sum(self.queues ** 2) / 100.0  # نقسم لتقليص الرقم
        wait_cost = -np.sum(self.waits) / 500.0         # عقوبة التأخير
        switch_cost = self.switch_pen if self.phase != old_phase else 0.0
        service_reward = served * 2.0                   # مكافأة لكل سيارة تمر
        
        reward = queue_cost + wait_cost + switch_cost + service_reward

        terminated = False
        truncated = self.step_n >= self.max_steps
        
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _obs(self):
        # تجميع الحالة للموديل
        phase_oh = np.zeros(self.n_phase)
        phase_oh[self.phase] = 1
        
        obs = np.concatenate([
            self.queues,        # حالة الطوابير
            phase_oh,           # حالة الإشارة الحالية
            [self.timer],       # كم ثانية مضت
            self.waits          # أوقات الانتظار
        ])
        return obs.astype(np.float32)

    def _info(self):
        return {
            "switches": self.switches,
            "served": self.total_served,
            "avg_queue": np.mean(self.queues)
        }