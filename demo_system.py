import cv2
import time
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from src.config import bootstrap
from src.detector import load_best_or_final
from src.dataset import get_processor
from src.inference import detect_image
from src.environment import TrafficSignalEnv

def draw_traffic_light(img, phase, queues):
    """رسم حالة الإشارة وأعداد الانتظار على الفيديو"""
    # خلفية سوداء للنص
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # تحديد لون الإشارة
    # Phase 0/1 = NS Green (الشمال/الجنوب أخضر)
    # Phase 2/3 = EW Green (الشرق/الغرب أخضر)
    ns_color = (0, 255, 0) if phase in [0, 1] else (0, 0, 255) # Green / Red
    ew_color = (0, 255, 0) if phase in [2, 3] else (0, 0, 255)

    # رسم النصوص
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # حالة الإشارة
    cv2.putText(img, "SMART TRAFFIC CONTROL (AI AGENT)", (20, 30), font, 0.8, (255, 255, 255), 2)
    
    # NS Status
    cv2.putText(img, f"NS Signal: {'GREEN' if phase in [0,1] else 'RED'}", (20, 70), font, 0.7, ns_color, 2)
    cv2.putText(img, f"Queue N:{int(queues[0])} S:{int(queues[1])}", (20, 100), font, 0.6, (200, 200, 200), 1)

    # EW Status
    cv2.putText(img, f"EW Signal: {'GREEN' if phase in [2,3] else 'RED'}", (350, 70), font, 0.7, ew_color, 2)
    cv2.putText(img, f"Queue E:{int(queues[2])} W:{int(queues[3])}", (350, 100), font, 0.6, (200, 200, 200), 1)

    return img

def main():
    # 1. الإعدادات
    cfg, log, device = bootstrap("configs/config.yaml")
    
    # تحميل الموديلات
    detector = load_best_or_final(cfg, device)
    processor = get_processor(cfg)
    rl_model = PPO.load("models/rl_agents/final_ppo_agent")
    
    # إعداد البيئة (فقط للحفاظ على الحالة الداخلية)
    env = TrafficSignalEnv(cfg)
    obs, _ = env.reset()

    # 2. تحديد الفيديو (استخدم فيديو موجود عندك)
    # حاول تختار فيديو فيه حركة مرور واضحة
    video_path = "data/raw/DETRAC-Images/MVI_40111.mp4" # <--- تأكد من المسار هنا
    
    # إذا لم يكن لديك فيديو MP4، سنستخدم الصور كفيديو (Sequences)
    # سنفترض هنا أنك تريد تجربة المحاكاة فقط بدون فيديو حقيقي إذا لم يتوفر
    # لكن الأفضل تجربة فيديو. هل لديك ملف فيديو MP4؟
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Video not found! Running in Simulation Mode only...")
        sim_mode = True
    else:
        sim_mode = False

    log.info("Starting System...")
    
    try:
        while True:
            # A. الرؤية (Vision)
            if not sim_mode:
                ret, frame = cap.read()
                if not ret: break
                
                # الكشف عن السيارات (كل 5 إطارات لتسريع الأداء)
                # هنا سنستخدم الكشف الحقيقي لتحديث الطوابير (محاكاة الربط)
                # للتبسيط في هذا الديمو، سنعتمد على أرقام البيئة لكن نعرضها على الفيديو
                pass
            else:
                # شاشة سوداء للمحاكاة
                frame = np.zeros((600, 800, 3), dtype=np.uint8)

            # B. التفكير (RL Brain)
            # الذكاء يقرر بناءً على الحالة الحالية
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            
            # قراءة الحالة الحالية
            queues = obs[:4]      # [N, S, E, W]
            phase = np.argmax(obs[4:8]) # Current Phase

            # C. العرض (Visualization)
            frame = draw_traffic_light(frame, phase, queues)
            
            cv2.imshow("Smart Traffic System", frame)
            
            # إبطاء العرض قليلاً لنرى ما يحدث
            key = cv2.waitKey(100) # 100ms delay
            if key == 27: # ESC to exit
                break

    except KeyboardInterrupt:
        pass
    finally:
        if not sim_mode: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()