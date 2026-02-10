import cv2
import yaml
import numpy as np
import os
import logging
from stable_baselines3 import PPO
from src.detector import DETRDetector
from src.tracker import DeepSortTracker

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("TrafficSystem")

class SmartTrafficSystem:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        # 1. Initialize Detection and Tracking
        log.info("Initializing Detector...")
        self.detector = DETRDetector(self.cfg)
        
        log.info("Initializing Tracker...")
        self.tracker = DeepSortTracker(self.cfg)
        
        # 2. Load RL Agent
        model_path = "models/rl_agents/final_model"
        if not os.path.exists(model_path + ".zip"):
             log.warning(f"Model not found at {model_path}, attempting fallback to 'final_ppo_agent_v2'...")
             model_path = "models/rl_agents/final_ppo_agent_v2"
        
        log.info(f"Loading AI Agent from: {model_path}")
        self.agent = PPO.load(model_path)
        
        # Initialize State Variables
        self.queues = np.zeros(8, dtype=np.float32)
        self.waits = np.zeros(8, dtype=np.float32)
        self.current_phase = 0
        self.timer = 0
        
        # ROI Zones (Adjust coordinates based on your video resolution)
        self.zones = {
            'North': [350, 0, 600, 250],   
            'South': [350, 300, 600, 540], 
            'East':  [600, 150, 960, 400], 
            'West':  [0, 150, 350, 400]    
        }

    def update_counts(self, tracks):
        """Update vehicle counts per zone based on tracking data."""
        counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        
        for t in tracks:
            # t = [x1, y1, x2, y2, id]
            x1, y1, x2, y2 = t[:4]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            for zone_name, box in self.zones.items():
                if box[0] < cx < box[2] and box[1] < cy < box[3]:
                    counts[zone_name] += 1
        
        # Split logic: 80% Straight, 20% Left Turn
        split = 0.2
        self.queues[0] = counts['North'] * (1 - split) # N Straight
        self.queues[1] = counts['North'] * split       # N Left
        self.queues[2] = counts['South'] * (1 - split) # S Straight
        self.queues[3] = counts['South'] * split       # S Left
        self.queues[4] = counts['East'] * (1 - split)  # E Straight
        self.queues[5] = counts['East'] * split        # E Left
        self.queues[6] = counts['West'] * (1 - split)  # W Straight
        self.queues[7] = counts['West'] * split        # W Left

    def run(self, video_path):
        if not os.path.exists(video_path):
            log.error(f"Video file not found at: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("Failed to open video file.")
            return

        log.info(f"Starting processing: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize for consistent processing
            frame = cv2.resize(frame, (960, 540))
            
            # 1. Detect and Track
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            
            # 2. Update System State
            self.update_counts(tracks)
            self.timer += 1
            
            # 3. Prepare RL Observation
            # State: [Queues(8), Phase(4), Timer(1), Waits(8)]
            phase_oh = np.zeros(4)
            phase_oh[self.current_phase] = 1
            obs = np.concatenate([self.queues, phase_oh, [self.timer], self.waits]).astype(np.float32)
            
            # 4. Agent Decision
            action, _ = self.agent.predict(obs, deterministic=True)
            
            # 5. Execute Action
            status_text = "KEEP"
            status_color = (0, 255, 0)
            
            if action == 1:
                self.current_phase = (self.current_phase + 1) % 4
                self.timer = 0
                status_text = "CHANGE -> NEXT"
                status_color = (0, 165, 255)
            elif action == 2:
                # Simplified smart switch logic
                self.timer = 0
                status_text = "SMART SWITCH"
                status_color = (0, 0, 255)

            # 6. Visualization
            # Draw Vehicle Bounding Boxes
            for t in tracks:
                x1, y1, x2, y2 = map(int, t[:4])
                tid = int(t[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, str(tid), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            # Draw Status Panel
            cv2.rectangle(frame, (0, 0), (350, 160), (0, 0, 0), -1)
            cv2.putText(frame, f"Phase: {self.current_phase}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Action: {status_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Draw Queue Info
            q_ns = int(self.queues[0] + self.queues[2])
            q_ew = int(self.queues[4] + self.queues[6])
            cv2.putText(frame, f"N/S Queue: {q_ns}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"E/W Queue: {q_ew}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Smart Traffic Control AI", frame)
            
            if cv2.waitKey(1) == 27: # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update this path to your video file location
    video_path = "/Users/abdulaziz/Downloads/traffic3.mp4"
    
    print("System Starting...")
    try:
        system = SmartTrafficSystem("configs/config.yaml")
        system.run(video_path)
    except Exception as e:
        print(f"Fatal Error: {e}")