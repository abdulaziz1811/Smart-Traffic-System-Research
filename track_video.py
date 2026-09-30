#!/usr/bin/env python3
"""
Real-Time Pipeline: Video -> DETR -> SORT -> RL Signal Control
===============================================================
Detects and tracks vehicles in a video, estimates per-lane queues from
approach zones, and lets the trained PPO agent decide the signal phase.

The agent was trained with 1 environment step = 1 second, so it is queried
once per second of video (not once per frame) and the same timing rules as
the simulator are applied: min_green, max_green and yellow/all-red time.

Usage:
    python track_video.py --video path/to/traffic.mp4
    python track_video.py --video traffic.mp4 --agent models/rl_agents/free_agent
"""

import os
import argparse
import logging
from collections import deque

import cv2
import numpy as np
from stable_baselines3 import PPO

from src.config import load_config
from src.detector import DETRDetector
from src.tracker import DeepSortTracker
from src.environment import GREEN_MAP, build_observation, infer_action_mode

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("TrafficSystem")

# ROI Zones for a 960x540 frame (adjust to your camera view)
ZONES = {
    'North': [350, 0, 600, 250],
    'South': [350, 300, 600, 540],
    'East':  [600, 150, 960, 400],
    'West':  [0, 150, 350, 400],
}

# ASSUMPTION: one zone per approach, so the left-turn share is estimated.
# Replace with one zone per lane (8 zones) for real deployments.
LEFT_TURN_SHARE = 0.2

# Class names counted as vehicles (fine-tuned UA-DETRAC or COCO fallback model)
VEHICLE_NAMES = {"car", "bus", "van", "others", "truck", "motorcycle"}


class SignalController:
    """
    Applies the trained agent with the same timing rules as TrafficSignalEnv.
    Call `step(queues)` once per second.
    """

    def __init__(self, cfg, agent):
        rc = cfg["rl"]
        self.agent = agent
        self.n_phase = rc["num_phases"]
        self.min_green = rc["min_green"]
        self.max_green = rc["max_green"]
        self.yellow_time = int(rc.get("yellow_time", 0))
        self.action_mode = infer_action_mode(agent, self.n_phase)

        self.phase = 0
        self.timer = 0
        self.clearance_left = 0
        window = max(int(rc.get("trend_window", 4)), 2)
        self.queue_history = deque([np.zeros(rc["num_approaches"], dtype=np.float32)] * window,
                                   maxlen=window)

    def _target_phase(self, action):
        a = int(np.asarray(action).reshape(-1)[0])
        if self.action_mode == "free":
            return None if a == self.phase else a % self.n_phase
        return (self.phase + 1) % self.n_phase if a == 1 else None

    def _switch_to(self, phase):
        self.phase = phase
        self.timer = 0
        self.clearance_left = self.yellow_time

    def step(self, queues):
        """Advance one second. Returns a short status string for display."""
        self.queue_history.append(np.asarray(queues, dtype=np.float32).copy())

        if self.clearance_left > 0:
            self.clearance_left -= 1
            return "YELLOW / ALL-RED"

        obs = build_observation(queues, self.phase, self.timer, self.max_green,
                                self.n_phase, GREEN_MAP, self.queue_history)
        action, _ = self.agent.predict(obs, deterministic=True)

        target = self._target_phase(action) if self.timer >= self.min_green else None
        if target is not None:
            self._switch_to(target)
            return f"CHANGE -> {target}"

        self.timer += 1
        if self.timer >= self.max_green:
            self._switch_to((self.phase + 1) % self.n_phase)
            return "MAX GREEN -> NEXT"
        return "KEEP"


class SmartTrafficSystem:
    def __init__(self, config_path, agent_path):
        self.cfg = load_config(config_path)

        # 1. Initialize Detection and Tracking
        log.info("Initializing Detector...")
        self.detector = DETRDetector(self.cfg)
        id2label = self.detector.model.config.id2label
        self.vehicle_ids = {int(i) for i, n in id2label.items() if str(n).lower() in VEHICLE_NAMES}

        log.info("Initializing Tracker...")
        self.tracker = DeepSortTracker(self.cfg)

        # 2. Load RL Agent
        if not os.path.exists(agent_path + ".zip"):
            raise FileNotFoundError(f"RL agent not found: {agent_path}.zip (run train_rl_agent.py)")
        log.info(f"Loading AI Agent from: {agent_path}")
        agent = PPO.load(agent_path, device="cpu", custom_objects={
            "learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2})
        self.controller = SignalController(self.cfg, agent)
        log.info(f"Agent action mode: {self.controller.action_mode}")

        # Initialize State Variables
        self.queues = np.zeros(self.cfg["rl"]["num_approaches"], dtype=np.float32)

    def update_counts(self, tracks):
        """Update vehicle counts per zone based on tracking data."""
        counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}

        for t in tracks:
            # t = [x1, y1, x2, y2, id]
            x1, y1, x2, y2 = t[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            for zone_name, box in ZONES.items():
                if box[0] < cx < box[2] and box[1] < cy < box[3]:
                    counts[zone_name] += 1

        split = LEFT_TURN_SHARE
        self.queues[0] = counts['North'] * (1 - split) # N Straight
        self.queues[1] = counts['North'] * split       # N Left
        self.queues[2] = counts['South'] * (1 - split) # S Straight
        self.queues[3] = counts['South'] * split       # S Left
        self.queues[4] = counts['East'] * (1 - split)  # E Straight
        self.queues[5] = counts['East'] * split        # E Left
        self.queues[6] = counts['West'] * (1 - split)  # W Straight
        self.queues[7] = counts['West'] * split        # W Left

    def run(self, video_path, loop=True):
        if not os.path.exists(video_path):
            log.error(f"Video file not found at: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("Failed to open video file.")
            return

        # One agent decision per second of video (1 env step = 1 second)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames_per_step = max(1, int(round(fps)))
        log.info(f"Starting processing: {video_path} ({fps:.1f} FPS, "
                 f"decision every {frames_per_step} frames)")

        conf = self.cfg["inference"]["confidence_threshold"]
        frame_idx = 0
        status_text = "KEEP"

        while True:
            ret, frame = cap.read()
            if not ret:
                if not loop:
                    break
                log.info("End of video, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize for consistent processing
            frame = cv2.resize(frame, (960, 540))

            # 1. Detect (vehicles only) and Track every frame
            detections = self.detector.detect(frame, conf_thresh=conf)
            if len(detections):
                detections = detections[np.isin(detections[:, 5].astype(int), list(self.vehicle_ids))]
            tracks = self.tracker.update(detections)

            # 2. Update System State
            self.update_counts(tracks)

            # 3. Agent decision once per simulated second
            if frame_idx % frames_per_step == 0:
                status_text = self.controller.step(self.queues)
            frame_idx += 1

            # 4. Visualization
            for t in tracks:
                x1, y1, x2, y2 = map(int, t[:4])
                tid = int(t[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, str(tid), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            ctl = self.controller
            in_clearance = ctl.clearance_left > 0
            status_color = (0, 255, 255) if in_clearance else (
                (0, 255, 0) if status_text == "KEEP" else (0, 165, 255))

            cv2.rectangle(frame, (0, 0), (380, 185), (0, 0, 0), -1)
            cv2.putText(frame, f"Phase: {ctl.phase}  Green: {ctl.timer}s", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Action: {status_text}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            q_ns = int(round(self.queues[:4].sum()))
            q_ew = int(round(self.queues[4:].sum()))
            cv2.putText(frame, f"N/S Vehicles: {q_ns}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"E/W Vehicles: {q_ew}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Open loop: the video does not react to the signal", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow("Smart Traffic Control AI", frame)

            if cv2.waitKey(1) == 27: # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Video -> detection -> tracking -> RL control")
    ap.add_argument("--video", required=True, help="path to a traffic video file")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--agent", default="models/rl_agents/cyclic_agent",
                    help="trained PPO agent path (without .zip)")
    ap.add_argument("--no-loop", action="store_true", help="stop at the end of the video")
    args = ap.parse_args()

    print("System Starting...")
    system = SmartTrafficSystem(args.config, args.agent)
    system.run(args.video, loop=not args.no_loop)


if __name__ == "__main__":
    main()
