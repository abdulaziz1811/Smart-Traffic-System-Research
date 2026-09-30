#!/usr/bin/env python3
"""
Real-Time Pipeline: Video -> DETR -> SORT -> Queues -> RL Signal Control (+ VLM)
=================================================================================
Detects and tracks vehicles, estimates the queue of every lane (only
vehicles that are actually stopped, inside per-lane zones), and lets the
trained PPO agent decide the signal phase.

The agent was trained with 1 environment step = 1 second, so it is queried
once per second of video and the same timing rules as the simulator are
applied: min_green, max_green and yellow/all-red time.

Optional VLM (Claude vision, src/vlm.py): every `vlm.scan_interval_s`
seconds, and whenever a lane's stopped vehicles do not move for
`vlm.stall_trigger_s` seconds, the frame is analyzed in the background:
  * a confirmed emergency vehicle pre-empts the signal (green for its lane);
  * an incident is shown on screen and written to outputs/vlm_events.jsonl
    with the Arabic operations-room report.

Usage:
    python track_video.py --video path/to/traffic.mp4
    python track_video.py --video traffic.mp4 --agent models/rl_agents/free_agent --no-vlm
    python track_video.py --video traffic.mp4 --no-display --output outputs/results/annotated.mp4
"""

import os
import json
import time
import argparse
import logging
from collections import deque

import cv2
import numpy as np

from src.config import load_config
from src.environment import GREEN_MAP, build_observation
from src.agents import load_agent, agent_is_compatible
from src.queue_estimator import QueueEstimator
from src.vlm import VLMAnalyzer, VLMWorker, LANE_NAMES_EN

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("TrafficSystem")

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
        self.n_app = rc["num_approaches"]
        self.n_phase = rc["num_phases"]
        self.min_green = rc["min_green"]
        self.max_green = rc["max_green"]
        self.yellow_time = int(rc.get("yellow_time", 0))
        self.action_mode = agent.action_mode(self.n_phase)

        self.phase = 0
        self.timer = 0
        self.clearance_left = 0
        self.red_wait = np.zeros(self.n_app, dtype=np.float32)
        self.last_active_lanes = []
        window = max(int(rc.get("trend_window", 4)), 2)
        self.queue_history = deque([np.zeros(self.n_app, dtype=np.float32)] * window,
                                   maxlen=window)

        # Emergency pre-emption (set by the VLM)
        self.preempt_lane = None
        self.preempt_left = 0

    def preempt(self, lane, seconds):
        """Give green to `lane` (and hold it) for the next `seconds` seconds."""
        self.preempt_lane, self.preempt_left = int(lane), int(seconds)

    def _target_phase(self, action):
        a = int(np.asarray(action).reshape(-1)[0])
        if self.action_mode == "free":
            return None if a == self.phase else a % self.n_phase
        return (self.phase + 1) % self.n_phase if a == 1 else None

    def _preempt_target(self):
        """Phase to move towards for the emergency vehicle (None = hold current)."""
        target = next(p for p, lanes in GREEN_MAP.items() if self.preempt_lane in lanes)
        if target == self.phase:
            return None
        return target if self.action_mode == "free" else (self.phase + 1) % self.n_phase

    def _switch_to(self, phase):
        self.phase = phase
        self.timer = 0
        self.clearance_left = self.yellow_time

    def _update_waits(self, queues):
        has_cars = np.asarray(queues) >= 1.0
        self.red_wait = np.where(has_cars, self.red_wait + 1.0, 0.0).astype(np.float32)
        self.red_wait[self.last_active_lanes] = 0.0

    def step(self, queues):
        """Advance one second. Returns a short status string for display."""
        queues = np.asarray(queues, dtype=np.float32).copy()
        self._update_waits(queues)
        self.queue_history.append(queues)
        preempting = self.preempt_left > 0
        if preempting:
            self.preempt_left -= 1

        if self.clearance_left > 0:
            self.clearance_left -= 1
            self.last_active_lanes = []
            return "YELLOW / ALL-RED"

        old_phase = self.phase
        if preempting:
            target = self._preempt_target() if self.timer >= self.min_green else None
            status = "EMERGENCY HOLD" if target is None else "EMERGENCY -> NEXT"
        else:
            obs = build_observation(queues, self.phase, self.timer, self.max_green,
                                    self.n_phase, GREEN_MAP, self.queue_history,
                                    self.red_wait, self.clearance_left, self.yellow_time)
            action, _ = self.agent.predict(obs, deterministic=True)
            target = self._target_phase(action) if self.timer >= self.min_green else None
            status = "KEEP" if target is None else f"CHANGE -> {target}"

        if target is not None:
            self._switch_to(target)
        else:
            self.timer += 1
            if self.timer >= self.max_green and not preempting:
                self._switch_to((self.phase + 1) % self.n_phase)
                status = "MAX GREEN -> NEXT"

        # The deciding second is the last green second of the old phase
        self.last_active_lanes = list(GREEN_MAP[old_phase])
        return status


class SmartTrafficSystem:
    def __init__(self, config_path, agent_path, use_vlm=True, detector=None):
        self.cfg = load_config(config_path)

        # 1. Detection and Tracking
        if detector is None:
            from src.detector import DETRDetector
            log.info("Initializing Detector...")
            detector = DETRDetector(self.cfg)
        self.detector = detector
        id2label = self.detector.model.config.id2label
        self.vehicle_ids = {int(i) for i, n in id2label.items() if str(n).lower() in VEHICLE_NAMES}

        from src.tracker import DeepSortTracker
        log.info("Initializing Tracker...")
        self.tracker = DeepSortTracker(self.cfg)

        # 2. RL Agent
        log.info(f"Loading AI Agent from: {agent_path}")
        agent = load_agent(agent_path)
        if not agent_is_compatible(agent, self.cfg):
            raise ValueError(f"{agent_path} was trained on an older observation layout; retrain it.")
        self.controller = SignalController(self.cfg, agent)
        log.info(f"Agent action mode: {self.controller.action_mode}")

        # 3. VLM (optional)
        self.vlm_cfg = self.cfg.get("vlm", {})
        self.vlm = None
        self.vlm_status = "VLM: off"
        if use_vlm:
            analyzer = VLMAnalyzer(self.cfg)
            if analyzer.available:
                self.vlm = VLMWorker(analyzer, fallback_to_template=False)
                self.vlm_status = f"VLM: {analyzer.model}"
            else:
                self.vlm_status = f"VLM: off ({analyzer.disabled_reason})"
        log.info(self.vlm_status)
        self.vlm_events_path = os.path.join(self.cfg["paths"]["results_dir"], "vlm_events.jsonl")
        self.last_vlm_summary = ""

        self.queues = np.zeros(self.cfg["rl"]["num_approaches"], dtype=np.float32)
        self._stall_since = {}   # lane -> (set of stopped ids, second it started)

    # ------------------------------------------------------------------
    #  VLM helpers
    # ------------------------------------------------------------------

    def _stalled_lanes(self, estimator, second):
        """Lanes whose stopped vehicles have not changed for stall_trigger_s seconds."""
        limit = self.vlm_cfg.get("stall_trigger_s", 30)
        stalled = []
        for lane, ids in enumerate(estimator.stopped_ids_per_lane):
            if not ids:
                self._stall_since.pop(lane, None)
                continue
            prev = self._stall_since.get(lane)
            if prev is None or prev[0] != ids:
                self._stall_since[lane] = (set(ids), second)
            elif second - prev[1] >= limit:
                stalled.append(lane)
        return stalled

    def _vlm_context(self, trigger, stalled):
        ctl = self.controller
        return {
            "intersection": "camera_1",
            "trigger": trigger,
            "lane_order": LANE_NAMES_EN,
            "queues_stopped_vehicles": [int(q) for q in self.queues],
            "current_phase_green_lanes": [LANE_NAMES_EN[l] for l in GREEN_MAP[ctl.phase]],
            "signal_state": "yellow/all-red" if ctl.clearance_left > 0 else "green",
            "stalled_lanes": [LANE_NAMES_EN[l] for l in stalled],
        }

    def _handle_vlm_results(self):
        for a in self.vlm.poll():
            os.makedirs(os.path.dirname(self.vlm_events_path) or ".", exist_ok=True)
            with open(self.vlm_events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **a.to_dict()},
                                   ensure_ascii=False) + "\n")
            lane = a.emergency_lane()
            if lane is not None:
                self.controller.preempt(lane, self.vlm_cfg.get("preempt_s", 20))
                log.warning(f"[VLM] Emergency vehicle: {LANE_NAMES_EN[lane]} -> pre-empting signal")
            if a.incident_present:
                log.warning(f"[VLM] {a.summary_en}\n{a.report_ar}")
            if a.emergency_present or a.incident_present:
                self.last_vlm_summary = a.summary_en

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    def run(self, video_path, loop=True, display=True, output=None, max_frames=None):
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
        estimator = QueueEstimator(self.cfg["video"], fps)
        w, h = self.cfg["video"].get("frame_size", [960, 540])
        writer = None
        if output:
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            writer = cv2.VideoWriter(output, getattr(cv2, "VideoWriter_fourcc")(*"mp4v"), fps, (w, h))
        log.info(f"Starting processing: {video_path} ({fps:.1f} FPS, "
                 f"decision every {frames_per_step} frames)")

        conf = self.cfg["inference"]["confidence_threshold"]
        scan_every = int(self.vlm_cfg.get("scan_interval_s", 10))
        frame_idx = 0
        status_text = "KEEP"

        while max_frames is None or frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                if not loop:
                    break
                log.info("End of video, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (w, h))

            # 1. Detect (vehicles only) and track every frame
            detections = self.detector.detect(frame, conf_thresh=conf)
            if len(detections):
                detections = detections[np.isin(detections[:, 5].astype(int), list(self.vehicle_ids))]
            tracks = self.tracker.update(detections)

            # 2. Queue per lane (stopped vehicles inside each lane zone)
            self.queues, _ = estimator.update(tracks, frame_idx)

            # 3. Once per second: agent decision (+ VLM checks)
            if frame_idx % frames_per_step == 0:
                second = frame_idx // frames_per_step
                status_text = self.controller.step(self.queues)
                if self.vlm is not None:
                    stalled = self._stalled_lanes(estimator, second)
                    trigger = "stalled_lanes" if stalled else (
                        "periodic_scan" if second % scan_every == 0 else None)
                    if trigger:
                        self.vlm.submit(frame, self._vlm_context(trigger, stalled))
            if self.vlm is not None:
                self._handle_vlm_results()
            frame_idx += 1

            # 4. Visualization
            self._draw(frame, tracks, estimator, status_text)
            if writer is not None:
                writer.write(frame)
            if display:
                cv2.imshow("Smart Traffic Control AI", frame)
                if cv2.waitKey(1) == 27:  # ESC to exit
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if self.vlm is not None:
            self.vlm.close()
        if display:
            cv2.destroyAllWindows()

    def _draw(self, frame, tracks, estimator, status_text):
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        estimator.draw(frame, self.queues)

        ctl = self.controller
        if ctl.preempt_left > 0:
            status_color = (0, 0, 255)
        elif ctl.clearance_left > 0:
            status_color = (0, 255, 255)
        else:
            status_color = (0, 255, 0) if status_text == "KEEP" else (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (400, 205), (0, 0, 0), -1)
        cv2.putText(frame, f"Phase: {ctl.phase}  Green: {ctl.timer}s", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Action: {status_text}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"N/S queue: {int(self.queues[:4].sum())}   "
                           f"E/W queue: {int(self.queues[4:].sum())}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, self.vlm_status[:55], (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if self.last_vlm_summary:
            cv2.putText(frame, self.last_vlm_summary[:60], (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(frame, "Open loop: the video does not react to the signal", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def main():
    ap = argparse.ArgumentParser(description="Video -> detection -> tracking -> RL control (+ VLM)")
    ap.add_argument("--video", required=True, help="path to a traffic video file")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--agent", default="models/rl_agents_sumo/cyclic_agent",
                    help="trained PPO agent path (without .zip); default: trained in SUMO, "
                         "closest to real traffic dynamics")
    ap.add_argument("--no-vlm", action="store_true", help="disable the vision-language model")
    ap.add_argument("--no-loop", action="store_true", help="stop at the end of the video")
    ap.add_argument("--no-display", action="store_true", help="run headless")
    ap.add_argument("--output", default=None, help="save the annotated video to this .mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    print("System Starting...")
    system = SmartTrafficSystem(args.config, args.agent, use_vlm=not args.no_vlm)
    system.run(args.video, loop=not args.no_loop, display=not args.no_display,
               output=args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
