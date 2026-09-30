"""
Lane-Level Queue Estimation from Tracks
========================================
Turns SORT tracks into the 8 per-lane queue lengths the RL agent expects.

  * One polygon per lane (config `video.lane_zones`), tested against the
    bottom-centre of each box (the point where the vehicle touches the road,
    less affected by perspective than the box centre).
  * A queue is vehicles WAITING, not every vehicle in the zone: a tracked
    vehicle is counted only when its ground-point speed is below
    `video.stopped_speed_px_s` (queue_mode "stopped"). Use queue_mode "all"
    to count every vehicle in the zone.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class QueueEstimator:
    def __init__(self, video_cfg: dict, fps: float, n_lanes: int = 8):
        zones = video_cfg["lane_zones"]
        if len(zones) != n_lanes:
            raise ValueError(f"video.lane_zones must have {n_lanes} polygons, got {len(zones)}")
        self.zones = [np.asarray(z, dtype=np.float32).reshape(-1, 1, 2) for z in zones]
        self.fps = max(float(fps), 1e-3)
        self.stop_speed = float(video_cfg.get("stopped_speed_px_s", 20.0))
        self.count_all = video_cfg.get("queue_mode", "stopped") == "all"
        self.n_lanes = n_lanes

        # track id -> (last ground point, frame index, smoothed speed px/s)
        self._state: Dict[int, Tuple[np.ndarray, int, float]] = {}
        self.stopped_ids_per_lane: List[set] = [set() for _ in range(n_lanes)]

    def lane_of(self, point) -> Optional[int]:
        pt = (float(point[0]), float(point[1]))
        for i, poly in enumerate(self.zones):
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                return i
        return None

    def update(self, tracks, frame_idx: int):
        """
        tracks: array of [x1, y1, x2, y2, track_id] for the current frame.
        Returns (queues, vehicles): float arrays of length n_lanes with the
        number of waiting vehicles and of all vehicles per lane.
        """
        queues = np.zeros(self.n_lanes, dtype=np.float32)
        vehicles = np.zeros(self.n_lanes, dtype=np.float32)
        self.stopped_ids_per_lane = [set() for _ in range(self.n_lanes)]
        seen = set()

        for t in tracks:
            x1, y1, x2, y2, tid = t[:5]
            tid = int(tid)
            seen.add(tid)
            ground = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)

            # Speed of the ground point, smoothed over frames
            if tid in self._state:
                prev, prev_f, prev_speed = self._state[tid]
                dt = max(frame_idx - prev_f, 1) / self.fps
                inst = float(np.linalg.norm(ground - prev)) / dt
                speed = 0.6 * prev_speed + 0.4 * inst
            else:
                speed = float("inf")  # unknown until seen twice
            self._state[tid] = (ground, frame_idx, speed if np.isfinite(speed) else 0.0)

            lane = self.lane_of(ground)
            if lane is None:
                continue
            vehicles[lane] += 1
            stopped = np.isfinite(speed) and speed < self.stop_speed
            if self.count_all or stopped:
                queues[lane] += 1
            if stopped:
                self.stopped_ids_per_lane[lane].add(tid)

        # Forget tracks that have not been seen for a while
        for tid in list(self._state):
            if tid not in seen and frame_idx - self._state[tid][1] > 5 * self.fps:
                del self._state[tid]

        return queues, vehicles

    def draw(self, frame, queues=None, color=(0, 200, 255)):
        """Overlay lane polygons (and queue counts) for calibration / display."""
        for i, poly in enumerate(self.zones):
            pts = poly.astype(np.int32)
            cv2.polylines(frame, [pts], True, color, 1)
            if queues is not None:
                x, y = pts.reshape(-1, 2).mean(axis=0).astype(int)
                cv2.putText(frame, f"{i}:{int(queues[i])}", (int(x) - 12, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return frame
