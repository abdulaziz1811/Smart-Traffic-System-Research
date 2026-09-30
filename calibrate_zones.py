#!/usr/bin/env python3
"""
Lane Zone Calibration
=====================
Draw the 8 lane polygons used by track_video.py on the first frame of your
video and print them as a `lane_zones:` block for configs/config.yaml.

Lane order: 0 N straight, 1 N left, 2 S straight, 3 S left,
            4 E straight, 5 E left, 6 W straight, 7 W left

Controls:
    left click  add a corner to the current lane
    ENTER       finish the current lane (needs >= 3 corners)
    BACKSPACE   remove the last corner
    ESC         quit without saving

Usage:
    python calibrate_zones.py --video path/to/traffic.mp4
"""

import argparse

import cv2
import numpy as np

LANES = ["N straight", "N left", "S straight", "S left",
         "E straight", "E left", "W straight", "W left"]


def main():
    ap = argparse.ArgumentParser(description="Draw per-lane zones for track_video.py")
    ap.add_argument("--video", required=True)
    ap.add_argument("--size", default="960x540", help="processing frame size used by track_video.py")
    args = ap.parse_args()
    w, h = map(int, args.size.lower().split("x"))

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")
    frame = cv2.resize(frame, (w, h))

    zones, current = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    while len(zones) < len(LANES):
        img = frame.copy()
        for i, z in enumerate(zones):
            pts = np.array(z, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 200, 255), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(img, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if current:
            cv2.polylines(img, [np.array(current, dtype=np.int32)], False, (0, 255, 0), 2)
        cv2.putText(img, f"Lane {len(zones)}: {LANES[len(zones)]}  (click corners, ENTER to finish)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("calibrate", img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            print("Cancelled.")
            return
        if key in (13, 10) and len(current) >= 3:
            zones.append(list(current))
            current.clear()
        if key == 8 and current:
            current.pop()

    cv2.destroyAllWindows()
    print("  lane_zones:")
    for i, z in enumerate(zones):
        print(f"    - {z}  # {i} {LANES[i]}")


if __name__ == "__main__":
    main()
