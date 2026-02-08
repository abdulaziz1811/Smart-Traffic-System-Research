"""
Visualization
==============
PIL & OpenCV drawing: boxes, labels, tracks, overlays.
"""

from typing import Dict, List, Optional
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS_RGB = {"Car":(220,50,50), "Bus":(50,50,220), "Van":(50,180,50), "Others":(255,165,0)}
COLORS_BGR = {k:(b,g,r) for k,(r,g,b) in COLORS_RGB.items()}

def _font(sz=16):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "arial.ttf"):
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()


def draw_boxes_pil(img, boxes, labels, scores):
    """Draw detection boxes on PIL image."""
    draw = ImageDraw.Draw(img); font = _font(16)
    for bx, lb, sc in zip(boxes, labels, scores):
        c = COLORS_RGB.get(lb, (255,255,255))
        draw.rectangle(bx, outline=c, width=3)
        txt = f"{lb} {sc:.0%}"
        tb = draw.textbbox((0,0), txt, font=font)
        tw, th = tb[2]-tb[0]+8, tb[3]-tb[1]+6
        o = (bx[0], bx[1]-th-2)
        draw.rectangle([o, (o[0]+tw, o[1]+th)], fill=c)
        draw.text((o[0]+4, o[1]+2), txt, fill="white", font=font)
    return img


def draw_boxes_cv2(frame, boxes, labels, scores, track_ids=None):
    """Draw detection boxes on OpenCV frame."""
    for i,(bx,lb,sc) in enumerate(zip(boxes, labels, scores)):
        c = COLORS_BGR.get(lb, (255,255,255))
        x1,y1,x2,y2 = map(int, bx)
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
        txt = f"{lb} {sc:.0%}"
        if track_ids: txt = f"ID:{track_ids[i]} {txt}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+4,y1), c, -1)
        cv2.putText(frame, txt, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame


def draw_tracks(frame, tracks, color_map):
    """Draw tracked objects with persistent colors."""
    rng = np.random.RandomState(42)
    for trk in tracks:
        x1,y1,x2,y2,tid = map(int, trk)
        if tid not in color_map:
            color_map[tid] = tuple(rng.randint(60,255,3).tolist())
        c = color_map[tid]
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
        txt = f"ID:{tid}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+4,y1), c, -1)
        cv2.putText(frame, txt, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return frame


def draw_overlay(frame, stats: Dict[str, str]):
    """Semi-transparent stats panel."""
    ov = frame.copy()
    h = 10 + 24*len(stats)
    cv2.rectangle(ov, (8,8), (270,h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    y = 28
    for k,v in stats.items():
        cv2.putText(frame, f"{k}: {v}", (16,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y += 24
    return frame
