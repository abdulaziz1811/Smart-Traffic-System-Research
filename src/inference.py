"""
Inference Pipelines
====================
- detect_image()          → single image detection
- process_video()         → detection video
- process_tracked_video() → detection + tracking video
"""

import os, time, logging
from dataclasses import dataclass, field
from typing import Dict, List, Any

import cv2, numpy as np, torch
from PIL import Image
from tqdm import tqdm

from src.viz import draw_boxes_pil, draw_boxes_cv2, draw_tracks, draw_overlay

log = logging.getLogger("traffic")


# ── Single image ─────────────────────────────────────────

@dataclass
class DetResult:
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

@torch.no_grad()
def detect_image(model, processor, image, device, threshold=0.5):
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    ts = torch.tensor([image.size[::-1]]).to(device)
    raw = processor.post_process_object_detection(outputs, target_sizes=ts, threshold=threshold)[0]
    r = DetResult()
    for sc, lb, bx in zip(raw["scores"], raw["labels"], raw["boxes"]):
        r.boxes.append(bx.cpu().tolist())
        r.scores.append(sc.cpu().item())
        r.labels.append(model.config.id2label.get(lb.item(), f"cls_{lb.item()}"))
    return r


# ── Video detection ──────────────────────────────────────

def process_video(model, processor, folder, output, device, threshold=0.5, fps=25, overlay=True):
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png")))
    if not frames: log.error(f"No images in {folder}"); return {}

    sample = cv2.imread(os.path.join(folder, frames[0]))
    assert sample is not None
    h,w = sample.shape[:2]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output, getattr(cv2, "VideoWriter_fourcc")(*"mp4v"), fps, (w,h))

    total_det = 0
    times: list[float] = []
    for i, fn in enumerate(tqdm(frames, desc="Detecting")):
        t0 = time.perf_counter()
        pil = Image.open(os.path.join(folder, fn)).convert("RGB")
        res = detect_image(model, processor, pil, device, threshold)
        total_det += len(res.scores)
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img = draw_boxes_cv2(img, res.boxes, res.labels, res.scores)
        dt = time.perf_counter()-t0; times.append(dt)
        if overlay:
            recent = times[-30:]
            img = draw_overlay(img, {"Frame":f"{i+1}/{len(frames)}",
                                     "Detections":str(len(res.scores)),
                                     "FPS":f"{1/(sum(recent)/len(recent)):.1f}",
                                     "Total":str(total_det)})
        writer.write(img)
    writer.release()
    avg = len(frames)/sum(times) if times else 0
    log.info(f"Video: {output} ({avg:.1f} FPS, {total_det} detections)")
    return {"frames":len(frames), "detections":total_det, "fps":avg}


# ── Video tracking ───────────────────────────────────────

def process_tracked_video(model, processor, tracker, folder, output, device,
                          threshold=0.5, fps=25, overlay=True):
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png")))
    if not frames: log.error(f"No images in {folder}"); return {}

    sample = cv2.imread(os.path.join(folder, frames[0]))
    assert sample is not None
    h,w = sample.shape[:2]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output, getattr(cv2, "VideoWriter_fourcc")(*"mp4v"), fps, (w,h))

    uids: set[int] = set()
    cmap: dict[str, Any] = {}
    times: list[float] = []
    for i, fn in enumerate(tqdm(frames, desc="Tracking")):
        t0 = time.perf_counter()
        pil = Image.open(os.path.join(folder, fn)).convert("RGB")
        res = detect_image(model, processor, pil, device, threshold)
        dets = np.array([b+[s] for b,s in zip(res.boxes, res.scores)]) if res.boxes else np.empty((0,5))
        tracks = tracker.update(dets)
        for t in tracks: uids.add(int(t[4]))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img = draw_tracks(img, tracks, cmap)
        dt = time.perf_counter()-t0; times.append(dt)
        if overlay:
            recent = times[-30:]
            img = draw_overlay(img, {"Frame":f"{i+1}/{len(frames)}",
                                     "Active":str(len(tracks)),
                                     "Unique":str(len(uids)),
                                     "FPS":f"{1/(sum(recent)/len(recent)):.1f}"})
        writer.write(img)
    writer.release()
    avg = len(frames)/sum(times) if times else 0
    log.info(f"Tracked: {output} ({len(uids)} vehicles, {avg:.1f} FPS)")
    return {"frames":len(frames), "unique":len(uids), "fps":avg}
