"""
Detection Model Wrapper (DETR)
==============================
Two APIs:
  1. DETRDetector class  → for track_video.py / real-time (accepts BGR frames)
  2. build_detector()    → for train.py (returns raw HF model)
     load_best_or_final()→ for predict.py / inference.py (returns raw HF model)
"""

import os
import logging
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

log = logging.getLogger("traffic")


# ══════════════════════════════════════════════════════════
#  CLASS API — for track_video.py, demo real-time pipeline
# ══════════════════════════════════════════════════════════

class DETRDetector:
    """Self-contained detector: loads model+processor, accepts BGR frames."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        log.info(f"Detector device: {self.device}")

        self.model_path = self._find_best_model_path()

        try:
            log.info(f"Loading model from: {self.model_path}")
            self.processor = DetrImageProcessor.from_pretrained(self.model_path)
            self.model = DetrForObjectDetection.from_pretrained(self.model_path)
        except Exception as e:
            log.warning(f"Failed to load from {self.model_path}: {e}")
            log.warning("Falling back to pretrained backbone.")
            bb = cfg["model"]["backbone"]
            self.processor = DetrImageProcessor.from_pretrained(bb)
            self.model = DetrForObjectDetection.from_pretrained(bb)

        self.model.to(self.device)
        self.model.eval()
        self.conf_threshold = cfg["inference"]["confidence_threshold"]

    def _find_best_model_path(self):
        wd = self.cfg["paths"]["weights_dir"]
        for name in ("best_model", "final_model", "detr_finetuned"):
            p = os.path.join(wd, name)
            if os.path.isdir(p) and os.listdir(p):
                return p
        return self.cfg["model"]["backbone"]

    def detect(self, image):
        """
        Detect on a BGR frame (OpenCV format).
        Returns np.ndarray (N, 6): [x1, y1, x2, y2, score, class_id]
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.conf_threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [round(i, 2) for i in box.tolist()]
            detections.append([x1, y1, x2, y2, score.item(), label.item()])

        return np.array(detections) if detections else np.empty((0, 6))


# ══════════════════════════════════════════════════════════
#  FUNCTIONAL API — for train.py, predict.py, inference.py
# ══════════════════════════════════════════════════════════

def build_detector(cfg, device, checkpoint=None):
    """Build raw HF model for training pipeline."""
    from src.config import get_categories
    _, id2label, label2id, nc = get_categories(cfg)

    if checkpoint:
        log.info(f"Loading checkpoint: {checkpoint}")
        model = DetrForObjectDetection.from_pretrained(checkpoint)
    else:
        bb = cfg["model"]["backbone"]
        log.info(f"Init from backbone: {bb}")
        model = DetrForObjectDetection.from_pretrained(
            bb, num_labels=nc, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    model.to(device)
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Params: {tot:,} total, {trn:,} trainable")
    return model


def load_best_or_final(cfg, device):
    """Load best trained model (fallback chain). Returns raw HF model."""
    wd = cfg["paths"]["weights_dir"]
    for name in ("best_model", "final_model", "detr_finetuned"):
        p = os.path.join(wd, name)
        if os.path.isdir(p) and os.listdir(p):
            return build_detector(cfg, device, checkpoint=p)
    log.warning("No trained model found — loading pretrained backbone")
    return build_detector(cfg, device)