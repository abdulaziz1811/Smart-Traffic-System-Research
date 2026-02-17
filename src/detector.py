"""
Detection Model Wrapper (DETR)
==============================
Handles model loading and building for both training and inference.
"""

import os
import torch
import cv2
import logging
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# Initialize logger
log = logging.getLogger("traffic")

# ══════════════════════════════════════════════════════════
#  CLASS API — for Real-Time Inference (Video / Demo)
# ══════════════════════════════════════════════════════════

class DETRDetector:
    """
    Self-contained detector class for inference.
    Loads model + processor and accepts BGR frames (OpenCV format).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Auto-detect device (CUDA -> MPS -> CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        log.info(f"Detector initialized on device: {self.device}")

        # Find best available model weights
        self.model_path = self._find_best_model_path()
        log.info(f"Loading model weights from: {self.model_path}")

        try:
            # Load processor and model
            self.processor = DetrImageProcessor.from_pretrained(
                self.model_path,
                revision="no_timm" 
            )
            self.model = DetrForObjectDetection.from_pretrained(
                self.model_path,
                revision="no_timm"
            ).to(self.device)
            self.model.eval()
            
        except Exception as e:
            log.error(f"Failed to load model from {self.model_path}: {e}")
            log.warning("Falling back to default 'facebook/detr-resnet-50'...")
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(self.device)

    def _find_best_model_path(self):
        """Helper to find the most relevant model checkpoint."""
        weights_dir = self.cfg["paths"]["weights_dir"]
        candidates = [
            os.path.join(weights_dir, "best_model"),      # Top priority: Best training result
            os.path.join(weights_dir, "final_model"),     # Second: Final training result
            "facebook/detr-resnet-50"                     # Fallback: Pretrained base model
        ]
        for path in candidates:
            if os.path.exists(path) or path.startswith("facebook/"):
                return path
        return "facebook/detr-resnet-50"

    def detect(self, frame_bgr, conf_thresh=0.5):
        """
        Run detection on a single BGR frame.
        Returns: detections list [[x1, y1, x2, y2, score, class_id], ...]
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process (convert logits to boxes)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_thresh
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # Append [x1, y1, x2, y2, score, class_id]
            detections.append([*box, score.item(), label.item()])

        return np.array(detections) if detections else np.empty((0, 6))


# ══════════════════════════════════════════════════════════
#  FUNCTIONAL API — for Training Pipeline (train.py)
# ══════════════════════════════════════════════════════════

def build_detector(cfg, device, checkpoint=None):
    """
    Builds the DETR model for training.
    Correctly handles HuggingFace model paths and custom checkpoints.
    """
    # 1. Get correct model path (fix for OSError)
    # We prioritize 'pretrained_model' key, default to 'facebook/detr-resnet-50'
    model_name = cfg["model"].get("pretrained_model", "facebook/detr-resnet-50")
    
    # 2. Extract Dataset Info
    id2label = {int(k): v for k, v in cfg["dataset"]["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    num_classes = len(id2label)

    log.info(f"Building detector from base: {model_name}")

    # 3. Load Model with correct config
    # ignore_mismatched_sizes=True is crucial when fine-tuning on new classes
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        revision="no_timm" 
    )

    # 4. Load Checkpoint (Resume Training) if provided
    if checkpoint and os.path.exists(checkpoint):
        log.info(f"Resuming training from checkpoint: {checkpoint}")
        cpt = torch.load(checkpoint, map_location=device)
        
        # Handle different checkpoint formats (full dict vs state_dict only)
        state_dict = cpt['model_state_dict'] if 'model_state_dict' in cpt else cpt
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            log.warning(f"Strict loading failed, trying non-strict: {e}")
            model.load_state_dict(state_dict, strict=False)

    model.to(device)
    
    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model Stats: {total_params:,} total parameters | {trainable_params:,} trainable")
    
    return model

def load_best_or_final(cfg, device):
    """
    Helper to load the best trained model for inference/testing.
    """
    weights_dir = cfg["paths"]["weights_dir"]
    
    # Priority list
    candidates = ["best_model", "final_model", "checkpoint_latest_backup.pth"]
    
    model_path = "facebook/detr-resnet-50" # Default fallback
    
    for name in candidates:
        path = os.path.join(weights_dir, name)
        if os.path.exists(path):
            model_path = path
            break
            
    log.info(f"Loading inference model from: {model_path}")
    
    model = DetrForObjectDetection.from_pretrained(
        model_path,
        revision="no_timm"
    ).to(device)
    model.eval()
    
    return model