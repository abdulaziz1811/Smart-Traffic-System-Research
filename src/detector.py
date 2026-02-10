"""
Detection Model Wrapper (DETR)
==============================
Wraps HuggingFace DETR for inference on images and video frames.
Automatically loads the best available model weights or falls back to backbone.
"""

import os
import logging
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

log = logging.getLogger("TrafficSystem")

class DETRDetector:
    def __init__(self, cfg):
        """
        Initialize the DETR detector.
        
        Args:
            cfg (dict): Configuration dictionary.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        log.info(f"Detector initializing on device: {self.device}")
        
        # Determine model path
        self.model_path = self._find_best_model_path()
        
        # Load Processor and Model
        try:
            log.info(f"Loading model from: {self.model_path}")
            self.processor = DetrImageProcessor.from_pretrained(self.model_path)
            self.model = DetrForObjectDetection.from_pretrained(self.model_path)
        except Exception as e:
            log.warning(f"Failed to load custom model from {self.model_path}. Error: {e}")
            log.warning("Falling back to pretrained backbone.")
            fallback = cfg["model"]["backbone"]
            self.processor = DetrImageProcessor.from_pretrained(fallback)
            self.model = DetrForObjectDetection.from_pretrained(fallback)

        self.model.to(self.device)
        self.model.eval()
        self.conf_threshold = cfg["inference"]["confidence_threshold"]

    def _find_best_model_path(self):
        """
        Logic to find the best checkpoint:
        1. 'final_model' inside weights_dir
        2. 'best_model' inside weights_dir
        3. Pretrained backbone (as defined in config)
        """
        weights_dir = self.cfg["paths"]["weights_dir"]
        backbone = self.cfg["model"]["backbone"]
        
        candidates = [
            os.path.join(weights_dir, "final_model"),
            os.path.join(weights_dir, "best_model"),
        ]
        
        for path in candidates:
            if os.path.exists(path) and len(os.listdir(path)) > 0:
                return path
        
        return backbone

    def detect(self, image):
        """
        Perform detection on a single image frame (BGR format from OpenCV).
        
        Args:
            image (np.ndarray): Input image in BGR format.
            
        Returns:
            np.ndarray: List of detections [[x1, y1, x2, y2, score, class_id], ...]
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([pil_img.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=self.conf_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score.item(), label.item()])
            
        return np.array(detections)