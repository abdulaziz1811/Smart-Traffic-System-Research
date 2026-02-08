"""
Detection Model
================
Build / load DETR detectors. Compatible with models/weights/final_model.
"""

import os, logging
import torch
from transformers import DetrForObjectDetection
from src.config import get_categories

log = logging.getLogger("traffic")


def build_detector(cfg, device, checkpoint=None):
    """Build from pretrained or load from checkpoint."""
    _, id2label, label2id, nc = get_categories(cfg)
    if checkpoint:
        log.info(f"Loading: {checkpoint}")
        model = DetrForObjectDetection.from_pretrained(checkpoint)
    else:
        bb = cfg["model"]["backbone"]
        log.info(f"Init from: {bb}")
        model = DetrForObjectDetection.from_pretrained(
            bb, num_labels=nc, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True)
    model.to(device)
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Params: {tot:,} total, {trn:,} trainable")
    return model


def load_best_or_final(cfg, device):
    """Load best_model → final_model → pretrained (fallback chain)."""
    wd = cfg["paths"]["weights_dir"]
    for name in ("best_model", "final_model", "detr_finetuned"):
        p = os.path.join(wd, name)
        if os.path.isdir(p) and os.listdir(p):
            return build_detector(cfg, device, checkpoint=p)
    log.warning("No trained model — loading pretrained backbone")
    return build_detector(cfg, device)
