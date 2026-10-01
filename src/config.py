"""
Configuration · Device · Seed · Logging
=========================================
Bootstrap everything in one import:

    from src.config import bootstrap, load_config
    cfg, log, device = bootstrap()
"""

import os, sys, random, time, logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml, numpy as np, torch

# ━━━━━ Config ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config(path: str = "configs/config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict):
    for k in ("weights_dir", "results_dir", "log_dir", "annotations_dir"):
        d = cfg["paths"].get(k, "")
        if d: os.makedirs(d, exist_ok=True)


# ━━━━━ Categories ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COCO JSON files keep the dataset category ids (1-indexed).
# The DETR model needs contiguous 0-indexed labels, because HF DETR uses
# index `num_labels` as the "no object" class: with 1-indexed labels the
# 4th class ("Others" = 4) collided with "no object" and could never be
# predicted.

def get_categories(cfg: dict) -> Tuple[Dict[str,int], Dict[int,str], Dict[str,int], int]:
    """
    Returns: (cat_map, id2label, label2id, num_classes)
    cat_map  = {"car":1, "bus":2, "van":3, "others":4}   (COCO category ids)
    id2label = {0:"Car", 1:"Bus", 2:"Van", 3:"Others"}   (model labels)
    label2id = {"Car":0, "Bus":1, "Van":2, "Others":3}
    """
    cat_map = cfg["dataset"]["categories"]
    names = [k for k, _ in sorted(cat_map.items(), key=lambda kv: kv[1])]
    id2label = {i: n.capitalize() for i, n in enumerate(names)}
    label2id = {v: k for k, v in id2label.items()}
    return cat_map, id2label, label2id, len(id2label)


def category_to_label_map(cfg: dict, label2id: Optional[Dict[str, int]] = None) -> Dict[int, int]:
    """
    Map COCO category ids -> model label ids, matched by class name.

    Pass `model.config.label2id` to evaluate an already-trained model in its
    own label space (e.g. an old model trained with 1-indexed labels).
    """
    cat_map = cfg["dataset"]["categories"]
    if label2id is None:
        _, _, label2id, _ = get_categories(cfg)
    by_name = {str(k).lower(): int(v) for k, v in label2id.items()}
    return {int(cid): by_name[name.lower()] for name, cid in cat_map.items() if name.lower() in by_name}


# ━━━━━ Device ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d, n = torch.device("cuda"), torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d, n = torch.device("mps"), "Apple MPS"
    else:
        d, n = torch.device("cpu"), "CPU"
    logging.getLogger("traffic").info(f"Device: {n}")
    return d


# ━━━━━ Seed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.getLogger("traffic").info(f"Seed: {seed}")


# ━━━━━ Logger ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_LOG_INIT = False
def get_logger(log_dir: str = "outputs/logs") -> logging.Logger:
    global _LOG_INIT
    lg = logging.getLogger("traffic")
    if _LOG_INIT: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); lg.addHandler(ch)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"), encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    _LOG_INIT = True
    return lg


# ━━━━━ Bootstrap (one-call setup) ━━━━━━━━━━━━━━━━━━━━━━━

def bootstrap(config_path: str = "configs/config.yaml"):
    """Load config → create dirs → logger → seed → device."""
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    log = get_logger(cfg["paths"]["log_dir"])
    set_seed(cfg["training"]["seed"])
    device = get_device()
    return cfg, log, device
