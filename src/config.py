"""
Configuration · Device · Seed · Logging
=========================================
Bootstrap everything in one import:

    from src.config import bootstrap, load_config
    cfg, log, device = bootstrap()
"""

import os, sys, random, time, logging
from pathlib import Path
from typing import Dict, Tuple

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


# ━━━━━ Categories (1-indexed ← متوافق مع النموذج المدرب) ━━

def get_categories(cfg: dict) -> Tuple[Dict[str,int], Dict[int,str], Dict[str,int], int]:
    """
    Returns: (cat_map, id2label, label2id, num_classes)
    cat_map  = {"car":1, "bus":2, "van":3, "others":4}
    id2label = {1:"Car", 2:"Bus", 3:"Van", 4:"Others"}
    """
    cat_map = cfg["dataset"]["categories"]
    id2label = {v: k.capitalize() for k, v in cat_map.items()}
    label2id = {v: k for k, v in id2label.items()}
    return cat_map, id2label, label2id, cfg["dataset"]["num_classes"]


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
