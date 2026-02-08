#!/usr/bin/env python3
"""Run detection on images. Compatible with models/weights/final_model."""
import argparse, sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from tqdm import tqdm
from src.config import bootstrap
from src.detector import load_best_or_final
from src.dataset import get_processor
from src.inference import detect_image
from src.viz import draw_boxes_pil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/results")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--random", action="store_true")
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    model = load_best_or_final(cfg, device)
    processor = get_processor(cfg)
    thr = args.threshold or cfg["inference"]["confidence_threshold"]

    paths = []
    if args.image: paths = [args.image]
    elif args.image_dir:
        paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                 if f.lower().endswith((".jpg",".jpeg",".png"))]
    elif args.random:
        imgs = []
        for r,_,fs in os.walk(cfg["paths"]["images_dir"]):
            imgs.extend(os.path.join(r,f) for f in fs if f.lower().endswith((".jpg",".png")))
        if imgs: paths = [random.choice(imgs)]

    os.makedirs(args.output_dir, exist_ok=True)
    for p in tqdm(paths, desc="Detecting"):
        pil = Image.open(p).convert("RGB")
        res = detect_image(model, processor, pil, device, thr)
        log.info(f"{os.path.basename(p)}: {len(res.scores)} detections")
        out = draw_boxes_pil(pil.copy(), res.boxes, res.labels, res.scores)
        out.save(os.path.join(args.output_dir, f"det_{os.path.basename(p)}"))

if __name__ == "__main__":
    main()
