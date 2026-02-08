#!/usr/bin/env python3
"""Detection + SORT tracking → tracked MP4 with persistent IDs."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import bootstrap
from src.detector import load_best_or_final
from src.dataset import get_processor
from src.tracker import SORTTracker
from src.inference import process_tracked_video

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--max-age", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    model = load_best_or_final(cfg, device)
    processor = get_processor(cfg)
    
    if args.max_age: cfg["tracking"]["max_age"] = args.max_age
    tracker = SORTTracker.from_config(cfg)

    if args.image_dir:
        folder, name = args.image_dir, os.path.basename(args.image_dir)
    elif args.sequence:
        name = args.sequence
        folder = os.path.join(cfg["paths"]["images_dir"], name)
    else:
        base = cfg["paths"]["images_dir"]
        seqs = [d for d in os.listdir(base) if d.startswith("MVI")]
        name = seqs[0] if seqs else ""
        folder = os.path.join(base, name)

    process_tracked_video(model, processor, tracker, folder,
                          args.output or f"outputs/results/tracking_{name}.mp4",
                          device,
                          args.threshold or cfg["inference"]["confidence_threshold"],
                          args.fps or cfg["video"]["fps"],
                          cfg["video"]["show_overlay"])

if __name__ == "__main__":
    main()
