#!/usr/bin/env python3
"""Train vehicle detector with validation, mAP, early stopping."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import bootstrap
from src.dataset import get_processor, get_dataloaders
from src.detector import build_detector
from src.trainer import Trainer, evaluate_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--resume", type=str, help="Checkpoint path to resume from")
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    tc = cfg["training"]
    if args.epochs: tc["epochs"] = args.epochs
    if args.batch_size: tc["batch_size"] = args.batch_size
    if args.lr: tc["learning_rate"] = args.lr

    processor = get_processor(cfg)
    train_ld, val_ld, test_ld = get_dataloaders(cfg, processor)
    model = build_detector(cfg, device, checkpoint=args.resume)

    trainer = Trainer(model, processor, cfg, device)
    trainer.fit(train_ld, val_ld)

    if test_ld:
        log.info("── Test Evaluation ──")
        best = os.path.join(cfg["paths"]["weights_dir"], "best_model")
        if os.path.isdir(best):
            from transformers import DetrForObjectDetection
            m = DetrForObjectDetection.from_pretrained(best).to(device)
            r = evaluate_map(m, processor, test_ld, device)
            log.info(f"Test mAP@50={r['mAP@50']:.4f}  mAP@75={r['mAP@75']:.4f}")

if __name__ == "__main__":
    main()
