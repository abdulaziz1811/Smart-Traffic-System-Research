#!/usr/bin/env python3
"""
Train vehicle detector with timestamped backups.
Features:
- Saves periodic backup every 30 mins (backup_periodic_YYYYMMDD_HHMM.pth).
- Saves emergency backup on Ctrl+C (backup_interrupted_YYYYMMDD_HHMM.pth).
- Keeps disk clean by retaining only recent periodic backups.
"""

import argparse
import sys
import os
import time
import glob
from datetime import datetime
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import bootstrap
from src.dataset import get_processor, get_dataloaders
from src.detector import build_detector
from src.trainer import Trainer, evaluate_map

class SafeTrainer(Trainer):
    def __init__(self, model, processor, cfg, device, backup_interval_min=30):
        super().__init__(model, processor, cfg, device)
        self.backup_interval = backup_interval_min * 60  # Convert to seconds
        self.last_backup_time = time.time()
        self.weights_dir = cfg["paths"]["weights_dir"]

    def _get_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _cleanup_old_backups(self, keep_last=5):
        """Keeps only the last N periodic backups to save space."""
        # We only clean periodic backups, NEVER interrupted ones.
        files = sorted(glob.glob(os.path.join(self.weights_dir, "backup_periodic_*.pth")))
        if len(files) > keep_last:
            for f in files[:-keep_last]:
                try:
                    os.remove(f)
                    print(f"   [Cleanup] Removed old backup: {os.path.basename(f)}")
                except OSError:
                    pass

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 1. Standard Training Step
            pixel_values = batch["pixel_values"].to(self.device)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss = loss / self.grad_accum
            loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optim.step()
                self.optim.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.grad_accum

            # 2. 🛡️ Time-Based Backup (Unique Names)
            current_time = time.time()
            if current_time - self.last_backup_time > self.backup_interval:
                timestamp = self._get_timestamp()
                # Name includes Epoch to know progress easily
                filename = f"backup_periodic_{timestamp}_epoch{epoch_idx}.pth"
                backup_path = os.path.join(self.weights_dir, filename)
                
                print(f"\n[Auto-Save] ⏰ 30 minutes passed. Saving: {filename}")
                
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, backup_path)
                
                self._cleanup_old_backups(keep_last=5) # Keep only last 5 periodic backups
                self.last_backup_time = current_time
                self.model.train() # Ensure we stay in train mode

        return total_loss / len(dataloader)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    
    # Overrides
    tc = cfg["training"]
    if args.epochs: tc["epochs"] = args.epochs
    if args.batch_size: tc["batch_size"] = args.batch_size
    if args.lr: tc["learning_rate"] = args.lr

    log.info(f"Initializing SafeTraining on device: {device}")

    processor = get_processor(cfg)
    train_ld, val_ld, test_ld = get_dataloaders(cfg, processor)
    
    # Build Model
    model = build_detector(cfg, device, checkpoint=args.resume)

    # Use the new SafeTrainer
    trainer = SafeTrainer(model, processor, cfg, device, backup_interval_min=30)
    
    # Start Training
    try:
        trainer.fit(train_ld, val_ld)
    except KeyboardInterrupt:
        log.info("\n🛑 Training interrupted by user!")
        
        # Save with UNIQUE timestamp so we never overwrite previous interrupts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backup_interrupted_{timestamp}.pth"
        save_path = os.path.join(cfg["paths"]["weights_dir"], filename)
        
        log.info(f"Saving emergency backup to: {filename}")
        torch.save({
            'epoch': 'interrupted', # Marker
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optim.state_dict(),
        }, save_path)
        
        sys.exit(0)

    # Test Evaluation
    if test_ld:
        log.info("── Test Evaluation ──")
        best = os.path.join(cfg["paths"]["weights_dir"], "best_model")
        if os.path.isdir(best):
            from transformers import DetrForObjectDetection
            m = DetrForObjectDetection.from_pretrained(best).to(device) # type: ignore
            evaluate_map(m, processor, test_ld, device)

if __name__ == "__main__":
    main()