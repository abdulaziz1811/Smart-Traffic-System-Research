#!/usr/bin/env python3
"""
Detection Validation for Research Paper
========================================
Runs full evaluation and outputs:
  - Per-class AP table
  - Overall mAP table
  - Detection speed benchmark
  - Saves everything to JSON + prints LaTeX-ready table

Usage:
    python validate_detection.py
    python validate_detection.py --split val
"""

import argparse, os, sys, time, json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import bootstrap
from src.dataset import get_processor, get_dataloaders
from src.detector import load_best_or_final


def _sync(device):
    """Wait for queued GPU work so the timer measures real latency."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate_full(model, processor, loader, device):
    """Extended mAP evaluation with per-class breakdown."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("ERROR: pip install pycocotools"); return {}

    model.eval()
    preds, gts = [], []
    iid = 0
    latencies = []

    for batch in tqdm(loader, desc="Evaluating"):
        pv = batch["pixel_values"].to(device)
        pm = batch["pixel_mask"].to(device)

        _sync(device)
        t0 = time.perf_counter()
        out = model(pixel_values=pv, pixel_mask=pm)
        _sync(device)
        # Model forward only (excludes image decoding, resizing and post-processing)
        latencies.append((time.perf_counter() - t0) / pv.shape[0])

        # Post-process the whole batch (expects the model output object, not a dict)
        ts = torch.stack([lab["orig_size"] for lab in batch["labels"]]).to(device)
        batch_res = processor.post_process_object_detection(out, target_sizes=ts, threshold=0.01)

        for lab, res in zip(batch["labels"], batch_res):
            iid += 1
            orig = lab["orig_size"]

            for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"]):
                x1, y1, x2, y2 = bx.tolist()
                preds.append({
                    "image_id": iid,
                    "category_id": lb.item(),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": sc.item(),
                })

            h, w = orig.tolist()
            for j in range(len(lab["class_labels"])):
                cx, cy, bw, bh = lab["boxes"][j].tolist()
                gts.append({
                    "id": len(gts) + 1,
                    "image_id": iid,
                    "category_id": lab["class_labels"][j].item(),
                    "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                    "area": bw * w * bh * h,
                    "iscrowd": 0,
                })

    if not preds or not gts:
        print("No predictions or ground truths!"); return {}

    # Build COCO objects
    gt_coco = COCO()
    gt_coco.dataset = { # type: ignore
        "images": [{"id": i} for i in sorted({g["image_id"] for g in gts})],
        "annotations": gts,
        "categories": [{"id": c} for c in sorted({g["category_id"] for g in gts})],
    }
    gt_coco.createIndex()
    dt_coco = gt_coco.loadRes(preds) # type: ignore

    # ── Overall mAP ──
    ev = COCOeval(gt_coco, dt_coco, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    results = {
        "overall": {
            "mAP@50:95": ev.stats[0],
            "mAP@50": ev.stats[1],
            "mAP@75": ev.stats[2],
            "mAP_small": ev.stats[3],
            "mAP_medium": ev.stats[4],
            "mAP_large": ev.stats[5],
            "AR@1": ev.stats[6],
            "AR@10": ev.stats[7],
            "AR@100": ev.stats[8],
        },
    }

    # ── Per-Class AP ──
    results["per_class"] = {}
    cat_names = {int(k): v for k, v in model.config.id2label.items()}
    # Count instances per class
    class_counts = {}
    for g in gts:
        cid = g["category_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1

    for cat_id in sorted(gt_coco.getCatIds()):
        ev_c = COCOeval(gt_coco, dt_coco, "bbox")
        ev_c.params.catIds = [cat_id]
        ev_c.evaluate()
        ev_c.accumulate()
        ev_c.summarize()

        name = cat_names.get(cat_id, f"class_{cat_id}")
        results["per_class"][name] = {
            "category_id": cat_id,
            "AP@50:95": ev_c.stats[0],
            "AP@50": ev_c.stats[1],
            "AP@75": ev_c.stats[2],
            "instances": class_counts.get(cat_id, 0),
        }

    # ── Speed ──
    results["speed"] = {
        "note": "model forward pass only, batched; end-to-end FPS is lower",
        "avg_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "fps": 1.0 / np.mean(latencies) if latencies else 0,
        "device": str(device),
        "total_images": iid,
    }

    return results


def print_tables(results):
    """Print formatted tables ready for the paper."""
    
    print("\n" + "=" * 70)
    print("TABLE 1: Overall Detection Performance")
    print("=" * 70)
    o = results["overall"]
    print(f"  mAP@50:95  = {o['mAP@50:95']:.4f}")
    print(f"  mAP@50     = {o['mAP@50']:.4f}")
    print(f"  mAP@75     = {o['mAP@75']:.4f}")
    print(f"  mAP_small  = {o['mAP_small']:.4f}")
    print(f"  mAP_medium = {o['mAP_medium']:.4f}")
    print(f"  mAP_large  = {o['mAP_large']:.4f}")

    print("\n" + "=" * 70)
    print("TABLE 2: Per-Class Average Precision")
    print("=" * 70)
    print(f"  {'Class':<12s} {'AP@50':>8s} {'AP@75':>8s} {'AP@50:95':>10s} {'Instances':>10s}")
    print("  " + "-" * 52)
    for name, vals in results["per_class"].items():
        print(f"  {name:<12s} {vals['AP@50']:>8.4f} {vals['AP@75']:>8.4f} "
              f"{vals['AP@50:95']:>10.4f} {vals['instances']:>10,d}")

    print("\n" + "=" * 70)
    print("TABLE 3: Inference Speed")
    print("=" * 70)
    s = results["speed"]
    print(f"  Device:      {s['device']}")
    print(f"  Latency:     {s['avg_latency_ms']:.1f} ± {s['std_latency_ms']:.1f} ms")
    print(f"  FPS:         {s['fps']:.1f}  (forward pass only)")
    print(f"  Images:      {s['total_images']}")

    # LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE (copy-paste to paper):")
    print("=" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Per-class detection performance on UA-DETRAC test set}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Class & AP@50 & AP@75 & AP@50:95 & Instances \\")
    print(r"\hline")
    for name, vals in results["per_class"].items():
        print(f"{name} & {vals['AP@50']:.3f} & {vals['AP@75']:.3f} & "
              f"{vals['AP@50:95']:.3f} & {vals['instances']:,d} \\\\")
    o = results["overall"]
    print(r"\hline")
    print(f"Overall & {o['mAP@50']:.3f} & {o['mAP@75']:.3f} & {o['mAP@50:95']:.3f} & — \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--output", default="outputs/validation")
    args = ap.parse_args()

    cfg, log, device = bootstrap(args.config)
    model = load_best_or_final(cfg, device)
    processor = get_processor(cfg)
    # Ground truth is mapped into the loaded model's own label space
    _, val_ld, test_ld = get_dataloaders(cfg, processor, label2id=model.config.label2id)
    loader = val_ld if args.split == "val" else test_ld

    if loader is None:
        print(f"ERROR: No {args.split}.json found in annotations dir!")
        return

    print(f"🔬 Evaluating on {args.split} set...")
    results = evaluate_full(model, processor, loader, device)

    print_tables(results)

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f"detection_{args.split}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved to {path}")


if __name__ == "__main__":
    main()